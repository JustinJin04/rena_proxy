import json
import asyncio
import fastapi
import httpx
import traceback
import uvicorn
from dataclasses import dataclass
from collections import defaultdict
from math import inf
import argparse


@dataclass
class ProxyConfig:
    port: int
    model_to_port: dict[str, int]
    classifier_name: str
    tool_cap: list
    rena_log_path: str

    @staticmethod
    def load_from_file(config_path: str) -> 'ProxyConfig':
        with open(config_path, 'r') as f:
            data = json.load(f)
        return ProxyConfig(
            port=data.get("port", 8030), 
            model_to_port=data.get("model_to_port", {}), 
            classifier_name=data.get("classifier_name", "unmask_75"),
            tool_cap=data.get("tool_cap", []),
            rena_log_path=data.get("rena_log_path", "log.jsonl")
        )


class Proxier:
    def __init__(self, config: ProxyConfig):
        self.config = config

    def _get_most_occurance_tool_name(self, response: dict) -> str:
        def extract_tool_name_from_content(content: str) -> str:
            if "<tool_call>" in content and "</tool_call>" in content:
                start = content.index("<tool_call>") + len("<tool_call>")
                end = content.index("</tool_call>")
                return json.loads(content[start:end])["name"]
            return ""
        
        tool_calls = []
        for choice in response.get("choices", []):
            if choice.get("message").get("tool_calls"):
                tool_calls.append(choice["message"]["tool_calls"][0]["function"]["name"])
            else:
                tool_calls.append(extract_tool_name_from_content(choice["message"].get("content", "")))
        counts = defaultdict(int)
        first_pos = {}
        for i, name in enumerate(tool_calls):
            counts[name] += 1
            if name not in first_pos:
                first_pos[name] = i  # remember first appearance
        most_tool, most_count, earliest = None, 0, inf
        for name, cnt in counts.items():
            pos = first_pos[name]
            if cnt > most_count or (cnt == most_count and pos < earliest):
                most_tool, most_count, earliest = name, cnt, pos
        
        return most_tool

    def tool_cap(self, raw_req_payload: dict) -> dict:
        """raw_req_payload: raw request from rena (wo tool cap)
        return: after tool cap"""
        
        req_payload = raw_req_payload.copy()
        req_payload["tools"] = self.config.tool_cap
        return req_payload

    def append_log(self, log_json: dict):
        with open(self.config.rena_log_path, 'a') as f:
            f.write(json.dumps(log_json) + "\n")

    async def classify(self, req_payload: dict) -> str:
        """req_payload: after tool cap
        return: tool_adapter name"""
        
        classifier_port = self.config.model_to_port.get(self.config.classifier_name)
        assert classifier_port is not None, f"Classifier {self.config.classifier_name} not found in model_to_port mapping."
        req_payload["model"] = self.config.classifier_name
        url = f"http://localhost:{classifier_port}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=req_payload)
        
        print(f"Response: {response.json()}")
        tool_name = self._get_most_occurance_tool_name(response.json())
        print(f"Classified tool name: {tool_name}")
        return tool_name

    async def tool_adaption(self, req_payload: dict, tool_name: str) -> httpx.Response:
        """req_payload: after tool cap
        return: response in json"""

        model_name = tool_name
        if model_name == "summarize":
            model_name = "unsloth/Qwen2.5-7B-Instruct"
        elif model_name == "list_allowed_directories":
            response = httpx.Response(200, json={
                "id": "xxxxxxxxxxxx",  # just random id
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                            "name": "list_allowed_directories",
                            "arguments": "{}"
                            }
                        }],
                    }
                }]
            })
            return response


        tool_adapter_port = self.config.model_to_port.get(model_name)
        assert tool_adapter_port is not None, f"Tool adapter {model_name} not found in model_to_port mapping."
        req_payload["model"] = model_name
        url = f"http://localhost:{tool_adapter_port}/v1/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=req_payload)

        if tool_name == "summarize":
            messages = req_payload.get("messages").copy()
            messages.append(response.json().get("choices")[0].get("message"))
            self.append_log({
                "messages": messages
            })

        return response


app = fastapi.FastAPI()
proxier = None


@app.post("/v1/chat/completions")
async def chat_completions(request: fastapi.Request):
    try:
        raw_req_payload = await request.json()
        req_payload = proxier.tool_cap(raw_req_payload)
        tool_name = await proxier.classify(req_payload)
        response = await proxier.tool_adaption(req_payload, tool_name)
        return fastapi.responses.JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )
    except Exception as e:
        traceback.print_exc()
        return fastapi.responses.JSONResponse(
            status_code=500, content={"error": str(e)}
        )


def get_args():
    parser = argparse.ArgumentParser(description="VLLM Proxier")
    parser.add_argument("--config_path", type=str, default="config.json", help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = ProxyConfig.load_from_file(args.config_path)
    proxier = Proxier(config=config)
    uvicorn_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=config.port,
        log_level="info",
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    try:
        asyncio.run(uvicorn_server.serve())
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except asyncio.CancelledError:
        print("Server cancelled.")
