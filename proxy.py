import json
import asyncio
import fastapi
import httpx
import traceback
import uvicorn
from dataclasses import dataclass
from collections import defaultdict
from math import inf

@dataclass
class ProxyConfig:
    port: int
    model_to_port: dict[str, int]
    tool_cap: list

    @staticmethod
    def load_from_file(config_path: str) -> 'ProxyConfig':
        with open(config_path, 'r') as f:
            data = json.load(f)
        return ProxyConfig(
            port=data.get("port", 8030), 
            model_to_port=data.get("model_to_port", {}), 
            tool_cap=data.get("tool_cap", [])
        )


config = ProxyConfig.load_from_file("config.json")
app = fastapi.FastAPI()


def handle_response(response: dict) -> dict:

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
    
    classifier_response_json = {"name": most_tool}
    response["choices"] = [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": f"<tool_call>\n{json.dumps(classifier_response_json)}\n</tool_call>",
        }
    }]
    return response


@app.post("/proxy/chat/completions")
async def chat_completions(request: fastapi.Request):
    try:
        payload = await request.json()
        model_name = payload.get("model")
        engine_port = config.model_to_port.get(model_name)
        engine_url = f"http://localhost:{engine_port}/v1/chat/completions"

        # Modify the tool lists
        payload["tools"] = config.tool_cap

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(engine_url, json=payload)
        
        # Modify the choices of resonses
        response_json = response.json()
        response_json = handle_response(response_json)
        
        return fastapi.responses.JSONResponse(
            status_code=response.status_code,
            content=response_json
        )
    except Exception as e:
        traceback.print_exc()
        return fastapi.responses.JSONResponse(
            status_code=500, content={"error": str(e)}
        )


if __name__ == "__main__":
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
