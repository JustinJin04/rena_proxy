import os
import json
import httpx

import logging
logger = logging.getLogger(__name__)

class ToolAdaptor:
    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        """req_payload: after tool cap"""
        raise NotImplementedError

class FinetunedToolAdaptor(ToolAdaptor):

    def __init__(self, adaptor_dict: dict):
        self.adaptor_dict = adaptor_dict

    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        req_copy = req_payload.copy()
        model = self.adaptor_dict["tool_name"]["model"]
        port = self.adaptor_dict["tool_name"]["port"]
        req_copy["model"] = model
        url = f"http://localhost:{port}/v1/chat/completions"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=req_copy)
        return response

class GPTToolAdaptor(ToolAdaptor):

    async def adapt(self, req_payload: dict, tool_name: str) -> httpx.Response:
        req_copy = req_payload.copy()
        messages = req_copy.get("messages", [])
        tools = req_copy.get("tools", [])
        if tool_name == "summarize":
            messages.append({
                "role": "user",
                "content": f"You must summarize the content."
            })
            tools = []
        else:
            messages.append({
                "role": "user",
                "content": f"You must use the {tool_name} tool."
            })
            for tool in tools:
                if tool["function"]["name"] == tool_name:
                    tools = [tool]
                    break
        url = f"https://api.openai.com/v1/chat/completions"
        req_copy.pop("max_tokens", None)
        req_copy["model"] = "gpt-5-mini"
        req_copy["messages"] = messages
        req_copy["tools"] = tools
        api_key = os.environ.get("OPENAI_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        max_retries = 10
        tries = 0
        while True:
            tries += 1
            if tries > max_retries:
                logger.error("GPToolAdaptor: Max retries exceeded")
                raise RuntimeError("Max retries exceeded")
            async with httpx.AsyncClient(timeout=60.0, headers=headers) as client:
                response = await client.post(url, json=req_copy)
            try:
                if tool_name != "summarize":
                    response_tool = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
                    assert response_tool == tool_name, f"Expected tool name {tool_name}, but got {response_tool}."
            except Exception as e:
                logger.info(f"GPTToolAdaptor: Error occurred. Tries {tries}/{max_retries}")
            else:
                break
        return response

def get_tool_adaptor(adaptor_name_or_path) -> ToolAdaptor:
    """adaptor_name_or_path:
     - gpt
     - path to tool adaptor config
    """
    if adaptor_name_or_path == "gpt":
        return GPTToolAdaptor()
    else:
        with open(adaptor_name_or_path, "r") as f:
            config = json.load(f)
        return FinetunedToolAdaptor(config)
