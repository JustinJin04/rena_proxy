import os
import json
import argparse
from dataclasses import dataclass
from math import inf
from typing import Optional
import fastapi
import httpx
import traceback
import uvicorn

from classifier import get_classifier
from tool_adaptor import get_tool_adaptor
from utils.logging_setup import setup_logging

import logging
logger = logging.getLogger(__name__)

@dataclass
class ProxyConfig:
    port: int = 8030
    tool_cap: Optional[dict] = None  # tool_cap json or none (no tool_cap)
    classifier_name_or_path: str = "gpt"  # path to classifier config or name
    tool_adaptor_name_or_path: str = "gpt"  # path to tool adaptor config or name

class Proxier:
    def __init__(self, config: ProxyConfig):
        self.config = config

        self.app = fastapi.FastAPI()
        self._register_routes()

        if self.config.tool_cap:
            with open(self.config.tool_cap, "r") as f:
                self.config.tool_cap = json.load(f)
        self.classifier = get_classifier(config.classifier_name_or_path)
        self.tool_adaptor = get_tool_adaptor(config.tool_adaptor_name_or_path)
    
    def tool_cap(self, raw_req_payload: dict) -> dict:
        req_copy = raw_req_payload.copy()
        if self.config.tool_cap:
            req_copy["tools"] = self.config.tool_cap["tool_cap"]
        return req_copy
    
    async def classify(self, req_payload: dict) -> str:
        return await self.classifier.classify(req_payload)
    
    async def tool_adaption(self, req_payload: dict, tool_name: str) -> httpx.Response:
        return await self.tool_adaptor.adapt(req_payload, tool_name)
    
    def _register_routes(self):
        @self.app.post("/proxy/chat/completions")
        async def chat_completions(request: fastapi.Request):
            try:
                raw_req_payload = await request.json()
                logger.info(f"Received request messages: {json.dumps(raw_req_payload['messages'])}")
                req_payload = self.tool_cap(raw_req_payload)
                tool_name = await self.classify(req_payload)
                logger.info(f"Classified tool name: {tool_name}")
                response = await self.tool_adaption(req_payload, tool_name)
                logger.info(f"Tool adaptation response: {json.dumps(response.json())}")
                return fastapi.responses.JSONResponse(
                    status_code=response.status_code,
                    content=response.json()
                )
            except Exception as e:
                traceback.print_exc()
                return fastapi.responses.JSONResponse(
                    status_code=500, content={"error": str(e)}
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--tool_cap", type=str, default=None)
    parser.add_argument("--classifier_name_or_path", type=str, default="gpt")
    parser.add_argument("--tool_adaptor_name_or_path", type=str, default="gpt")
    parser.add_argument("--log_file_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.log_file_path)):
        os.makedirs(os.path.dirname(args.log_file_path))
    with open(args.log_file_path, "w") as f:
        f.write("")
    setup_logging(args.log_file_path)

    config = ProxyConfig(
        port=args.port,
        tool_cap=args.tool_cap,
        classifier_name_or_path=args.classifier_name_or_path,
        tool_adaptor_name_or_path=args.tool_adaptor_name_or_path
    )

    proxier = Proxier(config)
    uvicorn.run(proxier.app, host="0.0.0.0", port=config.port)

