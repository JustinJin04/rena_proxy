import os
import json
import copy
import threading
import argparse
from dataclasses import dataclass
from math import inf
from typing import Optional
import fastapi
import httpx
import traceback
import uvicorn
from pathlib import Path

from .classifier import get_classifier
from .tool_adaptor import get_tool_adaptor
from .utils.logging_setup import setup_logging

import logging
logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

@dataclass
class ProxyConfig:
    port: int = 8030
    tool_cap: Optional[str] = None  # tool_cap json or none (no tool_cap)
    classifier_name_or_path: str = "gpt"  # path to classifier config or name
    tool_adaptor_name_or_path: str = "gpt"  # path to tool adaptor config or name
    tool_list: str = ""
    error_queries_log_path: Optional[str] = None

class Proxier:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = fastapi.FastAPI()
        self._register_routes()

        if self.config.tool_cap:
            with open(self.config.tool_cap, "r") as f:
                self.config.tool_cap = json.load(f)
        self.tool_list = json.load(open(self.config.tool_list, "r"))
        assert isinstance(self.tool_list, list), "tool_list should be a list of tools"
        self.classifier = get_classifier(config.classifier_name_or_path)
        self.tool_adaptor = get_tool_adaptor(config.tool_adaptor_name_or_path)

        # server相关
        self._server = None
        self._thread = None

    def __enter__(self):
        """启动uvicorn server (后台线程)"""
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.config.port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)

        def run_server():
            # run 会阻塞，所以放在线程里
            import asyncio
            asyncio.run(self._server.serve())

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        logger.info(f"Proxier started at http://0.0.0.0:{self.config.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """关闭server"""
        if self._server and self._server.started:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Proxier stopped.")

    def tool_cap(self, raw_req_payload: dict) -> dict:
        req_copy = copy.deepcopy(raw_req_payload)

        # tool_capabilities
        tools = req_copy["tools"]
        tool_caps = (getattr(self.config, "tool_cap", None) and self.config.tool_cap.get("tool_capabilities", {})) or {}
        for tool in tools:
            fn = tool.get("function")
            name = fn.get("name")
            cap = tool_caps.get(name)
            if cap:
                fn.update(copy.deepcopy(cap))
                tool["function"] = fn
                print(f"pppppppppproxy: {tool['function']}")

        # _workflow_patterns
        req_copy["_workflow_patterns"] = (getattr(self.config, "tool_cap", None) and self.config.tool_cap.get("_workflow_patterns", [])) or []
        req_copy["tool_selection_guidelines"] = (getattr(self.config, "tool_cap", None) and self.config.tool_cap.get("tool_selection_guidelines", "")) or ""

        return req_copy
    
    async def classify(self, req_payload: dict) -> str:
        return await self.classifier.classify(req_payload)
    
    async def tool_adaption(self, req_payload: dict, tool_name: str) -> httpx.Response:
        return await self.tool_adaptor.adapt(req_payload, tool_name)
    
    def _register_routes(self):
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: fastapi.Request):
            try:
                raw_req_payload = await request.json()
                
                # Used for substitue tool names. Note that it should be done before tool_cap
                raw_req_payload["tools"] = self.tool_list
                
                logger.info(f"Received request messages: {json.dumps(raw_req_payload['messages'])}")
                req_payload = self.tool_cap(raw_req_payload)
                tool_name = await self.classify(req_payload)
                logger.info(f"Classified tool name: {tool_name}")
                response = await self.tool_adaption(req_payload, tool_name)

                # used for substitute tool names
                Substitute_tool_list = {
                    "get_users_by_name": "list_users_and_teams"
                }
                response_json = response.json()
                for tool_call in response_json["choices"][0]["message"].get("tool_calls", []):
                    if tool_call["function"]["name"] in Substitute_tool_list:
                        tool_call["function"]["name"] = Substitute_tool_list[tool_call["function"]["name"]]
                response = httpx.Response(response.status_code, json=response_json)



                logger.info(f"Tool adaptation response: {json.dumps(response.json())}")
                return fastapi.responses.JSONResponse(
                    status_code=response.status_code,
                    content=response.json()
                )
            except Exception as e:
                logger.error(f"Error when processing request: {json.dumps(raw_req_payload)}")
                logger.error(f"Exception occurred: {str(e)}")
                traceback.print_exc()
                if self.config.error_queries_log_path:
                    with open(self.config.error_queries_log_path, "a") as f:
                        error_log = {
                            "request": raw_req_payload,
                            "error": str(e)
                        }
                        f.write(json.dumps(error_log) + "\n")
                return fastapi.responses.JSONResponse(
                    status_code=500, content={"error": str(e)}
                )
                

def start_proxy(port: int, tool_name: str, prompt_tuning: bool, classifier: bool, tool_adapters: bool, tool_capabilities: bool, logging_dir: Optional[str] = None, error_queries_log_path: Optional[str] = None) -> Proxier:
    classifier_name_or_path = str(PACKAGE_ROOT / "config" / tool_name / "classifier.json") if classifier else "gpt"
    tool_adaptor_name_or_path = str(PACKAGE_ROOT / "config" / tool_name / "tool_adaptor.json") if tool_adapters else "gpt"
    tool_cap = str(PACKAGE_ROOT / "config" / tool_name / "tool_cap_augmented.json") if tool_capabilities else None
    tool_list = str(PACKAGE_ROOT / "config" / tool_name / "tool_list.json")

    if logging_dir:
        log_file_path = str(Path(logging_dir)/ f"{prompt_tuning}{classifier}{tool_adapters}{tool_capabilities}.log")
    else:
        log_file_path = str(PACKAGE_ROOT / "logs" / tool_name / f"{prompt_tuning}{classifier}{tool_adapters}{tool_capabilities}.log")

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    print(f"log_file_path: {log_file_path}")
    # with open(log_file_path, "w") as f:
    #     f.write("")
    if error_queries_log_path:
        os.makedirs(os.path.dirname(error_queries_log_path), exist_ok=True)
        with open(error_queries_log_path, "w") as f:
            f.write("")
    setup_logging(str(log_file_path))

    config = ProxyConfig(
        port=port,
        classifier_name_or_path=classifier_name_or_path,
        tool_adaptor_name_or_path=tool_adaptor_name_or_path,
        tool_cap=tool_cap,
        tool_list=tool_list,
        error_queries_log_path=error_queries_log_path
    )
    return Proxier(config)
