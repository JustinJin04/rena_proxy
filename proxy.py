import json
import asyncio
import fastapi
import httpx
import traceback
import uvicorn
from dataclasses import dataclass


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
