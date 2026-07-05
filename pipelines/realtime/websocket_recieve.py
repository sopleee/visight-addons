from fastapi.security import HTTPBearer

import modal

image = modal.Image.debian_slim().pip_install("fastapi[standard]")
app = modal.App("websocket-ex", image=image)

auth_scheme = HTTPBearer()
web_token = modal.Secret.from_name("web-access-keys", 
                                   required_keys=["WEBSOCKET_SIMPLE"])

@app.function()
@modal.asgi_app()
def sample_endpt(): 
    from fastapi import FastAPI, WebSocket
    app = FastAPI()
    
    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None: 
        frame_i = 0
        await websocket.accept()
        while True: 
            data = await websocket.receive_text()
            await websocket.send_text(f"Received frame: {frame_i}")
            # await websocket.send_json({"status": "success", "message": data})
            frame_i+=1

    return app