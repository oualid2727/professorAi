import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/ws/professor"

    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": "Explain the central limit theorem."
        }))

        while True:
            try:
                response = await websocket.recv()
                print(response)
            except Exception as e:
                print("Done:", e)
                break

asyncio.run(test())