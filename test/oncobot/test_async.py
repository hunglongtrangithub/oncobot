import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient


async def retrieve_docs():
    await asyncio.sleep(5)  # Simulate document retrieval delay
    formatted_docs = "<retrieved docs>"
    yield "add", "/logs", {}
    yield "add", "/logs/FindDocs", {}
    yield "add", "/logs/FindDocs/final_output", {"output": formatted_docs}


async def chat_response():
    await asyncio.sleep(3)  # Simulate initial delay
    response = ""
    yield "add", "/streamed_output", []
    for chunk in range(20):
        chunk = f"|{chunk}|"
        yield "add", "/streamed_output/-", chunk
        response += str(chunk)
        await asyncio.sleep(0.1)  # Simulate delay between responses
    yield "replace", "/final_output", {"output": response}


async def stream_log():
    yield "replace", "", {}

    # Retrieve documents
    async for result in retrieve_docs():
        yield result

    # Iterate over chat_response yields
    async for result in chat_response():
        yield result


def post_processing(op, path, chunk):
    return json.dumps(
        {"ops": [{"op": op, "path": path, "value": chunk}]}, separators=(",", ":")
    )


async def stream_generator(subscription):
    try:
        async for op, path, chunk in subscription:
            yield f"event: data\ndata: {post_processing(op, path, chunk)}\n\n"
        yield "event: end\n\n\n"
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Stream timed out")
    finally:
        pass


def test_async():
    app = FastAPI()

    @app.post("/")
    async def endpoint():
        subscription = stream_log()
        return StreamingResponse(
            stream_generator(subscription), media_type="text/event-stream"
        )

    client = TestClient(app)
    response = client.post("/")  # Enable streaming

    for line in response.iter_lines():
        if line:
            print(line)
