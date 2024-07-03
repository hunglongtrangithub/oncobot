"""Main entrypoint for the app."""

import time
import asyncio
import os
import torch

from typing import AsyncGenerator
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from pathlib import Path
import json


from logger_config import get_logger
from retriever import CustomRetriever
from custom_chat_model import DummyChat, CustomChatHuggingFace
from ner import NERProcessor
from rag_chain import ChatRequest, RAGChain
from tts import DummyTTS, XTTS
from transcription import DummyOpenAIWhisperSTT, WhisperSTT
from talking_face import DummyTalker, CustomSadTalker
from config import settings

logger = get_logger(__name__)


chat_model = CustomChatHuggingFace(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    device="cuda:0",
    max_chat_length=2048,
)
# default_message = "Fake Patient 3 is diagnosed with stage 2 invasive ductal carcinoma of the right breast, metastatic to right axillary lymph nodes."
# chat_model = DummyChat(default_message=default_message)
retriever = CustomRetriever(num_docs=5, semantic_ratio=0.1)
ner = NERProcessor(device="cuda:0")
chain = RAGChain(retriever, chat_model, ner)
tts = XTTS(use_deepspeed=True)
transcribe = WhisperSTT(device="cuda:4")
# transcribe = DummyOpenAIWhisperSTT()
talker = CustomSadTalker(
    # batch_size=75,
    # device=[1, 2, 4],
    # parallel_mode="dp",
    batch_size=75,
    device="cuda:3",
    dtype=torch.float16,
)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


def post_processing(op, path, chunk):
    return json.dumps(
        {"ops": [{"op": op, "path": path, "value": chunk}]}, separators=(",", ":")
    )


async def astream_generator(subscription: AsyncGenerator):
    try:
        async for op, path, chunk in subscription:
            if path == "/stream-output/-":
                print(f"op: {op}, path: {path}, chunk: {chunk}")
            yield f"event: data\ndata: {post_processing(op, path, chunk)}\n\n"
            # HACK: This is a temporary fix to prevent the browser from being unable to handle the stream when it becomes too fast
            await asyncio.sleep(0.01)
    except asyncio.TimeoutError:
        error_message = "Stream timed out"
        logger.error(error_message)
        yield f"event: error\ndata: {error_message}\n\n"
    except Exception as e:
        error_message = f"Internal server error from endpoint /chat/astream_log: {e}"
        logger.error(error_message)
        yield f"event: error\ndata: {error_message}\n\n"
    finally:
        await subscription.aclose()
        yield "event: end\n\n\n"


@app.post("/chat/ainvoke_log")
async def chat(request: ChatRequest):
    try:
        response = await chain.ainvoke_log(request)
        return response
    except Exception as e:
        error_message = f"Internal server error from endpoint /chat/ainvoke_log: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


@app.post("/chat/astream_log")
async def achat(request: ChatRequest):
    subscription = chain.astream_log(request)
    stream = astream_generator(subscription)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
    )


@app.post("/transcribe_audio")
async def transcribe_audio(
    user_audio_file: UploadFile = File(...),
    conversationId: str = Form(...),
):
    file_name = f"{conversationId}.mp3"
    # save to local file
    upload_folder = Path(__file__).resolve().parent / "audio"
    upload_folder.mkdir(exist_ok=True)
    file_path = str(upload_folder / file_name)

    with open(file_path, "wb") as f:
        f.write(user_audio_file.file.read())
    try:
        transcript = await transcribe.arun(audio_path=file_path)
        return JSONResponse(
            content={"transcript": transcript},
            background=BackgroundTask(
                delete_file,
                file_path,
            ),
        )
    except Exception as e:
        error_message = f"Internal server error from endpoint /transcribe_audio: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )


def delete_file(*file_paths: str):
    """Deletes a file from the filesystem."""
    for file_path in file_paths:
        try:
            os.remove(file_path)
            logger.info(f"Deleted file {file_path}")
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")


@app.post("/text_to_speech")
async def text_to_speech(
    bot_voice_file: UploadFile = File(...),
    message: str = Form(...),
    conversationId: str = Form(...),
    chatbot: str = Form(...),
):
    start = time.time()
    speech_file_name = f"{conversationId}.wav"
    speech_folder = Path(__file__).resolve().parent / "audio"
    speech_folder.mkdir(exist_ok=True)
    speech_file_path = str(speech_folder / speech_file_name)

    voice_folder = Path(__file__).resolve().parent / "voices"
    voice_folder.mkdir(exist_ok=True)
    voice_file_path = str(voice_folder / f"{chatbot}.mp3")
    with open(voice_file_path, "wb") as f:
        f.write(bot_voice_file.file.read())

    try:
        await tts.arun(
            text=message,
            file_path=speech_file_path,
            voice_path=voice_file_path,
        )
        return FileResponse(
            speech_file_path,
            background=BackgroundTask(
                delete_file,
                speech_file_path,
                voice_file_path,
            ),
        )
    except Exception as e:
        error_message = f"Internal server error from endpoint /text_to_speech: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )
    finally:
        logger.info(f"/text_to_speech Total time taken: {time.time() - start}")


# TODO: Change this to text to video. Call tts and talker in sequence, then return the video file
@app.post("/text_to_video")
async def text_to_video(
    bot_voice_file: UploadFile = File(...),
    bot_image_file: UploadFile = File(...),
    message: str = Form(...),
    conversationId: str = Form(...),
    chatbot: str = Form(...),
):
    start = time.time()
    voice_folder = Path(__file__).resolve().parent / "voices"
    voice_folder.mkdir(exist_ok=True)
    voice_file_path = str(voice_folder / f"{chatbot}.mp3")
    with open(voice_file_path, "wb") as f:
        f.write(bot_voice_file.file.read())

    face_folder = Path(__file__).resolve().parent / "faces"
    face_folder.mkdir(exist_ok=True)
    face_file_path = str(face_folder / f"{chatbot}.jpg")
    with open(face_file_path, "wb") as f:
        f.write(bot_image_file.file.read())

    speech_file_name = f"{conversationId}.wav"
    speech_folder = Path(__file__).resolve().parent / "audio"
    speech_folder.mkdir(exist_ok=True)
    speech_file_path = str(speech_folder / speech_file_name)

    video_file_name = f"{chatbot}__{conversationId}.mp4"
    video_folder = Path(__file__).resolve().parent / "video"
    video_folder.mkdir(exist_ok=True)
    video_file_path = str(video_folder / video_file_name)

    try:
        await tts.arun(
            text=message,
            file_path=speech_file_path,
            voice_path=voice_file_path,
        )
        await talker.arun(
            video_path=video_file_path,
            audio_path=speech_file_path,
            image_path=face_file_path,
        )
        return FileResponse(
            video_file_path,
            background=BackgroundTask(
                delete_file,
                speech_file_path,
                voice_file_path,
                face_file_path,
                video_file_path,
            ),
        )
    except Exception as e:
        error_message = f"Internal server error from endpoint /text_to_video: {e}"
        logger.error(error_message)
        raise HTTPException(
            status_code=500,
            detail=error_message,
        )
    finally:
        logger.info(f"/text_to_video Total time taken: {time.time() - start}")


if __name__ == "__main__":
    import uvicorn

    port = settings.port
    uvicorn.run(app, host="0.0.0.0", port=port)
