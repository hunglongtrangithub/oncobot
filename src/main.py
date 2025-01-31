"""Main entrypoint for the app."""

import time
import asyncio
import os

from typing import AsyncGenerator
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.background import BackgroundTask

from pathlib import Path
import json


from src.utils.logger_config import logger
from src.utils.env_config import settings
from src.oncobot.retriever import CustomRetriever
from src.oncobot.custom_chat_model import CustomChatHuggingFace, DummyChat
from src.oncobot.ner import NERProcessor, DummyNERProcessor
from src.oncobot.rag_chain import ChatRequest, RAGChain
from src.oncobot.tts import XTTS, DummyTTS
from src.oncobot.transcription import WhisperSTT, DummyOpenAIWhisperSTT
from src.oncobot.talking_face import CustomSadTalker, DummyTalker, FakeTalker


# Load dummy models if in test mode
if os.environ.get("MODE") == "TEST":
    logger.info("Running in test mode")
    default_message = "Fake Patient 3 is diagnosed with stage 2 invasive ductal carcinoma of the right breast, metastatic to right axillary lymph nodes."
    chat_model = DummyChat(default_message=default_message)
    retriever = CustomRetriever(num_docs=5, semantic_ratio=0.1)
    ner = DummyNERProcessor()
    chain = RAGChain(retriever, chat_model, ner)
    tts = DummyTTS()
    transcribe = DummyOpenAIWhisperSTT()
    talker = DummyTalker()
else:
    chat_model = CustomChatHuggingFace(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda",
        max_chat_length=1024,
    )
    retriever = CustomRetriever(num_docs=5, semantic_ratio=0.1)
    ner = NERProcessor(device="cuda:1")
    chain = RAGChain(retriever, chat_model, ner)
    tts = XTTS(use_deepspeed=True)
    transcribe = WhisperSTT(device="cuda:1")
    # The comments below show a few options to configure the inference of the talker model.
    # The current settings works well on a 40GB NVIDIA A100 GPU.
    talker = CustomSadTalker(
        batch_size=50,
        device=[3, 4, 7],
        parallel_mode="dp",
        # torch_dtype="float16",
        # device=2,
        # batch_size=60,
        # quanto_weights="int8",
        # quanto_activations=None,
    )
    # talker = FakeTalker()


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


# TODO: Serve the docs by fetching from the database
app.mount(
    "/documents",  # cannot use docs as it is reserved for FastAPI
    StaticFiles(directory=Path(__file__).resolve().parent.parent / "docs"),
    name="documents",
)


def post_processing(op, path, chunk):
    return json.dumps(
        {"ops": [{"op": op, "path": path, "value": chunk}]}, separators=(",", ":")
    )


async def astream_generator(subscription: AsyncGenerator):
    try:
        async for op, path, chunk in subscription:
            if path == "/stream-output/-":
                logger.debug(f"op: {op}, path: {path}, chunk: {chunk}")
            yield f"event: data\ndata: {post_processing(op, path, chunk)}\n\n"
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
    start = time.perf_counter()
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
        logger.info(f"/text_to_speech Total time taken: {time.perf_counter() - start}")


@app.post("/text_to_video")
async def text_to_video(
    bot_voice_file: UploadFile = File(...),
    bot_image_file: UploadFile = File(...),
    message: str = Form(...),
    conversationId: str = Form(...),
    chatbot: str = Form(...),
):
    start = time.perf_counter()
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
                # video_file_path,
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
        logger.info(f"/text_to_video Total time taken: {time.perf_counter() - start}")


if __name__ == "__main__":
    import uvicorn

    port = settings.port
    uvicorn.run(app, host="0.0.0.0", port=port)
