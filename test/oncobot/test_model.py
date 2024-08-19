import asyncio
import pytest
from pathlib import Path
from src.oncobot.custom_chat_model import (
    CustomChatGroq,
    CustomChatHuggingFace,
    CustomChatLlamaReplicate,
    CustomChatOpenAI,
)
from src.oncobot.transcription import WhisperSTT


# Parametrized fixture for models, scoped to module
@pytest.fixture(
    scope="module",
    params=[
        CustomChatGroq,
        CustomChatHuggingFace,
        CustomChatLlamaReplicate,
        CustomChatOpenAI,
    ],
)
def model(request):
    model_class = request.param
    if model_class in [CustomChatHuggingFace, CustomChatLlamaReplicate]:
        checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
        return model_class(checkpoint)
    else:
        return model_class()


# Fixture for messages, scoped to module
@pytest.fixture(scope="module")
def messages():
    return [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]


# Test for synchronous invoke
def test_invoke(model, messages):
    print("Invoke:")
    print(model.invoke(messages))


# Test for synchronous stream
def test_stream(model, messages):
    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()


def run_async(test_func):
    def wrapper(*args, **kwargs):
        asyncio.run(test_func(*args, **kwargs))

    return wrapper


# Test for asynchronous invoke
@run_async
async def test_ainvoke(model, messages):
    print("Invoke with async:")
    print(await model.ainvoke(messages))


# Test for asynchronous stream
@run_async
async def test_astream(model, messages):
    print("Stream with async:")
    async for token in model.astream(messages):
        print(token, end="", flush=True)
    print()


# Test for transcription model (not included in model param fixture)
def test_transcription():
    transcribe = WhisperSTT()
    audio_path = Path(__file__).resolve().parents[2] / "examples" / "chatbot1.mp3"
    print("Transcribing audio file at:", audio_path)
    text = transcribe.run(str(audio_path))
    print(text)
