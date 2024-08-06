import asyncio
from pathlib import Path

from src.oncobot.custom_chat_model import (
    CustomChatGroq,
    CustomChatHuggingFace,
    CustomChatLlamaReplicate,
    CustomChatOpenAI,
)
from src.oncobot.transcription import WhisperSTT

# checkpoint = "facebook/opt-125m"
# checkpoint = "meta-llama/Llama-2-7b-chat-hf"
checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"


def test_hf_model():
    model = CustomChatHuggingFace(checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in await model.astream(messages):
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_llama_model():
    model = CustomChatLlamaReplicate(checkpoint)
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):  # type: ignore
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_openai_model():
    model = CustomChatOpenAI()
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):  # type: ignore
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_groq_model():
    model = CustomChatGroq()
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "An increasing sequence from 1 to 10:",
        },
    ]
    print("Invoke:")
    print(model.invoke(messages))

    print("Stream:")
    for token in model.stream(messages):
        print(token, end="", flush=True)
    print()

    print("Invoke with async:")

    async def invoke_async():
        print(await model.ainvoke(messages))

    asyncio.run(invoke_async())

    print("Stream with async:")

    async def stream_async():
        async for token in model.astream(messages):  # type: ignore
            print(token, end="", flush=True)
        print()

    asyncio.run(stream_async())


def test_transcription():
    transcribe = WhisperSTT()
    audio_path = Path(__file__).resolve().parent / "audio" / "gta-san-andreas.mp3"
    print("Transcribing audio file")
    text = transcribe.run(str(audio_path))
    print(text)
