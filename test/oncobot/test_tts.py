from pathlib import Path

from src.oncobot.tts import XTTS, BarkSuno, OpenAITTS, ReplicateTortoiseTTS

voice_path = str(Path(__file__).parents[2] / "examples" / "chatbot1.mp3")
test_audio_path = Path(__file__).parent / "audio"
test_audio_path.mkdir(exist_ok=True)


def test_barksuno_run():
    tts = BarkSuno()
    text = "Hello, world!"
    file_path = test_audio_path / "hello-world.mp3"
    tts.run(text, str(file_path))


def test_coqui_tts_run():
    tts = XTTS(use_deepspeed=False)
    file_path = str(test_audio_path / "coqui_test_run.mp3")
    tts.run("Hello from XTTS run method.", file_path, voice_path)
    assert Path(file_path).exists()
    print(f"XTTS.run test passed, saved to {file_path}")


async def test_coqui_tts_arun():
    tts = XTTS()
    file_path = str(test_audio_path / "coqui_test_arun.mp3")
    await tts.arun("Hello from TTS async run method.", file_path, voice_path)
    assert Path(file_path).exists()
    print(f"XTTS.arun test passed, saved to {file_path}")


def test_openai_tts_run():
    tts = OpenAITTS()
    file_path = str(test_audio_path / "openai_test_run.mp3")
    tts.run("Hello from OpenAI TTS run method.", file_path)
    assert Path(file_path).exists()
    print(f"OpenAITTS.run test passed, saved to {file_path}")


async def test_openai_tts_arun():
    tts = OpenAITTS()
    file_path = str(test_audio_path / "openai_test_arun.mp3")
    await tts.arun("Hello from OpenAI TTS async run method.", file_path)
    assert Path(file_path).exists()
    print(f"OpenAITTS.arun test passed, saved to {file_path}")


def test_replicate_tortoise_tts_run():
    tts = ReplicateTortoiseTTS()
    file_path = str(Path(__file__).parent / "audio" / "replicate_test_run.mp3")

    tts.run("Hello from Replicate Tortoise TTS run method.", file_path, voice_path)
    assert Path(file_path).exists()
    print(f"ReplicateTortoiseTTS.run test passed, saved to {file_path}")


async def test_replicate_tortoise_tts_arun():
    tts = ReplicateTortoiseTTS()
    file_path = str(Path(__file__).parent / "audio" / "replicate_test_arun.mp3")
    await tts.arun(
        "Hello from Replicate Tortoise TTS async run method.", file_path, voice_path
    )
    assert Path(file_path).exists()
    print(f"ReplicateTortoiseTTS.arun test passed, saved to {file_path}")
