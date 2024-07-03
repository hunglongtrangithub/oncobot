import time
from typing import Optional, Literal
import scipy
import shutil
from pathlib import Path
import requests
from typing import BinaryIO
import os
import asyncio

from openai import OpenAI, AsyncOpenAI
import replicate

from langdetect import detect
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio.numpy_transforms import save_wav

from transformers import AutoProcessor, BarkModel
import torch

from logger_config import get_logger
from config import settings

logger = get_logger(__name__)


def try_create_directory(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create directory {path}.")
        raise
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def try_open_audio_file(file_path: Path) -> BinaryIO:
    try:
        return open(file_path, "rb")
    except Exception as e:
        logger.error(f"Failed to open file at {file_path}: {e}")
        raise


class XTTS:
    def __init__(self, model_name: Optional[str] = None, use_deepspeed: bool = False):
        self.model_name = (
            "tts_models/multilingual/multi-dataset/xtts_v2"
            if model_name is None
            else model_name
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts_api = TTS()
        # Set environment variable to agree to the terms of service
        os.environ["COQUI_TOS_AGREED"] = "1"
        model_path, _, _ = self.tts_api.manager.download_model(self.model_name)

        self.config = XttsConfig()
        self.config.load_json(os.path.join(model_path, "config.json"))
        self.output_sample_rate = self.config.audio["output_sample_rate"] or 22050
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config,
            checkpoint_dir=model_path,
            eval=True,
            use_deepspeed=use_deepspeed,
        )
        self.model.to(self.device)

        logger.info(
            f"{self.model_name} initialized on device {self.device}. Using DeepSpeed: {use_deepspeed}."
        )
        logger.info(f"Model size: {self.get_model_size(self.model, 'GB'):.3f} GB.")
        logger.info(f"Using output sample rate of {self.output_sample_rate} Hz.")

    def get_model_size(self, model, unit="mb"):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all = param_size + buffer_size
        if unit.lower() == "mb":
            size_all /= 1024**2
        elif unit.lower() == "gb":
            size_all /= 1024**3
        return size_all

    def run(self, text: str, file_path: str, voice_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            lang_iso = detect(text)
            if lang_iso not in self.config.languages:
                lang_iso = "en"
        except Exception as e:
            logger.error(f"Failed to detect language for text: {e}. Defaulting to 'en'")
            lang_iso = "en"
        try:
            start = time.time()
            outputs = self.model.synthesize(
                text=text,
                config=self.config,
                speaker_wav=voice_path,
                language=lang_iso,
            )
            save_wav(
                wav=outputs["wav"],
                path=file_path,
                sample_rate=self.output_sample_rate,
            )
        except Exception as e:
            logger.error(f"Error in XTTS method: {e}", exc_info=True)
            raise
        logger.info(f"XTTS run method took {time.time() - start:.2f} seconds.")

    async def arun(self, text: str, file_path: str, voice_path: str):
        try:
            self.run(text, file_path, voice_path)
        except Exception as e:
            logger.error(f"Error in async XTTS method: {e}")
            raise

    def stream(self, text: str, voice_path: str, chunk_size: int = 16_384):
        try:
            lang_iso = detect(text)
            if lang_iso not in self.config.languages:
                lang_iso = "en"
        except Exception as e:
            logger.error(f"Failed to detect language for text: {e}. Defaulting to 'en'")
            lang_iso = "en"

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=voice_path,
            gpt_cond_len=self.config.gpt_cond_len,
            gpt_cond_chunk_len=self.config.gpt_cond_chunk_len,
            max_ref_length=self.config.max_ref_len,
            sound_norm_refs=self.config.sound_norm_refs,
        )
        streamer = self.model.inference_stream(
            text,
            lang_iso,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=chunk_size,
            enable_text_splitting=True,
        )
        return streamer


class OpenAITTS:
    def __init__(self, voice: str = "alloy"):
        self.api_key = self._get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        if voice not in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            raise ValueError(
                f"Invalid voice '{voice}' selected. Please select from: 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'."
            )
        self.voice = voice

    def _get_openai_api_key(self):
        return settings.openai_api_key.get_secret_value()

    def run(self, text: str, file_path: str, voice_path=None):
        if voice_path is not None:
            logger.warning(
                "OpenAI TTS does not support custom voices. Ignoring voice_path argument."
            )
        try_create_directory(Path(file_path).resolve().parent)
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,  # type: ignore
                input=text,
            )
            response.write_to_file(file_path)
        except Exception as e:
            logger.error(f"Error in OpenAI TTS method: {e}")
            raise

    async def arun(self, text: str, file_path: str, voice_path=None):
        if voice_path is not None:
            logger.warning(
                "OpenAI Async TTS does not support custom voices. Ignoring voice_path argument."
            )
        try_create_directory(Path(file_path).resolve().parent)
        try:
            response = await self.async_client.audio.speech.create(
                model="tts-1",
                voice=self.voice,  # type: ignore
                input=text,
            )
            response.write_to_file(file_path)
        except Exception as e:
            logger.error(f"Error in OpenAI Async TTS method: {e}")
            raise


class ReplicateTortoiseTTS:
    def __init__(self) -> None:
        self.replicate_id = "afiaka87/tortoise-tts:e9658de4b325863c4fcdc12d94bb7c9b54cbfe351b7ca1b36860008172b91c71"

    def run(self, text: str, file_path: str, voice_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        custom_voice = try_open_audio_file(Path(voice_path))
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }
        try:
            output_url = replicate.run(
                self.replicate_id,
                input=input,
            )
            r = requests.get(output_url)  # type: ignore
            with open(file_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logger.error(f"Failed to run ReplicateTortoiseTTS's run method: {e}")
            raise

    async def arun(self, text: str, file_path: str, voice_path: str):
        try_create_directory(Path(file_path).resolve().parent)
        custom_voice = try_open_audio_file(Path(voice_path))
        input = {
            "voice_a": "custom_voice",
            "custom_voice": custom_voice,
            "text": text,
            "preset": "fast",
        }
        try:
            output_url = await replicate.async_run(
                self.replicate_id,
                input=input,
            )
            r = requests.get(output_url)  # type: ignore
            with open(file_path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            logger.error(f"Failed to run ReplicateTortoiseTTS's arun method: {e}")
            raise


class BarkSuno:
    def __init__(self):
        self.voice_preset = "v2/en_speaker_6"
        self.model_name = "suno/bark-small"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = BarkModel.from_pretrained(self.model_name)

        logger.info(f"{self.model_name} initialized.")

    def run(self, text: str, file_path: str, voice_path: Optional[str] = None):
        if voice_path is not None:
            logger.warning(
                "BarkSuno TTS does not support custom voices. Ignoring voice_path argument."
            )
        try_create_directory(Path(file_path).resolve().parent)
        try:
            inputs = self.processor(
                text,
                voice_preset=self.voice_preset,
                return_tensors="pt",
            )
            audio_array = self.model.generate(**inputs)  # type: ignore
            audio_array = audio_array.cpu().numpy().squeeze()

            sampling_rate = model.generation_config.sample_rate  # type: ignore
            scipy.io.wavfile.write(
                file_path,
                rate=sampling_rate,
                data=audio_array,
            )
        except Exception as e:
            logger.error(f"Error in BarkSuno method: {e}")
            raise

    async def arun(self, text: str, file_path: str, voice_path: Optional[str] = None):
        if voice_path is not None:
            logger.warning(
                "BarkSuno Async TTS does not support custom voices. Ignoring voice_path argument."
            )
        try:
            self.run(text, file_path)
        except Exception as e:
            logger.error(f"Error in async BarkSuno method: {e}")
            raise


class DummyTTS:
    def __init__(self):
        logger.info("Dummy TTS initialized.")

    def run(self, text: str, file_path: str, voice_path: str):
        """Synchronously copy audio content to specified file path."""
        try_create_directory(Path(file_path).resolve().parent)
        try:
            # Copy the content of the voice file to the specified file path
            shutil.copy(voice_path, file_path)
            logger.info("Dummy TTS file copied successfully.")
        except Exception as e:
            logger.error(f"Error in Dummy TTS method: {e}")
            raise

    async def arun(self, text: str, file_path: str, voice_path: str):
        """Asynchronously copy audio content to specified file path."""
        try_create_directory(Path(file_path).resolve().parent)
        try:
            # Simulate async operation, if needed
            await asyncio.sleep(0)  # No delay needed, just for async syntax

            # Copy the content of the voice file to the specified file path
            shutil.copy(voice_path, file_path)
            logger.info("Dummy Async TTS file copied successfully.")
        except Exception as e:
            logger.error(f"Error in Dummy Async TTS method: {e}")
            raise


if __name__ == "__main__":
    pass
