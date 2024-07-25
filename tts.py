import time
from typing import Optional
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
import pysbd

from transformers import AutoProcessor, BarkModel
import torch
import numpy as np

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
    def __init__(self, use_deepspeed: bool = False):
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
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

        logger.info(f"Supported languages: {self.config.languages}")
        self.segs = []
        for lang in self.config.languages:
            try:
                segmenter = pysbd.Segmenter(language=lang)
            except Exception:
                logger.info(
                    f"Failed to load pysbd segmenter for language {lang}. Defaulting to 'en'."
                )
                segmenter = pysbd.Segmenter(language="en")
            self.segs.append(segmenter)

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

    def split_into_sentences(self, text: str, lang_iso: str):
        return self.segs[self.config.languages.index(lang_iso)].segment(text)

    def run(
        self, text: str, file_path: str, voice_path: str, split_sentences: bool = True
    ):
        try_create_directory(Path(file_path).resolve().parent)
        try:
            lang_iso = detect(text)
            if lang_iso not in self.config.languages:
                lang_iso = "en"
        except Exception as e:
            logger.error(f"Failed to detect language for text: {e}. Defaulting to 'en'")
            lang_iso = "en"

        sens = [text]
        if split_sentences:
            print(" > Text splitted to sentences.")
            sens = self.split_into_sentences(text, lang_iso)
        print(sens)

        start = time.time()
        wavs = []
        try:
            for sen in sens:
                outputs = self.model.synthesize(
                    text=sen,
                    config=self.config,
                    speaker_wav=voice_path,
                    language=lang_iso,
                )
                waveform = outputs["wav"]
                waveform = waveform.squeeze()
                wavs += list(waveform)
                wavs += [0] * 10000

            save_wav(
                wav=np.array(wavs),
                path=file_path,
                sample_rate=self.output_sample_rate,
            )
        except Exception as e:
            logger.error(f"Error in XTTS method: {e}", exc_info=True)
            raise
        # compute stats
        process_time = time.time() - start
        audio_time = len(wavs) / self.config.audio["sample_rate"]
        logger.info(f" > Processing time: {process_time}")
        logger.info(f" > Real-time factor: {process_time / audio_time}")
        logger.info(f"XTTS run method took {time.time() - start:.2f} seconds.")

    async def arun(self, text: str, file_path: str, voice_path: str):
        try:
            self.run(text, file_path, voice_path)
        except Exception as e:
            logger.error(f"Error in async XTTS method: {e}")
            raise


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

            sampling_rate = self.model.generation_config.sample_rate  # type: ignore
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
    def __init__(self, dummy_audio_file: str = "examples/chatbot1.mp3"):
        logger.info("Dummy TTS initialized.")
        self.dummy_audio_file = dummy_audio_file

    def run(self, text: str, file_path: str, voice_path: str):
        """Synchronously copy audio content to specified file path."""
        try_create_directory(Path(file_path).resolve().parent)
        try:
            # Copy the content of the dummy audio file to the specified file path
            shutil.copy(self.dummy_audio_file, file_path)
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

            shutil.copy(self.dummy_audio_file, file_path)
            logger.info("Dummy Async TTS file copied successfully.")
        except Exception as e:
            logger.error(f"Error in Dummy Async TTS method: {e}")
            raise


if __name__ == "__main__":
    pass
