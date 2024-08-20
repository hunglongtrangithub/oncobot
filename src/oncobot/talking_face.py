import os
from pathlib import Path
import shutil

import torch
from optimum import quanto

from src.sadtalker import SadTalker
from src.utils.logger_config import get_logger

logger = get_logger(__name__)


# The async method in this class is just for demonstration purposes. It is not actually async.
class DummyTalker:
    def __init__(self, dummy_video_file: str = "examples/chatbot1.mp4"):
        logger.info("DummyTalker initialized")
        self.dummy_video_file = dummy_video_file

    def run(self, video_path: str, audio_path: str, image_path: str):
        shutil.copy(self.dummy_video_file, video_path)
        logger.info(f"Created video {video_path}")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


class CustomSadTalker(SadTalker):
    accepted_weights = {
        "float8": quanto.qfloat8,
        "int8": quanto.qint8,
        "int4": quanto.qint4,
        "int2": quanto.qint2,
    }
    accepted_activations = {
        "none": None,
        "int8": quanto.qint8,
        "float8": quanto.qfloat8,
    }
    accepted_dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
    }

    def __init__(
        self,
        batch_size=150,
        device=None,
        torch_dtype=None,
        parallel_mode=None,
        quanto_weights=None,
        quanto_activations=None,
    ):
        dtype, quanto_config = self._check_arguments(
            torch_dtype=torch_dtype,
            quanto_weights=quanto_weights,
            quanto_activations=quanto_activations,
        )
        logger.info(f"quanto_config: {quanto_config}")
        checkpoint_path = Path(__file__).parent.parent / "sadtalker/checkpoints"
        gfpgan_path = Path(__file__).parent.parent / "sadtalker/gfpgan/weights"
        config_path = Path(__file__).parent.parent / "sadtalker/src/config"
        super().__init__(
            checkpoint_path=str(checkpoint_path),
            gfpgan_path=str(gfpgan_path),
            config_path=str(config_path),
            device=device,
            dtype=dtype,
            parallel_mode=parallel_mode,
            **quanto_config,
        )
        self.batch_size = batch_size
        logger.info("CustomSadTalker initialized.")

    def _check_arguments(self, torch_dtype, quanto_weights, quanto_activations):
        if torch_dtype:
            if torch_dtype not in self.accepted_dtypes:
                raise ValueError(
                    f"Only support dtypes in {self.accepted_dtypes.keys()} but found {torch_dtype}"
                )
        dtype = self.accepted_dtypes[torch_dtype] if torch_dtype else None
        quanto_config = {}
        if quanto_weights:
            if quanto_weights not in self.accepted_weights:
                raise ValueError(
                    f"Only support weights in {self.accepted_weights.keys()} but found {quanto_weights}"
                )
            quanto_config["weights"] = self.accepted_weights[quanto_weights]
        if quanto_activations:
            if quanto_activations not in self.accepted_activations:
                raise ValueError(
                    f"Only support weights in {self.accepted_activations.keys()} but found {quanto_activations}"
                )
            quanto_config["activations"] = quanto_activations
        return dtype, quanto_config

    def run(
        self,
        video_path: str,
        audio_path: str,
        image_path: str,
        batch_size=None,
        delete_generated_files=False,
    ):
        logger.info(f"Generating video at {video_path}")
        animation_tag = Path(video_path).stem
        result_dir = Path(__file__).parent / "video"
        result_dir.mkdir(exist_ok=True)
        try:
            returned_video_path = super().test(
                source_image=image_path,
                driven_audio=audio_path,
                batch_size=batch_size or self.batch_size,
                result_dir=str(result_dir),
                tag=animation_tag,
            )

            # move returned_video_path to video_path
            shutil.move(returned_video_path, video_path)
            logger.info(f"Created video at {video_path}")
        except Exception as e:
            error_message = f"Failed to generate video: {e}"
            logger.error(error_message, exc_info=True)
            raise
        finally:
            if delete_generated_files:
                delete_path = os.path.join(result_dir, animation_tag)
                if os.path.exists(delete_path):
                    shutil.rmtree(delete_path, ignore_errors=True)
                    logger.info(f"Deleted generated files at {delete_path}")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


if __name__ == "__main__":
    pass
