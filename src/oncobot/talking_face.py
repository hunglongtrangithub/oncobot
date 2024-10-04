import os
from pathlib import Path
import shutil
import random

from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

from src.sadtalker import SadTalker
from src.utils.logger_config import logger
from src.sadtalker.src.facerender.animate import (
    ACCEPTED_WEIGHTS,
    ACCEPTED_ACTIVATIONS,
    ACCEPTED_DTYPES,
)


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


class FakeTalker:

    def __init__(
        self,
        sample_video_bank_full_path: str = str(Path(__file__).parents[2] / "examples"),
    ):
        self.sample_video_bank = sample_video_bank_full_path
        logger.info(
            f"FakeTalker initialized. Looking for sample videos in {self.sample_video_bank}"
        )

    def run(self, video_path: str, audio_path: str, image_path: str):
        bot_name = Path(image_path).stem
        sample_video_path = Path(self.sample_video_bank) / f"{bot_name}.mp4"

        logger.info(f"Sample video path: {sample_video_path}")
        if not sample_video_path.exists():
            raise FileNotFoundError(f"Sample video not found at {sample_video_path}")

        sample_video = VideoFileClip(str(sample_video_path))
        audio_clip = AudioFileClip(str(audio_path))
        audio_duration = audio_clip.duration

        video_data = []
        total_duration = 0

        while total_duration < audio_duration:
            n = random.uniform(1, min(10, sample_video.duration))
            # Pick a random start time within the video
            start_time = random.uniform(0, sample_video.duration - n)

            video_segment = sample_video.subclip(start_time, start_time + n)
            video_data.append(video_segment)
            total_duration += n

        final_video = concatenate_videoclips(video_data)
        # Ensure the final video matches the audio duration
        final_video = final_video.subclip(0, audio_duration)
        # Set the audio to the final video
        final_video = final_video.set_audio(audio_clip)
        # Save the generated video to the specified path
        final_video.write_videofile(str(video_path), codec="libx264", audio_codec="aac")

        logger.info(f"Video saved to {video_path}")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


class CustomSadTalker(SadTalker):
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
        logger.debug(f"quanto_config: {quanto_config}")
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
            quanto_config=quanto_config,
        )
        self.batch_size = batch_size
        logger.info("CustomSadTalker initialized.")

    def _check_arguments(self, torch_dtype, quanto_weights, quanto_activations):
        dtype = None
        if torch_dtype:
            if torch_dtype not in ACCEPTED_DTYPES:
                raise ValueError(
                    f"Only support dtypes in {ACCEPTED_DTYPES.keys()} but found {torch_dtype}"
                )
            dtype = torch_dtype
        quanto_config = {}
        if quanto_weights:
            if quanto_weights not in ACCEPTED_WEIGHTS:
                raise ValueError(
                    f"Only support weights in {ACCEPTED_WEIGHTS.keys()} but found {quanto_weights}"
                )
            quanto_config["weights"] = quanto_weights
        if quanto_activations:
            if quanto_activations not in ACCEPTED_ACTIVATIONS:
                raise ValueError(
                    f"Only support weights in {ACCEPTED_ACTIVATIONS.keys()} but found {quanto_activations}"
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
