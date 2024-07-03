import os
from pathlib import Path
import shutil
from logger_config import get_logger

from sad_talker import SadTalker

logger = get_logger(__name__)


class DummyTalker:
    def __init__(self):
        logger.info("DummyTalker initialized")

    def run(self, video_path: str, audio_path: str, image_path: str):
        # copy video/chatbot1__1.mp4 to video_path
        shutil.copy(Path(__file__).parent / "video/chatbot1__1.mp4", video_path)
        logger.info(f"Created video {video_path}")

    async def arun(self, video_path: str, audio_path: str, image_path: str):
        self.run(video_path, audio_path, image_path)


class CustomSadTalker(SadTalker):
    def __init__(self, batch_size=150, device=None, dtype=None, parallel_mode=None):
        checkpoint_path = Path(__file__).parent / "sad_talker/checkpoints"
        config_path = Path(__file__).parent / "sad_talker/src/config"
        super().__init__(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            device=device,
            dtype=dtype,
            parallel_mode=parallel_mode,
        )
        self.batch_size = batch_size
        logger.info(f"CustomSadTalker initialized.")

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
        result_dir = str(Path(__file__).parent / "video")
        try:
            returned_video_path = super().test(
                source_image=image_path,
                driven_audio=audio_path,
                batch_size=batch_size or self.batch_size,
                result_dir=result_dir,
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
