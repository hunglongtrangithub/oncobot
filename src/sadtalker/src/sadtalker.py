import gc
import os
import torch
import uuid
from typing import Optional
from pydub import AudioSegment

from .utils.preprocess import CropAndExtract
from .test_audio2coeff import Audio2Coeff
from .facerender import CustomAnimateFromCoeff

from .utils.videoio import save_data_to_video
from .generate_batch import get_data
from .generate_facerender_batch import get_facerender_data
from .utils.init_path import init_path


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker:
    def __init__(
        self,
        checkpoint_path: str = "checkpoints",
        gfpgan_path: str = "gfpgan/weights",
        config_path: str = "src/config",
        image_size: int = 256,
        image_preprocess: str = "crop",
        device: Optional[str | int | list[int]] = None,
        dtype: Optional[str] = None,
        parallel_mode: Optional[str] = None,
        quanto_config: dict[str, str] = {},
    ):
        animate_devices = self._parse_devices(device)
        self.parallel_mode = parallel_mode
        if self.parallel_mode is not None and len(animate_devices) == 1:
            raise ValueError(
                "Parallel mode is enabled but only one device is detected. Please provide more devices."
            )

        print("SadTalker devices:", animate_devices)
        os.environ["TORCH_HOME"] = checkpoint_path
        self.device = torch.device(animate_devices[0])
        self.checkpoint_path = checkpoint_path
        self.gfpgan_path = gfpgan_path
        self.config_path = config_path
        self.image_size = image_size
        self.image_preprocess = image_preprocess
        self.dtype = dtype
        self.quanto_config = quanto_config
        self.initialize_all_models(animate_devices)

    def _parse_devices(self, device):
        if device is None:
            return ["cuda"] if torch.cuda.is_available() else ["cpu"]

        if isinstance(device, int):
            return [device]

        if isinstance(device, str):
            return [device]

        if isinstance(device, list):
            parsed_devices = []
            for d in device:
                if isinstance(d, int):
                    parsed_devices.append(d)
                else:
                    raise ValueError(
                        f"Invalid device type: {type(d)}. Expected int or str."
                    )
            return parsed_devices

        raise ValueError(
            f"Invalid device type: {type(device)}. Expected int, str, or list of int/str."
        )

    def initialize_all_models(self, animate_devices: list[int] | list[str]):
        sadtalker_paths = init_path(
            self.checkpoint_path,
            self.gfpgan_path,
            self.config_path,
            self.image_size,
            False,
            self.image_preprocess,
        )
        print("Loading models for SadTalker...")
        self.preprocess_model = CropAndExtract(sadtalker_paths, self.device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, self.device)
        self.animate_from_coeff = CustomAnimateFromCoeff(
            sadtalker_paths,
            animate_devices,
            self.dtype,
            self.quanto_config,
            self.parallel_mode,
        )
        print("Models loaded.")

    def _prepare_audio(self, driven_audio):
        if not os.path.isfile(driven_audio):
            raise AttributeError("No audio file is detected")

        if ".mp3" in driven_audio:
            audio_path = driven_audio.replace(".mp3", ".wav")
            mp3_to_wav(driven_audio, audio_path, 16000)
            os.remove(driven_audio)
        else:
            audio_path = driven_audio
        return audio_path

    def _preprocess_image(self, source_image, save_dir):
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)

        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            source_image, first_frame_dir, self.image_preprocess, True, self.image_size
        )

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        return first_coeff_path, crop_pic_path, crop_info

    # @profile
    def test(
        self,
        source_image,
        driven_audio,
        still_mode=False,
        batch_size=1,
        pose_style=0,
        exp_scale=1.0,
        use_blink=True,
        result_dir="./results/",
        tag=None,
    ):
        try:
            time_tag = str(uuid.uuid4()) if tag is None else tag
            save_dir = os.path.join(result_dir, time_tag)
            os.makedirs(save_dir, exist_ok=True)

            audio_path = self._prepare_audio(driven_audio)
            first_coeff_path, crop_pic_path, crop_info = self._preprocess_image(
                source_image, save_dir
            )

            batch = get_data(
                first_coeff_path,
                audio_path,
                self.device,
                ref_eyeblink_coeff_path=None,
                still=still_mode,
                use_blink=use_blink,
            )
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style)

            data_batches, frame_num, video_name = get_facerender_data(
                coeff_path,
                crop_pic_path,
                first_coeff_path,
                batch_size,
                still_mode=still_mode,
                expression_scale=exp_scale,
                preprocess=self.image_preprocess,
                size=self.image_size,
            )
            video_data = self.animate_from_coeff.generate(data_batches, frame_num)

            return_path = save_data_to_video(
                video_name,
                audio_path,
                video_data,
                crop_info,
                self.image_size,
                frame_num,
                self.image_preprocess,
                crop_pic_path,
                save_dir,
            )
            return return_path
        except Exception as e:
            print(f"Error in test: {e}")
            raise
        finally:
            self.clean()

    def clean(self, delete_models=False):
        print("Cleaning up...")
        if delete_models:
            del self.preprocess_model
            del self.audio_to_coeff
            del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
