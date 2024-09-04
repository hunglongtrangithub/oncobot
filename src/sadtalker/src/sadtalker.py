import gc
import time
import os
import torch
import uuid

from pydub import AudioSegment
import torch.distributed as dist
import torch.multiprocessing as mp

from .utils.preprocess import CropAndExtract
from .test_audio2coeff import Audio2Coeff
from .facerender.animate import AnimateFromCoeff

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
        checkpoint_path="checkpoints",
        gfpgan_path="gfpgan/weights",
        config_path="src/config",
        image_size=256,
        image_preprocess="crop",
        device=None,
        dtype=None,
        parallel_mode=None,
        **quanto_config,
    ):
        self.devices = self._parse_devices(device)
        self.parallel_mode = parallel_mode
        if self.parallel_mode is not None and len(self.devices) < 2:
            raise ValueError(
                "Parallel mode is enabled but only one device is detected. Please provide more devices."
            )

        print("SadTalker devices:", self.devices)
        os.environ["TORCH_HOME"] = checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.gfpgan_path = gfpgan_path
        self.config_path = config_path
        self.image_size = image_size
        self.image_preprocess = image_preprocess
        self.dtype = dtype
        self.quanto_config = quanto_config
        self.initialize_all_models()

    def _parse_devices(self, device):
        if device is None:
            return (
                [torch.device("cuda:0")]
                if torch.cuda.is_available()
                else [torch.device("cpu")]
            )

        if isinstance(device, int):
            return [torch.device(f"cuda:{device}")]

        if isinstance(device, str):
            return [torch.device(device)]

        if isinstance(device, list):
            parsed_devices = []
            for d in device:
                if isinstance(d, int):
                    parsed_devices.append(torch.device(f"cuda:{d}"))
                elif isinstance(d, str):
                    parsed_devices.append(torch.device(d))
                else:
                    raise ValueError(
                        f"Invalid device type: {type(d)}. Expected int or str."
                    )
            return parsed_devices

        raise ValueError(
            f"Invalid device type: {type(device)}. Expected int, str, or list of int/str."
        )

    def initialize_all_models(self):
        sadtalker_paths = init_path(
            self.checkpoint_path,
            self.gfpgan_path,
            self.config_path,
            self.image_size,
            False,
            self.image_preprocess,
        )
        print("Loading models for SadTalker...")
        self.preprocess_model = CropAndExtract(sadtalker_paths, self.devices[0])
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths, self.devices[0])
        if self.parallel_mode == "ddp":
            self.animate_from_coeff = [
                AnimateFromCoeff(
                    sadtalker_paths,
                    device=device,
                    dtype=self.dtype,
                    **self.quanto_config,
                )
                for device in self.devices
            ]
        elif self.parallel_mode == "dp":
            self.animate_from_coeff = [
                AnimateFromCoeff(
                    sadtalker_paths,
                    device=self.devices[0],
                    dtype=self.dtype,
                    dp_device_ids=self.devices,
                    **self.quanto_config,
                )
            ]
        else:
            self.animate_from_coeff = [
                AnimateFromCoeff(
                    sadtalker_paths,
                    device=self.devices[0],
                    dtype=self.dtype,
                    **self.quanto_config,
                )
            ]
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

    def _prepare_facerender_data(
        self,
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        still_mode,
        exp_scale,
    ):
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            expression_scale=exp_scale,
            preprocess=self.image_preprocess,
            size=self.image_size,
        )
        if self.parallel_mode != "ddp":
            return [data], data["video_name"], data["audio_path"], data["frame_num"]

        num_devices = len(self.devices)
        num_batches = data["target_semantics_list"].shape[0]
        batches_per_device = (num_batches + num_devices - 1) // num_devices
        frames_per_device = (data["frame_num"] + num_devices - 1) // num_devices
        # Divide the batches for each device for DDP
        batches = []
        for i in range(num_devices):
            start_idx = i * batches_per_device
            end_idx = min((i + 1) * batches_per_device, num_batches)
            batch_data = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in data.items()
            }
            batch_data["source_image"] = batch_data["source_image"][start_idx:end_idx]
            batch_data["source_semantics"] = batch_data["source_semantics"][
                start_idx:end_idx
            ]
            batch_data["frame_num"] = (
                frames_per_device
                if i < num_devices - 1
                else data["frame_num"] - frames_per_device * i
            )
            batch_data["target_semantics_list"] = batch_data["target_semantics_list"][
                start_idx:end_idx
            ]
            if "yaw_c_seq" in data:
                batch_data["yaw_c_seq"] = data["yaw_c_seq"][start_idx:end_idx]
            if "pitch_c_seq" in data:
                batch_data["pitch_c_seq"] = data["pitch_c_seq"][start_idx:end_idx]
            if "roll_c_seq" in data:
                batch_data["roll_c_seq"] = data["roll_c_seq"][start_idx:end_idx]

            batches.append(batch_data)
        return batches, data["video_name"], data["audio_path"], data["frame_num"]

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # initialize the process group
        # NOTE: use "gloo", otherwise the resultant tensor cannot be released from GPU memory to the main process
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def _ddp_worker(
        self, rank, world_size, data_batches, result_queue, spawn_start_time
    ):
        try:
            self.setup(rank, world_size)
            device = self.devices[rank]
            print(
                f"Time to spawn rank {rank}: {time.perf_counter() - spawn_start_time}"
            )
            print(f"Rank {rank} using device: {device}")

            torch.cuda.set_device(device)

            model = self.animate_from_coeff[rank]
            local_result = model.generate(data_batches[rank])

            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result.to(0))

            if rank == 0:
                # Concatenate all results only at rank 0
                final_result = torch.cat(gathered_results, dim=0).cpu()
                print("Final result shape:", final_result.shape)
                result_queue.put(final_result)

        except Exception as e:
            print(f"Rank {rank} failed with exception: {e}")
            raise
        finally:
            self.cleanup()

    def _get_generated_video_data(self, data_batches):
        if self.parallel_mode != "ddp":
            try:
                return self.animate_from_coeff[0].generate(data_batches[0])
            except Exception as e:
                print("Error in single process:", e)
                self.clean()
                raise

        with mp.Manager() as manager:
            world_size = len(self.devices)
            result_queue = manager.Queue()
            print("Spawning processes...")
            start_time = time.perf_counter()
            # Spawn processes
            mp.spawn(  # type: ignore
                self._ddp_worker,
                args=(
                    world_size,
                    data_batches,
                    result_queue,
                    start_time,
                ),
                nprocs=world_size,
                join=True,
            )

            print("All processes finished.")
            print("_get_generated_video_data:", time.perf_counter() - start_time)
            final_result = result_queue.get()  # This will be the result from rank 0
            return final_result

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
                self.devices[0],
                ref_eyeblink_coeff_path=None,
                still=still_mode,
                use_blink=use_blink,
            )
            coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style)

            data_batches, video_name, audio_path, frame_num = (
                self._prepare_facerender_data(
                    coeff_path,
                    crop_pic_path,
                    first_coeff_path,
                    audio_path,
                    batch_size,
                    still_mode,
                    exp_scale,
                )
            )
            video_data = self._get_generated_video_data(data_batches)

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
            for model in self.animate_from_coeff:
                del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
