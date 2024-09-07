import os
import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .animate import AnimateFromCoeff
from src.utils.logger_config import logger

mp.set_start_method("spawn", force=True)


class CustomAnimateFromCoeff:
    def __init__(
        self,
        sadtalker_paths: dict[str, str],
        devices: list[str] | list[int],
        dtype: Optional[str] = None,
        quanto_config: dict[str, str] = {},
        parallel_mode: Optional[str] = None,
    ):
        self.devices = devices
        self.dtype = dtype
        self.quanto_config = quanto_config
        self.parallel_mode = parallel_mode
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

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def _ddp_worker(
        self, rank, world_size, data_batches, result_queue, spawn_start_time
    ):
        try:
            self.setup(rank, world_size)
            device = self.devices[rank]
            logger.debug(f"Time to spawn rank {rank}: {time.time() - spawn_start_time}")
            logger.debug(f"Rank {rank} using device: {device}")

            torch.cuda.set_device(device)

            model = self.animate_from_coeff[rank]
            local_result = model.generate(data_batches[rank])

            gathered_results = [None] * world_size
            dist.all_gather_object(gathered_results, local_result.to(0))

            if rank == 0:
                # Concatenate all results only at rank 0
                final_result = torch.cat(gathered_results, dim=0).cpu()
                logger.debug("Final result shape:", final_result.shape)
                result_queue.put(final_result)

        except Exception as e:
            logger.error(f"Rank {rank} failed with exception: {e}")
            raise
        finally:
            self.cleanup()

    def get_batches(self, data: dict[str, torch.Tensor], frame_num: int):
        num_devices = len(self.devices)
        num_batches = data["target_semantics_list"].shape[0]
        batches_per_device = (num_batches + num_devices - 1) // num_devices
        frames_per_device = (frame_num + num_devices - 1) // num_devices

        # List to store the divided batches
        batches = []

        for i in range(num_devices):
            start_idx = i * batches_per_device
            end_idx = min((i + 1) * batches_per_device, num_batches)

            # Create a batch for this device
            batch_data = {}

            # Divide the tensors for each device
            batch_data["source_image"] = data["source_image"][start_idx:end_idx]
            batch_data["source_semantics"] = data["source_semantics"][start_idx:end_idx]
            batch_data["target_semantics_list"] = data["target_semantics_list"][
                start_idx:end_idx
            ]

            # Handle frame count distribution
            if i < num_devices - 1:
                batch_data["frame_num"] = frames_per_device
            else:
                # For the last device, handle the remaining frames
                batch_data["frame_num"] = frame_num - frames_per_device * i

            # Copy over the sequences if they exist in the data
            if "yaw_c_seq" in data:
                batch_data["yaw_c_seq"] = data["yaw_c_seq"][start_idx:end_idx]
            if "pitch_c_seq" in data:
                batch_data["pitch_c_seq"] = data["pitch_c_seq"][start_idx:end_idx]
            if "roll_c_seq" in data:
                batch_data["roll_c_seq"] = data["roll_c_seq"][start_idx:end_idx]

            # Append the batch for this device
            batches.append(batch_data)

        return batches

    def generate(self, data: dict[str, torch.Tensor], frame_num: int):
        data_batches = self.get_batches(data, frame_num)
        if self.parallel_mode != "ddp":
            try:
                return self.animate_from_coeff[0].generate(data_batches[0])
            except Exception as e:
                logger.error("Error in single process:", e)
                raise

        with mp.Manager() as manager:
            world_size = len(self.devices)
            result_queue = manager.Queue()
            logger.info("Spawning processes...")
            start_time = time.time()
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

            logger.info("All processes finished.")
            final_result = result_queue.get()  # This will be the result from rank 0
            logger.debug("Generated final result shape:", final_result.shape)
            return final_result
