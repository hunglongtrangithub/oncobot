from typing import Optional
import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from .app import run
from src.utils.env_config import settings


class CustomAnimateFromCoeff:
    def __init__(
        self,
        sadtalker_paths: dict[str, str],
        devices: list[str] | list[int],
        dtype: Optional[str] = None,
        quanto_config: dict[str, str] = {},
        parallel_mode: Optional[str] = None,
    ):
        run_devices = devices if parallel_mode == "ddp" else [devices[0]]

        # start server
        print("Starting server...")
        timeout = 10
        urls = []
        for i in range(1, len(run_devices) + 1):
            port = settings.port + i
            url = run(port, timeout=timeout)
            if url is None:
                raise Exception(f"Failed to start server on port {port}")
            urls.append(url)

        # call /setup endpoint
        for device, url in zip(run_devices, urls):
            setup_payload = {
                "sadtalker_paths": sadtalker_paths,
                "device": device,
                "dtype": dtype,
                "dp_device_ids": devices if parallel_mode == "dp" else None,
                "quanto_config": quanto_config,
            }
            response = requests.post(url + "/setup", json=setup_payload)
            if response.status_code != 200:
                raise Exception(f"Failed to set up server on port {url.split(':')[-1]}")
        print("Server started successfully.")
        self.urls = urls
        self.run_devices = run_devices

    def get_batches(self, data: dict[str, torch.Tensor], frame_num: int):
        num_devices = len(self.run_devices)
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
        batches = self.get_batches(data, frame_num)
        print("Sending requests to servers...")

        # List to store the predictions
        predictions = []

        # Function to send a request to a single server
        def send_request(batch, url):
            print(f"Turning tensors to lists for {url}...")
            batch = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            print(f"Sending request to {url}...")
            response = requests.post(url + "/predict", json={"x": batch})
            if response.status_code != 200:
                raise Exception(f"Failed to get prediction from {url}")
            pred = response.json()["prediction"]
            return torch.tensor(pred)

        # Use ThreadPoolExecutor to send requests concurrently
        with ThreadPoolExecutor() as executor:
            # Submit all the requests
            futures = [
                executor.submit(send_request, batch, url)
                for batch, url in zip(batches, self.urls)
            ]

            # Collect the results as they complete
            for future in as_completed(futures):
                pred = future.result()
                predictions.append(pred)

        # Concatenate all predictions into a single tensor
        print("Combining predictions...")
        predictions = torch.cat(predictions, dim=0)
        return predictions
