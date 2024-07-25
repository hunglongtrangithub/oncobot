import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simulate a model that takes in a source image, driving keypoints, and source keypoints
class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, source_image, kp_driving, kp_source):
        flattened_source_image = source_image.view(source_image.size(0), -1)
        flattened_kp_driving = kp_driving["value"].view(kp_driving["value"].size(0), -1)
        flattened_kp_source = kp_source["value"].view(kp_source["value"].size(0), -1)

        input = torch.cat(
            [flattened_source_image, flattened_kp_driving, flattened_kp_source], dim=1
        )
        output = self.fc(input)

        print(
            "\tIn Model: input size",
            source_image.size(),
            kp_driving["value"].size(),
            kp_source["value"].size(),
            "output size",
            output.size(),
        )
        return output


def test_parallel():
    import math

    num_frames = 521
    batch_size = 10**3
    target_semantics = torch.randn(
        batch_size, math.ceil(num_frames / batch_size), 70, 27
    ).to(device)

    source_image = torch.randn(batch_size, 3, 256, 256).to(device)
    kp_source = {"value": torch.randn(batch_size, 15, 3).to(device)}
    kp_norm = {"value": torch.randn(batch_size, 15, 3).to(device)}
    input_size = 3 * 256 * 256 + 15 * 3 + 15 * 3
    output_size = 2
    print("target_semantics", target_semantics.size())
    print("input_size", input_size)
    print("output_size", output_size)

    model = Model(input_size, output_size)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    output = model(source_image, kp_driving=kp_norm, kp_source=kp_source)
    print(
        "Outside: input size",
        source_image.size(),
        kp_source["value"].size(),
        kp_norm["value"].size(),
        "output_size",
        output.size(),
    )


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        # print("ToyModel init")
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        print("ToyModel forward:", x.size())
        result = self.net2(self.relu(self.net1(x)))
        time.sleep(5)  # Simulate a slow forward pass
        print("ToyModel forward:", result.size())
        return result


class ModelComposite:
    def __init__(self, device, device_ids=None):
        self.model = ToyModel()
        self.device = device
        if device_ids:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        self.model.to(device)
        self.model.eval()

    def generate(self, data):
        data = data.to(self.device)
        with torch.no_grad():
            return self.model(data)


class DDPDemo:
    def __init__(self, devices, parallel_mode=None):
        self.parallel_mode = parallel_mode
        self.devices = devices
        if self.parallel_mode == "dp":
            self.animate_from_coeff = [
                ModelComposite(self.devices[0], device_ids=self.devices)
            ]
        elif self.parallel_mode == "ddp":
            self.animate_from_coeff = [
                ModelComposite(device) for device in self.devices
            ]
        else:
            self.animate_from_coeff = [ModelComposite(self.devices[0])]

    def setup(self, rank, world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        mp.set_start_method("fork", force=True)
        # initialize the process group
        # NOTE: use "gloo", otherwise the resultant tensor cannot be released from GPU memory to the main process
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def _ddp_worker(
        self, rank, world_size, data_batches, result_queue, spawn_start_time
    ):
        try:
            self.setup(rank, world_size)
            device = self.devices[rank]
            print(f"Time to spawn rank {rank}: {time.time() - spawn_start_time}")
            print(f"Rank {rank} using device: {device}")

            torch.cuda.set_device(device)

            model_composite = self.animate_from_coeff[rank]
            # kinda not have to do this
            # model_composite.model = DDP(model_composite.model, device_ids=[device])
            print("Using AnimateFromCoeff model at device:", model_composite.device)
            print("Data batch device:", data_batches[rank].device)
            local_result = model_composite.generate(data_batches[rank])
            # local_result = torch.randn(10, 5).to(device)
            # print(f"Rank {rank} finished processing. Result shape:", local_result.shape)

            # Prepare a list to collect all results at rank 0
            gathered_results = [None] * world_size

            # Gather all results to rank 0
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
            start_time = time.time()
            result = self.animate_from_coeff[0].generate(data_batches[0])
            print("_get_generated_video_data:", time.time() - start_time)
            return result

        with mp.Manager() as manager:
            world_size = len(self.devices)
            result_queue = manager.Queue()
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

            print("All processes finished.")
            print("_get_generated_video_data:", time.time() - start_time)
            final_result = result_queue.get()  # This will be the result from rank 0
            return final_result

    def _generate_batches(self, batch_size):
        if self.parallel_mode != "ddp":
            return [torch.randn(batch_size, 10)]
        return [
            torch.randn(batch_size // len(self.devices), 10) for device in self.devices
        ]

    def run(self, batch_size):
        data_batches = self._generate_batches(batch_size)
        final_result = self._get_generated_video_data(data_batches)
        print("Finish run:", final_result.size())
        return final_result


if __name__ == "__main__":
    demo = DDPDemo([4, 5], parallel_mode="dp")
    final_result = demo.run(10**6)
