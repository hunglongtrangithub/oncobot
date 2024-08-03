import torch
import torch.multiprocessing as mp
import torch.nn as nn
import time
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def process(tensor):
    start = time.time()
    model = ToyModel()
    result = model(tensor)
    processing_time = time.time() - start
    print("Original Tensor shape:")
    print(tensor.shape)
    print("Processed Tensor:")
    print(result.shape)
    print("Processing time:", processing_time)
    return


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
    )


def cleanup():
    dist.destroy_process_group()


def process_batch(rank, world_size, models, batches, result_queue):
    setup(rank, world_size)
    print(f"Running DDP on rank {rank}")

    model = models[rank]
    print("Using model on device", model.dummy_param.device)
    ddp_model = DDP(model, device_ids=[rank])

    with torch.no_grad():
        print("Rank", rank, "received", batches[rank].shape, "samples")
        local_result = ddp_model(batches[rank].to(rank))
        print(f"local_result shape: {local_result.shape}")

    result_list = [None for _ in range(world_size)]

    dist.all_gather_object(result_list, local_result.to(0))

    if rank == 0:
        # Concatenate all results only at rank 0
        print("Concatenating results. Number of results:", len(result_list))
        for i in range(world_size):
            print(f"Result {i} shape:", result_list[i].shape) # type: ignore
        final_result = torch.cat(result_list, dim=0) # type: ignore
        print("Final result shape:", final_result.shape)
        final_result = final_result.cpu()
        print("Final result shape after moving to CPU:", final_result.shape)
        result_queue.put(final_result)
        print(f"Rank {rank} done.")


def test_ddp():

    # Create a large tensor
    data = torch.randn(10000, 10)

    # Number of processes to spawn
    world_size = 3
    models = [ToyModel().to(rank) for rank in range(world_size)]

    batch_size = data.shape[0]
    # Divide the batch into smaller batches
    batches_per_device = (batch_size + world_size - 1) // world_size
    batches = []
    for i in range(world_size):
        start_idx = i * batches_per_device
        end_idx = min((i + 1) * batches_per_device, batch_size)
        batch_data = data[start_idx:end_idx]
        batches.append(batch_data)

    print("Starting DDP.")
    print("Number of processes:", world_size)
    print("Number of batches:", len(batches))

    with mp.Manager() as manager:
        result_queue = manager.Queue()
        world_size = 3

        start = time.time()
        # Spawn processes
        mp.spawn( # type: ignore
            process_batch,
            args=(world_size, models, batches, result_queue),
            nprocs=world_size,
            join=True,
        )

        print("Results received.")
        final_result = result_queue.get()
        result = final_result

        print(result)

        print("Original Tensor shape:")
        print(data.shape)
        print("Processed Tensor:")
        print(result.shape)
        print("Processing time:", time.time() - start)


def test_divide_batch():
    data = torch.Tensor(range(200))
    print(data)
    data = data.view(25, 4, 2)
    total_frames = 98
    batches = []
    num_devices = 7
    num_batches = data.shape[0]
    batches_per_device = (num_batches + num_devices - 1) // num_devices
    frames_per_device = (total_frames + num_devices - 1) // num_devices
    for i in range(num_devices):
        start_idx = i * batches_per_device
        end_idx = min((i + 1) * batches_per_device, num_batches)
        batch_data = data[start_idx:end_idx]
        num_frames = (
            frames_per_device
            if i < num_devices - 1
            else total_frames - i * frames_per_device
        )
        batches.append([batch_data, num_frames])
        print(f"Batch {i}:", batch_data.shape)
        print(f"Num frames {i}:", num_frames)
    new_data = torch.cat([batch.flatten() for batch, _ in batches], dim=0)
    print(new_data)
    assert data == new_data, "Data not equal"
