import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        print("input shape:", x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        print("output shape:", x.shape)
        return x


# NOTE: HAVE TO USE GLOO, OTHERWISE CANNOT SEND THE RESULTANT TENSOR TO THE MAIN PROCESS
def process(rank, world_size, result_queue):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    print(f"Running on rank {rank}")

    local_result = torch.Tensor([rank])

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    input_tensor = torch.Tensor(range(10)).to(rank)
    with torch.no_grad():
        local_result = ddp_model(input_tensor)
        print(f"Rank {rank} processed {local_result.shape[0]} samples")

    if rank == 0:
        gathered_results = [torch.zeros_like(local_result) for _ in range(world_size)]
    else:
        gathered_results = None

    dist.gather(local_result, gather_list=gathered_results, dst=0)

    if rank == 0:
        # Concatenate all results only at rank 0
        print("Concatenating results")
        final_result = torch.cat(gathered_results, dim=0)
        result_queue.put(final_result.cpu())


def test_queue():
    with mp.Manager() as manager:
        result_queue = manager.Queue()
        world_size = 3

        # Spawn processes
        mp.spawn(
            process,
            args=(world_size, result_queue),
            nprocs=world_size,
            join=True,
        )

        final_result = result_queue.get()
        print(final_result)


def test_divide():
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
    data = torch.cat([batch.flatten() for batch, _ in batches], dim=0)
    print(data)


if __name__ == "__main__":
    test_queue()
