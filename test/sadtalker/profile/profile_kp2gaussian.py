import torch
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)


def make_coordinate_grid(spatial_size: tuple[int, int, int]) -> torch.Tensor:
    d, h, w = spatial_size
    x = torch.arange(w)
    y = torch.arange(h)
    z = torch.arange(d)

    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    z = 2 * (z / (d - 1)) - 1

    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    return meshed


def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp["value"]
    # TEST: spatial_size torch.Size([16, 64, 64])
    # print("spatial_size", spatial_size)
    coordinate_grid = make_coordinate_grid(spatial_size)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = coordinate_grid.to(mean.device, non_blocking=True) - mean

    out = torch.exp(-0.5 * (mean_sub**2).sum(-1) / kp_variance)

    return out


batch_size = 30
kp_driving = {"value": torch.randn(batch_size, 15, 3).type(torch.float32).to("cuda:0")}
spatial_size = (16, 64, 64)
kp_variance = 0.01


with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./log/kp2gaussian"),
) as prof:
    with record_function("kp2gaussian"):
        kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=kp_variance)
