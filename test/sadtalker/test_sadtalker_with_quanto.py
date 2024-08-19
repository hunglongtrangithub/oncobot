from pathlib import Path

import torch
from optimum import quanto

from src.sadtalker.src.facerender.animate import AnimateFromCoeff
from src.sadtalker.src.utils.init_path import init_path

image_size = 256
image_preprocess = "crop"
checkpoint_path = Path(__file__).parents[2] / "src/sadtalker/checkpoints"
config_path = Path(__file__).parents[2] / "src/sadtalker/src/config"
sadtalker_paths = init_path(
    str(checkpoint_path),
    str(config_path),
    image_size,
    False,
    image_preprocess,
)


def named_module_tensors(module, recurse=False):
    for named_parameter in module.named_parameters(recurse=recurse):
        name, val = named_parameter
        if hasattr(val, "_data") or hasattr(val, "_scale"):
            if hasattr(val, "_data"):
                yield name + "._data", val._data
            if hasattr(val, "_scale"):
                yield name + "._scale", val._scale
        else:
            yield named_parameter

    for named_buffer in module.named_buffers(recurse=recurse):
        yield named_buffer


def dtype_byte_size(dtype):
    """
    Returns the size (in bytes) occupied by one parameter of type `dtype`.
    """
    import re

    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def compute_module_sizes(model):
    """
    Compute the size of each submodule of a given model.
    """
    from collections import defaultdict

    module_sizes = defaultdict(int)
    for name, tensor in named_module_tensors(model, recurse=True):
        size = tensor.numel() * dtype_byte_size(tensor.dtype)
        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


# define a simple CNN
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = torch.nn.Linear(32 * 64 * 64, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 64 * 64)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# model = SimpleCNN()
model = AnimateFromCoeff(sadtalker_paths, device="cuda:6").generator.first

weights = quanto.qint8
activations = None


def test_compute():
    quanto.quantize(model, weights=weights, activations=activations)
    # Run the model
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
    print(output)
