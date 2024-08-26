import sys
from pathlib import Path
import torch
from torch.nn import DataParallel
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

sys.path.append(str(Path(__file__).parents[3]))
from src.sadtalker.src.utils.init_path import init_path
from src.sadtalker.src.facerender.animate import AnimateFromCoeff
from src.sadtalker.src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.sadtalker.src.facerender.modules.make_animation import keypoint_transformation

image_size = 256
image_preprocess = "crop"
checkpoint_path = Path(__file__).parents[3] / "src/sadtalker/checkpoints"
gfpgan_path = Path(__file__).parents[3] / "src/sadtalker/gfpgan/weights"
config_path = Path(__file__).parents[3] / "src/sadtalker/src/config"
sadtalker_paths = init_path(
    str(checkpoint_path),
    str(gfpgan_path),
    str(config_path),
    image_size,
    False,
    image_preprocess,
)


batch_size = 30


def generate(
    source_image: torch.Tensor,
    source_semantics: torch.Tensor,
    target_semantics: torch.Tensor,
    generator: (
        OcclusionAwareSPADEGenerator | DataParallel[OcclusionAwareSPADEGenerator]
    ),
    device,
    dtype,
):
    with torch.no_grad():
        kp_canonical = {
            "value": torch.rand((batch_size, 15, 3), device=device, dtype=dtype)
        }
        he_source = {
            "yaw": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "pitch": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "roll": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "t": torch.rand((batch_size, 3), device=device, dtype=dtype),
            "exp": torch.rand((batch_size, 45), device=device, dtype=dtype),
        }
        kp_source = keypoint_transformation(kp_canonical, he_source)

        he_driving = {
            "yaw": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "pitch": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "roll": torch.rand((batch_size, 66), device=device, dtype=dtype),
            "t": torch.rand((batch_size, 3), device=device, dtype=dtype),
            "exp": torch.rand((batch_size, 45), device=device, dtype=dtype),
        }
        kp_driving = keypoint_transformation(kp_canonical, he_driving)

        out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)

        return out


model = AnimateFromCoeff(
    sadtalker_paths,
    device="cuda:1",
    dtype=torch.float32,
)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./log/generator"),
) as prof:
    with record_function("model_inference"):
        # mock a source image, source semantics, and target semantics
        source_image = torch.rand(batch_size, 3, 256, 256)
        source_semantics = torch.rand(batch_size, 70, 27)
        target_semantics = torch.rand(batch_size, 8, 70, 27)
        source_image = source_image.type(model.dtype).to(model.device)
        source_semantics = source_semantics.type(model.dtype).to(model.device)
        target_semantics = target_semantics.type(model.dtype).to(model.device)
        for _ in range(1):
            generate(
                source_image,
                source_semantics,
                target_semantics,
                model.generator,
                model.device,
                model.dtype,
            )
