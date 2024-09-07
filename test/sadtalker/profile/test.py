import torch
import time
from torch.nn.parallel import DataParallel
import sys
import os
from pathlib import Path

sys.path.append(str(Path(os.path.abspath("")).parents[2]))

from src.sadtalker.src.utils.init_path import init_path
from src.sadtalker.src.facerender.animate import AnimateFromCoeff
from src.sadtalker.src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.sadtalker.src.facerender.modules.make_animation import keypoint_transformation

image_size = 256
image_preprocess = "crop"
checkpoint_path = Path(os.path.abspath("")).parents[2] / "src/sadtalker/checkpoints"
gfpgan_path = Path(os.path.abspath("")).parents[2] / "src/sadtalker/gfpgan/weights"
config_path = Path(os.path.abspath("")).parents[2] / "src/sadtalker/src/config"
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

        # start = time.time()
        out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
        # print("generator time", time.time() - start)
        return out


model = AnimateFromCoeff(
    sadtalker_paths,
    device="cuda:1",
    dtype=torch.float32,
)
source_image = torch.rand(batch_size, 3, 256, 256)
source_semantics = torch.rand(batch_size, 70, 27)
target_semantics = torch.rand(batch_size, 8, 70, 27)
source_image = source_image.type(model.dtype).to(model.device)
source_semantics = source_semantics.type(model.dtype).to(model.device)
target_semantics = target_semantics.type(model.dtype).to(model.device)


if __name__ == "__main__":
    print("start generating")
    for _ in range(1):  # Warm-up iterations
        torch.cuda.synchronize()
        start = time.time()
        print("start")
        generate(
            source_image,
            source_semantics,
            target_semantics,
            model.generator,
            model.device,
            model.dtype,
        )
        print("end")
        torch.cuda.synchronize()
        print("warmup time", time.time() - start)

    for _ in range(0):
        torch.cuda.synchronize()
        start = time.time()
        generate(
            source_image,
            source_semantics,
            target_semantics,
            model.generator,
            model.device,
            model.dtype,
        )
        torch.cuda.synchronize()
        print("time", time.time() - start)
