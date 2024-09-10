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
from src.sadtalker.src.facerender.modules.mapping import MappingNet
from src.sadtalker.src.facerender.modules.keypoint_detector import (
    HEEstimator,
    KPDetector,
)
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


def animate(
    source_image: torch.Tensor,
    source_semantics: torch.Tensor,
    target_semantics: torch.Tensor,
    generator: (
        OcclusionAwareSPADEGenerator | DataParallel[OcclusionAwareSPADEGenerator]
    ),
    kp_detector: KPDetector | DataParallel[KPDetector],
    he_estimator: HEEstimator | DataParallel[HEEstimator],
    mapping: MappingNet | DataParallel[MappingNet],
):
    with torch.inference_mode():
        predictions = []
        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)

        for frame_idx in range(target_semantics.shape[1]):
            he_driving = mapping(target_semantics[:, frame_idx])
            kp_driving = keypoint_transformation(kp_canonical, he_driving)

            out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
            predictions.append(out["prediction"])

        return torch.stack(predictions, dim=1)


batch_size = 60
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler(
        "./log/animate", "pdgx0002_animate_inference_mode"
    ),
) as prof:
    with record_function("model_init"):
        model = AnimateFromCoeff(
            sadtalker_paths,
            device="cuda:1",
            dtype=torch.float32,
        )
    with record_function("model_inference"):
        # mock a source image, source semantics, and target semantics
        source_image = torch.rand(batch_size, 3, 256, 256)
        source_semantics = torch.rand(batch_size, 70, 27)
        target_semantics = torch.rand(batch_size, 8, 70, 27)
        source_image = source_image.type(model.dtype).to(model.device)
        source_semantics = source_semantics.type(model.dtype).to(model.device)
        target_semantics = target_semantics.type(model.dtype).to(model.device)
        animate(
            source_image,
            source_semantics,
            target_semantics,
            model.generator,
            model.kp_extractor,
            model.he_estimator,
            model.mapping,
        )
