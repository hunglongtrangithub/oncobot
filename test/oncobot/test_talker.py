import os
import time
from pathlib import Path
import pytest
import torch
import yaml
import sys

sys.path.append(str(Path(__file__).parents[2]))
from src.oncobot.talking_face import CustomSadTalker
from src.sadtalker.src.facerender.modules.generator import OcclusionAwareSPADEGenerator
from src.sadtalker.src.facerender.modules.mapping import MappingNet
from src.sadtalker.src.utils.init_path import init_path
from src.sadtalker.src.facerender.modules.make_animation import (
    headpose_pred_to_degree,
    keypoint_transformation,
)


@pytest.fixture(scope="module")
def config():
    image_size = 256
    image_preprocess = "crop"
    checkpoint_path = Path(__file__).parents[2] / "src/sadtalker/checkpoints"
    gfpgan_path = Path(__file__).parents[2] / "src/sadtalker/gfpgan/weights"
    config_path = Path(__file__).parents[2] / "src/sadtalker/src/config"
    sadtalker_paths = init_path(
        str(checkpoint_path),
        str(gfpgan_path),
        str(config_path),
        image_size,
        False,
        image_preprocess,
    )
    with open(sadtalker_paths["facerender_yaml"]) as f:
        config = yaml.safe_load(f)
    print("config loaded")
    return config


def test_headpose_pred_to_degree():
    pred = torch.randn([30, 66])
    degree = headpose_pred_to_degree(pred)
    print(degree.shape)


def test_keypoint_transformation():
    kp_canonical = {"value": torch.randn([30, 15, 3])}
    he_source = {
        "yaw": torch.randn([30, 66]),
        "pitch": torch.randn([30, 66]),
        "roll": torch.randn([30, 66]),
        "t": torch.randn([30, 3]),
        "exp": torch.randn([30, 45]),
    }
    kp_transformed = keypoint_transformation(kp_canonical, he_source)
    print(kp_transformed["value"].shape)


# # @profile
def test_talker():
    talker = CustomSadTalker(
        batch_size=60,
        device=[1, 2],
        parallel_mode="dp",
        torch_dtype="float16",
        # quanto_weights="int8",
        # quanto_activations=None,
    )
    video_folder = Path(__file__).parent / "video"
    video_folder.mkdir(exist_ok=True)
    video_path = str(video_folder / "chatbot__1.mp4")
    audio_path = str(Path(__file__).parents[2] / "examples/fake_patient3.wav")
    image_path = str(Path(__file__).parents[2] / "examples/chatbot1.jpg")
    for _ in range(3):
        start = time.perf_counter()
        talker.run(
            video_path,
            audio_path,
            image_path,
            delete_generated_files=False,
        )
        print(f"Total time taken: {time.perf_counter() - start:.2f} seconds")
        assert os.path.exists(video_path)


def test_mapping_net(config):
    mapping = MappingNet(**config["model_params"]["mapping_params"])
    mapping.eval()
    batch_size = 3
    num_frames = 333
    batch_size = batch_size * num_frames
    flat_target_semantics = torch.randn(batch_size, 70, 27)
    output = mapping(flat_target_semantics)
    for key in output:
        print(key, output[key].shape)
        assert output[key].shape[0] == batch_size


def size(tensor):
    return tensor.element_size() * tensor.nelement()


def test_generator(config):
    generator = OcclusionAwareSPADEGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    generator.to(device="cuda")
    generator.eval()
    batch_size = 50
    num_frames = 15
    batch_size = batch_size * num_frames
    source_image = (
        torch.randn(batch_size, 3, 256, 256).type(torch.float32).to(device="cuda")
    )
    kp_source = {
        "value": torch.randn(batch_size, 15, 3).type(torch.float32).to(device="cuda")
    }
    kp_driving = {
        "value": torch.randn(batch_size, 15, 3).type(torch.float32).to(device="cuda")
    }
    print(f"source_image memory size: {size(source_image)/1024:.2f} KB")
    print(f"kp_source memory size: {size(kp_source['value'])/1024:.2f} KB")
    print(f"kp_driving memory size: {size(kp_driving['value'])/1024:.2f} KB")
    output = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
    print(output["prediction"].shape)
    assert output["prediction"].shape[0] == batch_size


if __name__ == "__main__":
    # test_headpose_pred_to_degree()
    # test_keypoint_transformation()
    test_talker()
    # test_generator()
    pass
