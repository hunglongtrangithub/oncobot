import torch
import time
import yaml
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from talking_face import CustomSadTalker
from sad_talker.src.utils.init_path import init_path
from sad_talker.src.facerender.modules.mapping import MappingNet
from sad_talker.src.facerender.modules.generator import OcclusionAwareSPADEGenerator

image_size = 256
image_preprocess = "crop"
checkpoint_path = Path(__file__).parent.parent / "sad_talker/checkpoints"
config_path = Path(__file__).parent.parent / "sad_talker/src/config"
sad_talker_paths = init_path(
    str(checkpoint_path),
    str(config_path),
    image_size,
    False,
    image_preprocess,
)
with open(sad_talker_paths["facerender_yaml"]) as f:
    config = yaml.safe_load(f)


def test_talker():
    talker = CustomSadTalker(
        batch_size=50,
        device=[3, 4],
        # dtype=torch.float16,
        # parallel_mode="dp",
    )
    video_path = str(Path(__file__).parent / "video/chatbot1__1.mp4")
    audio_path = str(Path(__file__).parent / "examples/fake_patient3.wav")
    image_path = str(Path(__file__).parent / "examples/chatbot1.jpg")
    start = time.time()
    talker.run(
        video_path,
        audio_path,
        image_path,
        delete_generated_files=True,
    )
    print(f"Total time taken: {time.time() - start:.2f} seconds")
    assert os.path.exists(video_path)
    print("Test passed")


def test_mapping_net():
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
    print("Test passed")


def size(tensor):
    return tensor.element_size() * tensor.nelement()


def test_generator():
    generator = OcclusionAwareSPADEGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    generator.to(device="cuda")
    generator.eval()
    batch_size = 1
    num_frames = 15
    batch_size = batch_size * num_frames
    source_image = (
        torch.randn(batch_size, 3, 256, 256).type(torch.FloatTensor).to(device="cuda")
    )
    kp_source = {
        "value": torch.randn(batch_size, 15, 3)
        .type(torch.FloatTensor)
        .to(device="cuda")
    }
    kp_driving = {
        "value": torch.randn(batch_size, 15, 3)
        .type(torch.FloatTensor)
        .to(device="cuda")
    }
    print(f"source_image memory size: {size(source_image)/1024:.2f} KB")
    print(f"kp_source memory size: {size(kp_source['value'])/1024:.2f} KB")
    print(f"kp_driving memory size: {size(kp_driving['value'])/1024:.2f} KB")
    output = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
    print(output["prediction"].shape)
    assert output["prediction"].shape[0] == batch_size
    print("Test passed")


if __name__ == "__main__":
    test_talker()
    # test_mapping_net()
    # test_generator()
