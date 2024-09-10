import sys
from pathlib import Path
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)

sys.path.append(str(Path(__file__).parents[3]))
from src.oncobot.talking_face import CustomSadTalker


video_folder = Path(__file__).parent / "video"
video_folder.mkdir(exist_ok=True)
video_path = str(video_folder / "chatbot__1.mp4")
audio_path = str(Path(__file__).parents[3] / "examples/fake_patient3.wav")
image_path = str(Path(__file__).parents[3] / "examples/chatbot1.jpg")


batch_size = 30
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=tensorboard_trace_handler("./log/sadtalker"),
) as prof:
    with record_function("model_init"):
        talker = CustomSadTalker(
            batch_size=batch_size,
            device=[1],
            # torch_dtype="float16",
            # parallel_mode="dp",
            # quanto_weights="int8",
            # quanto_activations=None,
        )
    with record_function("model_inference"):
        talker.run(
            video_path,
            audio_path,
            image_path,
            delete_generated_files=False,
        )
