import sys
from pathlib import Path
from viztracer import VizTracer

sys.path.append(str(Path(__file__).parents[3]))
from src.oncobot.talking_face import CustomSadTalker


talker = CustomSadTalker(
    batch_size=30,
    device=[0],
    # torch_dtype="float16",
    # parallel_mode="dp",
    # quanto_weights="int8",
    # quanto_activations=None,
)
video_folder = Path(__file__).parent / "video"
video_folder.mkdir(exist_ok=True)
video_path = str(video_folder / "chatbot__1.mp4")
audio_path = str(Path(__file__).parents[3] / "examples/fake_patient3.wav")
image_path = str(Path(__file__).parents[3] / "examples/chatbot1.jpg")

with VizTracer(
    tracer_entries=10001000, output_file="viztracer/sadtalker.json"
) as tracer:
    talker.run(
        video_path,
        audio_path,
        image_path,
        delete_generated_files=False,
    )
