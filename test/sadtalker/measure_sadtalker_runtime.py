import time
import csv
import statistics
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[2]))
from src.oncobot.talking_face import CustomSadTalker

# Initialize the model
talker = CustomSadTalker(
    batch_size=60,
    device=[1],
)

video_folder = Path(__file__).parent / "video"
video_folder.mkdir(exist_ok=True)
image_path = str(Path(__file__).parents[2] / "examples/chatbot1.jpg")

# Paths to the audio clips
audio_paths = [
    (str(Path(__file__).parents[2] / "examples/fake_patient3.wav"), "fake_patient3"),
    (str(Path(__file__).parents[2] / "examples/ellie_10s.wav"), "ellie_10s"),
    (str(Path(__file__).parents[2] / "examples/ellie_30s.wav"), "ellie_30s"),
    # (str(Path(__file__).parents[2] / "examples/ellie_60s.wav"), "ellie_60s"),
]

# Parameters
num_runs = 10  # Number of times to run the model for each audio clip
csv_file = str(video_folder / "runtime_data.csv")

# Run the model and record the runtime for each audio clip
run_times = {audio_path[1]: [] for audio_path in audio_paths}

for i in range(num_runs):
    for j, audio_path in enumerate(audio_paths):
        video_path = str(video_folder / f"chatbot__{audio_path[1]}.mp4")
        start_time = time.perf_counter()

        # Run the pipeline
        talker.run(
            video_path,
            audio_path[0],
            image_path,
            delete_generated_files=False,
        )

        end_time = time.perf_counter()
        run_time = end_time - start_time
        run_times[audio_path[1]].append(run_time)
        print(
            f"Run {i + 1}/{num_runs} for Audio Clip {audio_path[1]}: {run_time:.4f} seconds"
        )

# Write the runtime data to a CSV file
with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(run_times.keys())
    writer.writerows(zip(*run_times.values()))

print(f"Runtime data saved to {csv_file}")

# Calculate and display statistics for each audio clip
for audio_path, times in run_times.items():
    mean_time = statistics.mean(times)
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\nStatistics for Audio Clip {audio_path}:")
    print(f"Mean Time: {mean_time:.4f} seconds")
    print(f"Median Time: {median_time:.4f} seconds")
    print(f"Standard Deviation: {std_dev:.4f} seconds")
    print(f"Min Time: {min_time:.4f} seconds")
    print(f"Max Time: {max_time:.4f} seconds")

