import shutil
import uuid
import time
import cv2, os
import numpy as np
from tqdm import tqdm
import uuid
import imageio
import numpy as np
from skimage import img_as_ubyte
from pydub import AudioSegment

import torch
import torchvision
from torchaudio.io import StreamWriter
import os

import cv2
import subprocess
from ..utils.hparams import hparams as hp


def is_ffmpeg_installed():
    try:
        # Run the ffmpeg command to check if it is installed, suppressing output
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if result.returncode == 0:
            return True
        else:
            return False
    except:
        return False


def save_video_with_watermark(video, audio, save_path, watermark=False):
    if not is_ffmpeg_installed():
        raise Exception(
            "ffmpeg is not installed or not on PATH. Please install ffmpeg first."
        )

    temp_file = str(uuid.uuid4()) + ".mp4"
    cmd = (
        r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -vcodec copy "%s"'
        % (video, audio, temp_file)
    )
    os.system(cmd)

    if watermark is False:
        shutil.move(temp_file, save_path)
    else:
        # watermark
        try:
            ##### check if stable-diffusion-webui
            import webui
            from modules import paths

            watermark_path = (
                paths.script_path + "/extensions/SadTalker/docs/sadtalker_logo.png"
            )
        except:
            # get the root path of sadtalker.
            dir_path = os.path.dirname(os.path.realpath(__file__))
            watermark_path = dir_path + "/../../docs/sadtalker_logo.png"

        cmd = (
            r'ffmpeg -y -hide_banner -loglevel error -i "%s" -i "%s" -filter_complex "[1]scale=100:-1[wm];[0][wm]overlay=(main_w-overlay_w)-10:10" "%s"'
            % (temp_file, watermark_path, save_path)
        )
        os.system(cmd)
        os.remove(temp_file)


def paste_pic(
    video_path,
    pic_path,
    crop_info,
    new_audio_path,
    full_video_path,
    extended_crop=False,
):

    if not os.path.isfile(pic_path):
        raise ValueError("pic_path must be a valid path to video/image file")
    elif pic_path.split(".")[-1] in ["jpg", "png", "jpeg"]:
        # loader for first frame
        full_img = cv2.imread(pic_path)
    else:
        # loader for videos
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        full_frames = []
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
        full_img = frame
    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)

    if len(crop_info) != 3:
        print("you didn't crop the image")
        return
    else:
        r_w, r_h = crop_info[0]
        clx, cly, crx, cry = crop_info[1]
        lx, ly, rx, ry = crop_info[2]
        lx, ly, rx, ry = int(lx), int(ly), int(rx), int(ry)
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx
        # oy1, oy2, ox1, ox2 = cly+ly, cly+ry, clx+lx, clx+rx

        if extended_crop:
            oy1, oy2, ox1, ox2 = cly, cry, clx, crx
        else:
            oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx

    tmp_path = str(uuid.uuid4()) + ".mp4"
    out_tmp = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter.fourcc(*"MP4V"), fps, (frame_w, frame_h)
    )
    for crop_frame in tqdm(crop_frames, "seamlessClone:"):
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2 - ox1, oy2 - oy1))

        mask = 255 * np.ones(p.shape, p.dtype)
        location = ((ox1 + ox2) // 2, (oy1 + oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        out_tmp.write(gen_img)

    out_tmp.release()

    save_video_with_watermark(
        tmp_path, new_audio_path, full_video_path, watermark=False
    )
    os.remove(tmp_path)


def load_video_to_cv2(input_path):
    video_stream = cv2.VideoCapture(input_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        full_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return full_frames


def resize_video(predictions_video, crop_info, img_size):
    original_size = crop_info[0]
    resized_video = torchvision.transforms.functional.resize(
        predictions_video,
        (img_size, int(img_size * original_size[1] / original_size[0])),
    )
    return resized_video  # [T, C, H, W]2


def write_video_gpu(video_path, video_data, audio_data, device, fps, width, height):
    audio_data = audio_data.permute(1, 0)
    writer = StreamWriter(video_path)
    cuda_conf = {
        "format": "rgb24",
        "encoder": "h264_nvenc",  # Use CUDA HW decoder
        "encoder_format": "",
        "encoder_option": {"gpu": "0"},
        "hw_accel": f"cuda:{device.index}",
    }
    writer.add_audio_stream(hp.sample_rate, 1)
    writer.add_video_stream(fps, width, height, **cuda_conf)
    with writer.open():
        print(
            "Writing video... Audio shape:",
            audio_data.shape,
            "Video shape:",
            video_data.shape,
        )
        writer.write_audio_chunk(0, audio_data)
        writer.write_video_chunk(1, video_data.to(torch.uint8))


def write_video(video_path, video_data, audio_data, device, fps, width, height):
    video_data = video_data.permute(0, 2, 3, 1).cpu()
    audio_data = audio_data.cpu()
    save_video_with_watermark(video_data, audio_data, video_path)


def save_data_to_video(
    video_name,
    audio_data,
    device,
    predictions_video,
    crop_info,
    img_size,
    video_save_dir,
):
    # Prepare paths
    video_name = video_name + ".mp4"
    # temp_video_path = os.path.join(video_save_dir, "temp_" + video_name)
    final_video_path = os.path.join(video_save_dir, video_name)

    # Resize video
    start_time = time.time()
    video_data = resize_video(predictions_video, crop_info, img_size)
    print(f"Resizing time: {time.time() - start_time:.2f}s")

    start_time = time.time()
    write_video(
        final_video_path,
        video_data,
        audio_data,
        device,
        hp.fps,
        video_data.shape[2],
        video_data.shape[1],
    )
    print(f"Video saving time: {time.time() - start_time:.2f}s")

    return final_video_path


def save_data_to_video(
    video_name,
    audio_path,
    predictions_video,
    crop_info,
    img_size,
    frame_num,
    preprocess,
    pic_path,
    video_save_dir,
):
    predictions_video = predictions_video.cpu().numpy()
    video = np.transpose(predictions_video, [0, 2, 3, 1]).astype(np.float32)
    result = img_as_ubyte(video)

    original_size = crop_info[0]
    if original_size:
        result = [
            cv2.resize(
                result_i,
                (img_size, int(img_size * original_size[1] / original_size[0])),
            )
            for result_i in result
        ]

    video_name = video_name + ".mp4"
    path = os.path.join(video_save_dir, "temp_" + video_name)

    imageio.mimsave(path, result, fps=float(25))

    av_path = os.path.join(video_save_dir, video_name)
    return_path = av_path

    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
    new_audio_path = os.path.join(video_save_dir, audio_name + ".wav")
    start_time = 0
    sound = AudioSegment.from_file(audio_path)
    frames = frame_num
    end_time = start_time + frames * 1 / 25 * 1000
    word1 = sound.set_frame_rate(16000)
    word = word1[start_time:end_time]
    word.export(new_audio_path, format="wav")

    save_video_with_watermark(path, new_audio_path, av_path, watermark=False)
    print(f"The generated video is named {video_save_dir}/{video_name}")

    if "full" in preprocess.lower():
        video_name_full = video_name + "_full.mp4"
        full_video_path = os.path.join(video_save_dir, video_name_full)
        return_path = full_video_path
        paste_pic(
            path,
            pic_path,
            crop_info,
            new_audio_path,
            full_video_path,
            extended_crop=True if "ext" in preprocess.lower() else False,
        )
        print(f"The generated video is named {video_save_dir}/{video_name_full}")
    else:
        full_video_path = av_path

    os.remove(path)
    os.remove(new_audio_path)

    return return_path
