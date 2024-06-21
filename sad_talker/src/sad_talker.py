import gc
import os, shutil
import torch, uuid
from sad_talker.src.utils.preprocess import CropAndExtract
from sad_talker.src.test_audio2coeff import Audio2Coeff
from sad_talker.src.facerender.animate import AnimateFromCoeff
from sad_talker.src.generate_batch import get_data
from sad_talker.src.generate_facerender_batch import get_facerender_data

from sad_talker.src.utils.init_path import init_path

from pydub import AudioSegment
import time


def mp3_to_wav(mp3_filename, wav_filename, frame_rate):
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


class SadTalker:

    def __init__(
        self,
        checkpoint_path="checkpoints",
        config_path="src/config",
        image_size=256,
        image_preprocess="crop",
        lazy_load=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ["TORCH_HOME"] = checkpoint_path
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.image_size = image_size
        self.image_preprocess = image_preprocess

        self.models_loaded = False
        if not lazy_load:
            self.load()

    def load(self):
        if not self.models_loaded:
            print("Loading SadTalker models")
            self.sadtalker_paths = init_path(
                self.checkpoint_path,
                self.config_path,
                self.image_size,
                False,
                self.image_preprocess,
            )
            self.audio_to_coeff = Audio2Coeff(self.sadtalker_paths, self.device)
            self.preprocess_model = CropAndExtract(self.sadtalker_paths, self.device)
            self.animate_from_coeff = AnimateFromCoeff(
                self.sadtalker_paths, self.device
            )
            print("SadTalker models loaded. Device:", self.device)
            self.models_loaded = True

    def test(
        self,
        source_image,
        driven_audio,
        still_mode=False,
        use_enhancer=False,
        batch_size=1,
        pose_style=0,
        exp_scale=1.0,
        # use_ref_video=False,
        # ref_video=None,
        # ref_info=None,
        use_idle_mode=False,
        length_of_audio=0,
        use_blink=True,
        result_dir="./results/",
    ):
        self.load()

        time_tag = str(uuid.uuid4())
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        pic_path = os.path.join(input_dir, os.path.basename(source_image))
        shutil.copy(source_image, input_dir)

        if os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))

            #### mp3 to wav
            if ".mp3" in audio_path:
                mp3_to_wav(driven_audio, audio_path.replace(".mp3", ".wav"), 16000)
                audio_path = audio_path.replace(".mp3", ".wav")
            else:
                shutil.copy(driven_audio, input_dir)

        elif use_idle_mode:
            audio_path = os.path.join(
                input_dir, "idlemode_" + str(length_of_audio) + ".wav"
            )  ## generate audio from this new audio_path
            one_sec_segment = AudioSegment.silent(
                duration=1000 * length_of_audio
            )  # duration in milliseconds
            one_sec_segment.export(audio_path, format="wav")
        # else:
        #     print(use_ref_video, ref_info)
        #     assert use_ref_video == True and ref_info == "all" and ref_video is not None
        #     ref_video_videoname = os.path.basename(ref_video)
        #     audio_path = os.path.join(save_dir, ref_video_videoname + ".wav")
        #     print("new audiopath:", audio_path)
        #     # if ref_video contains audio, set the audio from ref_video.
        #     cmd = r"ffmpeg -y -hide_banner -loglevel error -i %s %s" % (
        #         ref_video,
        #         audio_path,
        #     )
        #     os.system(cmd)

        os.makedirs(save_dir, exist_ok=True)

        # crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)
        start = time.time()
        first_coeff_path, crop_pic_path, crop_info = self.preprocess_model.generate(
            pic_path,
            first_frame_dir,
            self.image_preprocess,
            True,
            self.image_size,
        )
        print("self.preprocess_model.generate:", time.time() - start)

        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        ref_video_coeff_path = None
        # if use_ref_video:
        #     assert ref_video is not None
        #     print("using ref video for generation")
        #     ref_video_videoname = os.path.splitext(os.path.split(ref_video)[-1])[0]
        #     ref_video_frame_dir = os.path.join(save_dir, ref_video_videoname)
        #     os.makedirs(ref_video_frame_dir, exist_ok=True)
        #     print("3DMM Extraction for the reference video providing pose")
        #     start = time.time()
        #     ref_video_coeff_path, _, _ = self.preprocess_model.generate(
        #         ref_video,
        #         ref_video_frame_dir,
        #         self.image_preprocess,
        #         source_image_flag=False,
        #     )
        #     print("self.preprocess_model.generate:", time.time() - start)

        ref_pose_coeff_path = None
        ref_eyeblink_coeff_path = None
        # if use_ref_video:
        #     if ref_info == "pose":
        #         ref_pose_coeff_path = ref_video_coeff_path
        #         ref_eyeblink_coeff_path = None
        #     elif ref_info == "blink":
        #         ref_pose_coeff_path = None
        #         ref_eyeblink_coeff_path = ref_video_coeff_path
        #     elif ref_info == "pose+blink":
        #         ref_pose_coeff_path = ref_video_coeff_path
        #         ref_eyeblink_coeff_path = ref_video_coeff_path
        #     elif ref_info == "all":
        #         ref_pose_coeff_path = None
        #         ref_eyeblink_coeff_path = None
        #     else:
        #         raise ValueError(
        #             "Invalid ref_info. Must be one of pose, blink, pose+blink, all."
        #         )

        # # audio2ceoff
        # if use_ref_video and ref_info == "all":
        #     coeff_path = ref_video_coeff_path  # self.audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)
        # else:
        start = time.time()
        batch = get_data(
            first_coeff_path,
            audio_path,
            self.device,
            ref_eyeblink_coeff_path=ref_eyeblink_coeff_path,
            still=still_mode,
            idlemode=use_idle_mode,
            length_of_audio=length_of_audio,
            use_blink=use_blink,
        )  # longer audio?
        coeff_path = self.audio_to_coeff.generate(
            batch,
            save_dir,
            pose_style,
            ref_pose_coeff_path,
        )
        print("self.audio_to_coeff.generate:", time.time() - start)

        # coeff2video
        start = time.time()
        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=self.image_preprocess,
            size=self.image_size,
            expression_scale=exp_scale,
        )
        print("get_facerender_data:", time.time() - start)
        start = time.time()
        return_path = self.animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer="gfpgan" if use_enhancer else None,
            preprocess=self.image_preprocess,
            img_size=self.image_size,
        )
        print("self.animate_from_coeff.generate:", time.time() - start)
        video_name = data["video_name"]
        print(f"The generated video is named {video_name} in {save_dir}")

        return return_path

    def clean(self):
        del self.preprocess_model
        del self.audio_to_coeff
        del self.animate_from_coeff

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()
