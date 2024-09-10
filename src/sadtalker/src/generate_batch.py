import os
import torchaudio
from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
from scipy import signal
from .utils.hparams import hparams as hp


def crop_pad_audio(wav, audio_length):  # works for torch tensors as well
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(
            wav, [0, audio_length - len(wav)], mode="constant", constant_values=0
        )
    return wav


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames


def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames, 1))
    frame_id = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id + start + 9 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 9, 0] = [
                0.5,
                0.6,
                0.7,
                0.9,
                1,
                0.9,
                0.7,
                0.6,
                0.5,
            ]
            frame_id = frame_id + start + 9
        else:
            break
    return ratio


def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames, 1))
    if num_frames <= 20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10, num_frames), min(int(num_frames / 2), 70)))
        if frame_id + start + 5 <= num_frames - 1:
            ratio[frame_id + start : frame_id + start + 5, 0] = [
                0.5,
                0.9,
                1.0,
                0.9,
                0.5,
            ]
            frame_id = frame_id + start + 5
        else:
            break
    return ratio


SAMPLE_RATE = hp.sample_rate
SYNCNET_MEL_STEP_SIZE = 16
FPS = hp.fps

N_FFT = hp.n_fft
HOP_SIZE = hp.hop_size
WIN_SIZE = hp.win_size
MEL_FREQ_BINS = hp.num_mels
NORMALIZE = hp.signal_normalization


def load_wav(wav_path, sr) -> torch.Tensor:
    wav, samplerate = torchaudio.load(wav_path)
    if samplerate != sr:
        wav = torchaudio.transforms.Resample(samplerate, sr)(wav)
    return wav


def preemphasis(wav, k=hp.preemphasis, preemphasize=hp.preemphasize):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def melspectrogram(wav, n_fft, win_length, hop_length, n_mels):
    # Apply preemphasis
    wav = preemphasis(wav)  # very important

    # Convert the numpy array to a PyTorch tensor
    wav = torch.tensor(wav, dtype=torch.float32)

    # Create the MelSpectrogram transform
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        normalized=True,
    )

    # Generate the mel-spectrogram
    S = mel_spectrogram_transform(wav)

    # Convert to dB
    S = torchaudio.transforms.AmplitudeToDB()(S)

    # Normalize as in the original code
    S = normalize(S)  # very important

    return S


def normalize(
    S,
    min_level_db=hp.min_level_db,
    max_abs_value=hp.max_abs_value,
    symmetric_mels=hp.symmetric_mels,
    allow_clipping_in_normalization=hp.allow_clipping_in_normalization,
):
    if allow_clipping_in_normalization:
        if symmetric_mels:
            return np.clip(
                (2 * max_abs_value) * ((S - min_level_db) / (-min_level_db))
                - max_abs_value,
                -max_abs_value,
                max_abs_value,
            )
        else:
            return np.clip(
                max_abs_value * ((S - min_level_db) / (-min_level_db)), 0, max_abs_value
            )

    assert S.max() <= 0 and S.min() - min_level_db >= 0
    if symmetric_mels:
        return (2 * max_abs_value) * (
            (S - min_level_db) / (-min_level_db)
        ) - max_abs_value
    else:
        return max_abs_value * ((S - min_level_db) / (-min_level_db))


def get_data(
    first_coeff_path,
    audio_path,
    device,
    ref_eyeblink_coeff_path=None,
    still=False,
    idlemode=False,
    length_of_audio=0,
    use_blink=True,
):
    pic_name = os.path.splitext(os.path.basename(first_coeff_path))[0]
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]

    if idlemode:
        num_frames = int(length_of_audio * FPS)
        indiv_mels = torch.zeros((num_frames, MEL_FREQ_BINS, SYNCNET_MEL_STEP_SIZE))
    else:
        # process audio
        # start = time.time()
        wav_tensor = load_wav(audio_path, SAMPLE_RATE)
        wav = wav_tensor[0]
        # print(f"load_wav time: {time.time() - start}")
        wav_length, num_frames = parse_audio_length(len(wav), SAMPLE_RATE, FPS)
        wav = crop_pad_audio(wav, wav_length)
        # start = time.time()
        orig_mel = melspectrogram(wav, N_FFT, WIN_SIZE, HOP_SIZE, MEL_FREQ_BINS).T
        # print(f"melspectrogram time: {time.time() - start}")
        indiv_mels = np.empty((num_frames, MEL_FREQ_BINS, SYNCNET_MEL_STEP_SIZE))

        # extract mel seg
        for i in tqdm(range(num_frames), "mel:"):
            start_frame_num = i - 2
            start_idx = max(int(MEL_FREQ_BINS * (start_frame_num / float(FPS))), 0)
            end_idx = start_idx + SYNCNET_MEL_STEP_SIZE
            m = orig_mel[start_idx:end_idx, :].T
            m = np.pad(
                m,
                ((0, 0), (0, max(0, SYNCNET_MEL_STEP_SIZE - m.shape[1]))),
                mode="edge",
            )
            indiv_mels[i] = m

    ratio = generate_blink_seq_randomly(num_frames)  # T
    source_semantics_dict = scio.loadmat(first_coeff_path)
    ref_coeff = np.repeat(
        source_semantics_dict["coeff_3dmm"][:1, :70], num_frames, axis=0
    )

    if ref_eyeblink_coeff_path is not None:
        refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
        refeyeblink_coeff = refeyeblink_coeff_dict["coeff_3dmm"][:, :64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        if refeyeblink_num_frames < num_frames:
            refeyeblink_coeff = np.tile(
                refeyeblink_coeff, (num_frames // refeyeblink_num_frames, 1)
            )
            refeyeblink_coeff = np.vstack(
                [
                    refeyeblink_coeff,
                    refeyeblink_coeff[: num_frames % refeyeblink_num_frames, :],
                ]
            )

        ref_coeff[:, :64] = refeyeblink_coeff[:num_frames]
        ratio[:num_frames] = 0

    # Convert to torch tensors and move to the specified device
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0).to(device)
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0).to(device)
    ratio = torch.FloatTensor(ratio).unsqueeze(0).to(device)

    if not use_blink:
        ratio.fill_(0.0)

    return {
        "wav": wav_tensor,
        "indiv_mels": indiv_mels,
        "ref": ref_coeff,
        "num_frames": num_frames,
        "ratio_gt": ratio,
        "audio_name": audio_name,
        "pic_name": pic_name,
    }
