from scipy.spatial import ConvexHull
import torch
import numpy as np
from tqdm import tqdm
from src.utils.logger_config import logger


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial["value"][0].data.cpu().numpy()
        ).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


# @profile
def headpose_pred_to_degree(pred):  # slow
    idx_tensor = torch.arange(pred.shape[1], device=pred.device)
    # NOTE: doesn't require pred and idx_tensor to have the same dtype
    degree = torch.sum(pred.softmax(1) * idx_tensor, 1) * 3 - 99
    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat(
        [
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.cos(pitch),
            -torch.sin(pitch),
            torch.zeros_like(pitch),
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    )
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ],
        dim=1,
    )
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            torch.zeros_like(roll),
            torch.sin(roll),
            torch.cos(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.ones_like(roll),
        ],
        dim=1,
    )
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation(
    kp_canonical,
    he,
    wo_exp=False,
):
    kp = kp_canonical["value"]  # (bs, k, 3)
    yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]

    if "yaw_in" in he:
        yaw = he["yaw_in"]
    else:
        yaw = headpose_pred_to_degree(yaw)
    if "pitch_in" in he:
        pitch = he["pitch_in"]
    else:
        pitch = headpose_pred_to_degree(pitch)
    if "roll_in" in he:
        roll = he["roll_in"]
    else:
        roll = headpose_pred_to_degree(roll)
    # synchronize?
    # torch.cuda.synchronize()
    rot_mat = get_rotation_matrix(yaw, pitch, roll)  # (bs, 3, 3)

    t, exp = he["t"], he["exp"]
    if wo_exp:
        exp = exp * 0

    logger.debug(f"t: {t.dtype}, exp: {exp.dtype}, kp: {kp.dtype}")
    # keypoint rotation
    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)
    logger.debug(f"kp_rotated: {kp_rotated.dtype}")
    # keypoint translation
    t[:, 0] = t[:, 0] * 0
    t[:, 2] = t[:, 2] * 0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {"value": kp_transformed}


# @profile
def make_animation(
    source_image,
    source_semantics,
    target_semantics,
    generator,
    kp_detector,
    he_estimator,
    mapping,
    yaw_c_seq=None,
    pitch_c_seq=None,
    roll_c_seq=None,
    use_exp=False,
    use_half=False,
):
    with torch.no_grad():
        predictions = []
        logger.info("Calculating kp_canonical")
        kp_canonical = kp_detector(source_image)
        logger.info("Calculating he_source")
        he_source = mapping(source_semantics)
        logger.info("Calculating kp_source")
        kp_source = keypoint_transformation(kp_canonical, he_source)
        for frame_idx in tqdm(range(target_semantics.shape[1]), "Face Renderer:"):
            target_semantics_frame = target_semantics[:, frame_idx]
            # source_semantics and target_semantics_frame always have the same shape.
            he_driving = mapping(target_semantics_frame)
            if yaw_c_seq is not None:
                he_driving["yaw_in"] = yaw_c_seq[:, frame_idx]
            if pitch_c_seq is not None:
                he_driving["pitch_in"] = pitch_c_seq[:, frame_idx]
            if roll_c_seq is not None:
                he_driving["roll_in"] = roll_c_seq[:, frame_idx]

            kp_driving = keypoint_transformation(kp_canonical, he_driving)
            out = generator(source_image, kp_source=kp_source, kp_driving=kp_driving)
            predictions.append(out["prediction"])
        predictions_ts = torch.stack(predictions, dim=1)
    return predictions_ts
