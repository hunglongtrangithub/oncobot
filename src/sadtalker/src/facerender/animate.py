import yaml
import warnings
import torch
import torch.nn as nn
import safetensors
import safetensors.torch
from optimum.quanto import quantize, freeze
from optimum import quanto

warnings.filterwarnings("ignore")


import torch


from .modules.keypoint_detector import HEEstimator, KPDetector
from .modules.mapping import MappingNet
from .modules.generator import OcclusionAwareSPADEGenerator
from .modules.make_animation import make_animation


try:
    import webui  # type: ignore # in webui

    in_webui = True
except:
    in_webui = False

ACCEPTED_WEIGHTS = {
    "float8": quanto.qfloat8,
    "int8": quanto.qint8,
    "int4": quanto.qint4,
    "int2": quanto.qint2,
}
ACCEPTED_ACTIVATIONS = {
    "none": None,
    "int8": quanto.qint8,
    "float8": quanto.qfloat8,
}
ACCEPTED_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
}


class AnimateFromCoeff:

    def __init__(
        self, sadtalker_paths, device, dtype=None, dp_device_ids=None, **quanto_config
    ):

        with open(sadtalker_paths["facerender_yaml"]) as f:
            config = yaml.safe_load(f)

        generator = OcclusionAwareSPADEGenerator(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"],
        )
        kp_extractor = KPDetector(
            **config["model_params"]["kp_detector_params"],
            **config["model_params"]["common_params"],
        )
        he_estimator = HEEstimator(
            **config["model_params"]["he_estimator_params"],
            **config["model_params"]["common_params"],
        )
        mapping = MappingNet(**config["model_params"]["mapping_params"])

        if dp_device_ids is not None:
            print("Using DataParallel with device ids:", dp_device_ids)
            generator = nn.DataParallel(
                generator,
                device_ids=dp_device_ids,
            )
            kp_extractor = nn.DataParallel(
                kp_extractor,
                device_ids=dp_device_ids,
            )
            he_estimator = nn.DataParallel(
                he_estimator,
                device_ids=dp_device_ids,
            )
            mapping = nn.DataParallel(
                mapping,
                device_ids=dp_device_ids,
            )

        for param in generator.parameters():
            param.requires_grad = False
        for param in kp_extractor.parameters():
            param.requires_grad = False
        for param in he_estimator.parameters():
            param.requires_grad = False
        for param in mapping.parameters():
            param.requires_grad = False

        self.quanto_config = quanto_config
        if self.quanto_config:
            print("Quanto config:", quanto_config)

        if sadtalker_paths is not None:
            if "checkpoint" in sadtalker_paths:  # use safe tensor
                self.load_cpk_facevid2vid_safetensor(
                    sadtalker_paths["checkpoint"],
                    kp_detector=kp_extractor,
                    generator=generator,
                    he_estimator=None,
                )
            else:
                self.load_cpk_facevid2vid(
                    sadtalker_paths["free_view_checkpoint"],
                    kp_detector=kp_extractor,
                    generator=generator,
                    he_estimator=he_estimator,
                )
        else:
            raise AttributeError(
                "Checkpoint should be specified for video head pose estimator."
            )

        if sadtalker_paths["mappingnet_checkpoint"] is not None:
            self.load_cpk_mapping(
                sadtalker_paths["mappingnet_checkpoint"], mapping=mapping
            )
        else:
            raise AttributeError(
                "Checkpoint should be specified for video head pose estimator."
            )

        self.kp_extractor = kp_extractor
        self.generator = generator
        self.he_estimator = he_estimator
        self.mapping = mapping

        self.dtype = torch.float32 if dtype is None else dtype

        self.kp_extractor.type(self.dtype)
        self.generator.type(self.dtype)
        self.he_estimator.type(self.dtype)
        self.mapping.type(self.dtype)

        self.kp_extractor.eval()
        self.generator.eval()
        self.he_estimator.eval()
        self.mapping.eval()

        generator.to(device)
        kp_extractor.to(device)
        he_estimator.to(device)
        mapping.to(device)

        self.device = device

        unit = "MB"
        print(
            f"OcclusionAwareSPADEGenerator model size: {self.get_model_size(self.generator, unit):.3f} {unit}"
        )
        print(
            f"KPDetector model size: {self.get_model_size(self.kp_extractor, unit):.3f} {unit}"
        )
        print(
            f"HEEstimator model size: {self.get_model_size(self.he_estimator, unit):.3f} {unit}"
        )
        print(
            f"MappingNet model size: {self.get_model_size(self.mapping, unit):.3f} {unit}"
        )
        print(
            "dtype:",
            self.dtype,
            "device:",
            self.device,
            "dp_device_ids:",
            dp_device_ids,
        )

    def get_model_size(self, model, unit="mb"):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all = param_size + buffer_size
        if unit.lower() == "mb":
            size_all /= 1024**2
        return size_all

    def _load_state_dict(self, model, state_dict):
        # If model is wrapped in DataParallel, load state_dict with 'module.' prefix
        if isinstance(model, nn.DataParallel):
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        if self.quanto_config:
            quantize(model, **self.quanto_config)
            freeze(model)
            print(f"Model quantized: {type(model)}")
            # for name, weight in model.named_parameters():
            #     print(f"{name} - {weight.type()}")

    def load_cpk_facevid2vid_safetensor(
        self,
        checkpoint_path,
        generator=None,
        kp_detector=None,
        he_estimator=None,
    ):
        print("Loading checkpoint from", checkpoint_path)
        checkpoint = safetensors.torch.load_file(checkpoint_path)

        if generator is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if "generator" in k:
                    x_generator[k.replace("generator.", "")] = v
            self._load_state_dict(generator, x_generator)
        if kp_detector is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if "kp_extractor" in k:
                    x_generator[k.replace("kp_extractor.", "")] = v
            self._load_state_dict(kp_detector, x_generator)
        if he_estimator is not None:
            x_generator = {}
            for k, v in checkpoint.items():
                if "he_estimator" in k:
                    x_generator[k.replace("he_estimator.", "")] = v
            self._load_state_dict(he_estimator, x_generator)

        return None

    def load_cpk_facevid2vid(
        self,
        checkpoint_path,
        generator=None,
        discriminator=None,
        kp_detector=None,
        he_estimator=None,
        optimizer_generator=None,
        optimizer_discriminator=None,
        optimizer_kp_detector=None,
        optimizer_he_estimator=None,
        device="cpu",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if generator is not None:
            self._load_state_dict(generator, checkpoint["generator"])
        if kp_detector is not None:
            self._load_state_dict(kp_detector, checkpoint["kp_detector"])
        if he_estimator is not None:
            self._load_state_dict(he_estimator, checkpoint["he_estimator"])
        if discriminator is not None:
            try:
                discriminator.load_state_dict(checkpoint["discriminator"])
            except:
                print(
                    "No discriminator in the state-dict. Dicriminator will be randomly initialized"
                )
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
        if optimizer_discriminator is not None:
            try:
                optimizer_discriminator.load_state_dict(
                    checkpoint["optimizer_discriminator"]
                )
            except RuntimeError:
                print(
                    "No discriminator optimizer in the state-dict. Optimizer will be not initialized"
                )
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint["optimizer_kp_detector"])
        if optimizer_he_estimator is not None:
            optimizer_he_estimator.load_state_dict(checkpoint["optimizer_he_estimator"])

        return checkpoint["epoch"]

    def load_cpk_mapping(
        self,
        checkpoint_path,
        mapping=None,
        discriminator=None,
        optimizer_mapping=None,
        optimizer_discriminator=None,
        device="cpu",
    ):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if mapping is not None:
            self._load_state_dict(mapping, checkpoint["mapping"])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint["discriminator"])
        if optimizer_mapping is not None:
            optimizer_mapping.load_state_dict(checkpoint["optimizer_mapping"])
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(
                checkpoint["optimizer_discriminator"]
            )

        return checkpoint["epoch"]

    def generate(self, x):

        source_image = x["source_image"].type(self.dtype)
        source_semantics = x["source_semantics"].type(self.dtype)
        target_semantics = x["target_semantics_list"].type(self.dtype)
        source_image = source_image.to(self.device)
        source_semantics = source_semantics.to(self.device)
        target_semantics = target_semantics.to(self.device)
        if "yaw_c_seq" in x:
            yaw_c_seq = x["yaw_c_seq"].type(self.dtype)
            yaw_c_seq = x["yaw_c_seq"].to(self.device)
        else:
            yaw_c_seq = None
        if "pitch_c_seq" in x:
            pitch_c_seq = x["pitch_c_seq"].type(self.dtype)
            pitch_c_seq = x["pitch_c_seq"].to(self.device)
        else:
            pitch_c_seq = None
        if "roll_c_seq" in x:
            roll_c_seq = x["roll_c_seq"].type(self.dtype)
            roll_c_seq = x["roll_c_seq"].to(self.device)
        else:
            roll_c_seq = None
        frame_num = x["frame_num"]

        predictions_video = make_animation(
            source_image,
            source_semantics,
            target_semantics,
            self.generator,
            self.kp_extractor,
            self.he_estimator,
            self.mapping,
            yaw_c_seq,
            pitch_c_seq,
            roll_c_seq,
        )

        predictions_video = predictions_video.reshape(
            (-1,) + predictions_video.shape[2:]
        )
        predictions = predictions_video[:frame_num]

        return predictions
