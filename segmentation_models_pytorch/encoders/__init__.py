import functools
import torch.utils.model_zoo as model_zoo
import torch
from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders
from .inceptionresnetv2 import inceptionresnetv2_encoders
from .inceptionv4 import inceptionv4_encoders
from .efficientnet import efficient_net_encoders
from .mobilenet import mobilenet_encoders
from .xception import xception_encoders
from .timm_efficientnet import timm_efficientnet_encoders
from .timm_resnest import timm_resnest_encoders
from .timm_res2net import timm_res2net_encoders
from .timm_regnet import timm_regnet_encoders
from .timm_sknet import timm_sknet_encoders
from ._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        if weights.lower() != "imagenet":
            checkpoint = torch.load(weights, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}

            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('classifier') or k.startswith('projection_head') or k.startswith('prototypes') or k.startswith('encoder_k') or k.startswith("queue"):
                    del state_dict[k]
            encoder.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}'".format(weights))
        else:
            try:
                settings = encoders[name]["pretrained_settings"][weights.lower()]
            except KeyError:
                raise KeyError("Wrong pretrained weights `{}` for encoder `{}`. Available options are: {}".format(
                    weights, name, list(encoders[name]["pretrained_settings"].keys()),
                ))
            print ("settings url",settings["url"])
            if settings["url"].startswith("http"):
                encoder.load_state_dict(model_zoo.load_url(settings["url"]))
            else:
                encoder.load_state_dict(torch.load(settings["url"], map_location='cpu'))
            print("=> loaded supervised ImageNet pre-trained model")

    encoder.set_in_channels(in_channels)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
