import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import (
    vgg16,
    vgg16_bn,
    VGG16_Weights,
    VGG16_BN_Weights,
    resnet50,
    ResNet50_Weights,
)
from models.backbones.pvt_v2 import (
    pvt_v2_b0,
    pvt_v2_b1,
    pvt_v2_b2,
    pvt_v2_b3,
    pvt_v2_b4,
    pvt_v2_b5,
)


def build_backbone(config, bb_name, pretrained=True, params_settings=""):
    bb = eval("{}({})".format(bb_name, params_settings))
    if pretrained:
        bb = load_weights(config, bb, bb_name)
    return bb


def load_weights(config, model, model_name):
    weight_path = getattr(config.weights, model_name)

    save_model = torch.load(
        weight_path, map_location="cpu", weights_only=True
    )
    model_dict = model.state_dict()
    state_dict = {
        k: v if v.size() == model_dict[k].size() else model_dict[k]
        for k, v in save_model.items()
        if k in model_dict.keys()
    }
    # to ignore the weights with mismatched size when I modify the backbones itself.
    if not state_dict:
        save_model_keys = list(save_model.keys())
        sub_item = save_model_keys[0] if len(save_model_keys) == 1 else None
        state_dict = {
            k: v if v.size() == model_dict[k].size() else model_dict[k]
            for k, v in save_model[sub_item].items()
            if k in model_dict.keys()
        }
        if not state_dict or not sub_item:
            print(
                "Weights are not successfully loaded. Check the state dict of weights file."
            )
            return None
        else:
            print(
                'Found correct weights in the "{}" item of loaded state_dict.'.format(
                    sub_item
                )
            )
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model
