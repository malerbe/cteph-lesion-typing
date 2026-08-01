# coding: utf-8

# External imports
import torch

# Local imports
from .cnn_models import *
from .foundation_models import *
from . import foundation_models


def build_model(model_cfg, input_size, num_classes):
    model_name = model_cfg['class']
    if model_name == "Basic3DCNN":
        return Basic3DCNN(num_classes=num_classes)

    raise ValueError(f"Unknown MODEL_NAME: {model_name}")