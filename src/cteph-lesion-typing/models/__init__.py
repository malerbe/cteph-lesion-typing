# coding: utf-8

# External imports
import torch

# Local imports
from .cnn_models import *
from .foundation_models import *
from . import foundation_models


# Map model class names to factory callables.
# Models whose __init__ doesn't follow (cfg, input_size, num_classes)
# need an explicit factory registered here.
_MODEL_FACTORIES = {
    "VoCoClassifier": foundation_models.VoCoClassifier_factory,
    "OriginalVoCoClassifier": foundation_models.OriginalVoCoClassifier_factory,
    "VoCoClassifierSeg": foundation_models.VoCoClassifierSeg_factory,
}


def build_model(cfg, input_size, num_classes):
    class_name = cfg["class"]
    if class_name in _MODEL_FACTORIES:
        return _MODEL_FACTORIES[class_name](cfg, input_size, num_classes)
    return eval(f"{class_name}(cfg, input_size, num_classes)")
