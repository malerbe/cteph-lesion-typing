# Standard imports
import logging
import sys
import yaml

# External imports
import torch

# Local imports
from . import data
from . import models
from . import optim
from . import utils

logging.basicConfig(level=logging.INFO)

######################################################################
# train and helper functions
######################################################################
def train(config):
    """
    Main entry point to train based on the provided config.
    """
    if config["data"]["task"] in ["classification", "multihead_classification"]:
        _train_patch_classification(config)


def _train_patch_classification(config):
    """
    Train a classification model based on patches
    """

    logging.info("Training patch classification model...")

    ######################################
    # Config
    ######################################
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    data_config = config["data"]
    model_config = config["model"]
    loss_config = config["loss"]
    logging_config = config["logging"]
    
    ######################################
    # Dataloaders
    ######################################
    logging.info("= Building dataloaders...")

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(data_config, use_cuda)





if __name__ == "__main__":
    # Read arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|evaluate>")
        sys.exit(-1)

    logging.info("Loading {} configuration file".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
