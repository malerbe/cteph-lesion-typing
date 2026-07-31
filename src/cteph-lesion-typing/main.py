# Standard imports
import logging
import sys
import yaml


import torch

# Local imports
from . import models
from . import data

def train(config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")







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
