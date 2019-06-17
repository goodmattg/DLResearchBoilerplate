import argparse
import yaml

from train import train
from utils import *


AVAILABLE_ACTIONS = ["train", "check", "test"]


def take_action(args):
    if args.action == "train":
        train(args.train_config)
    elif args.action == "check":
        # Check verifies that the model compiles by training for one epoch
        override_dotmap(["epochs", "i", 1], args.train_config)
        train(args.train_config)


parser = argparse.ArgumentParser(description="Interact with your research model")

parser.add_argument(
    "action", choices=AVAILABLE_ACTIONS, help="Do something with your model"
)

parser.add_argument(
    "--train-config",
    "-tc",
    type=load_training_config_file,
    default="gcn",
    help="Training configuration file. Must be YAML (.yaml) and stored in /config folder.",
)

parser.add_argument(
    "--override",
    "-o",
    nargs="*",
    help="Override key value pairs in training configuration file for ad-hoc testing",
)

args = parser.parse_args()

if args.override:
    override_dotmap(args.override, args.train_config)

take_action(args)
