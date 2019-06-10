import argparse
import yaml

from train import train
from utils import load_training_config_file

AVAILABLE_ACTIONS = ["train", "check", "test"]


def cast_to_type(type_abbrev, val):
    if type_abbrev == "f":
        return float(val)
    elif type_abbrev == "i":
        return int(val)
    elif type_abbrev == "b":
        return val.lower() in ("yes", "true", "t", "1")
    else:
        return val


def take_action(args):
    if args.action == "train":
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
    help="Override key value pairs in training configuration file",
)

args = parser.parse_args()

if args.override:
    for i in range(len(args.override) // 3):
        key, type_abbrev, val = args.override[i * 3 : (i + 1) * 3]
        args.train_config[key] = cast_to_type(type_abbrev, val)


take_action(args)
