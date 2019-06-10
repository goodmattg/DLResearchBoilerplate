import argparse
import yaml

from train import train
from utils import load_training_config_file

AVAILABLE_ACTIONS=[
    "train",
    "check",
    "test"
]

def take_action(args):
    print(args)
    if args.action == "train":
        train(args.train_config)

parser = argparse.ArgumentParser(description="Interact with your research model")

parser.add_argument(
    "action",
    choices=AVAILABLE_ACTIONS,
    help="Do something with your model")

parser.add_argument(
    "--train-config",
    "-tc",
    type=load_training_config_file,
    default="gcn",
    help="Training configuration file. Must be YAML (.yaml) and stored in /config folder."
)

args = parser.parse_args()
take_action(args)