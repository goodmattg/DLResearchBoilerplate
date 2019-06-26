import argparse
import yaml
import sys
import traceback
import os
import pdb

from train import train
from utils.argparse import *
from utils.file import load_training_config_file
from scripts import *

## SET ME BEFORE MOVING ON!
    # --evaluate               example evaluation mode using a restored model

AVAILABLE_ACTIONS = ["train", "check", "test"]


def take_action(args):
    if args.action == "train":
        train(args.config_file)

    elif args.action == "check":
        # Check verifies that the model compiles by training for one epoch
        override_dotmap(["epochs", "i", 1], args.config_file)
        train(args.config_file)


parser = argparse.ArgumentParser(description="Interact with your research model")

parser.add_argument(
    "action", choices=AVAILABLE_ACTIONS, help="Take an action with your model (e.g. 'train')"
)

parser.add_argument(
    "--config-file",
    type=load_training_config_file,
    default="config/default.yaml",
    help="Configuration file absolute path",
)

parser.add_argument(
    "--log-dir",
    type=dir_exists_write_privileges,
    default=".",
    help="Log file storage directory path"
)

parser.add_argument(
    "--data-dir",
    type=dir_exists_read_privileges,
    default="data",
    help="Data file storage directory path"
)


parser.add_argument(
    "--restore-weights",
    type=file_exists,
    help="Restore model with pre-trained weights" 
)

parser.add_argument(
    "--override",
    "-o",
    nargs="*",
    help="Override key value pairs in training configuration file for ad-hoc testing",
)

parser.add_argument("--freeze-graph-model", action="store_true")

args = parser.parse_args()

if args.override:
    override_dotmap(args.override, args.config_file)

try:
    take_action(args)
except:
    typ, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)



