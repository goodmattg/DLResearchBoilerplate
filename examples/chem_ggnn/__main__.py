import yaml
import sys
import traceback
import os
import pdb

from models import *

from argparse import ArgumentParser, ArgumentTypeError
from train import train
from evaluate import evaluate

from scripts.common_args import add_common_arguments
from scripts.chem_ggnn_args import add_ggnn_specific_arguments

from utils.misc import get_run_id


def take_action(args):

    # Instantiate the model.
    # Constants should change depending on research structure
    # For single model project, can remove if/else and switch to plain instantiate
    if args.model == "ggnn-dense":
        model = ChemModel(args)
    else:
        raise (
            ArgumentTypeError(
                "Not supplied with recognized model argument. Check spelling (case-sensitive)"
            )
        )

    print("Taking action...")
    if args.action == "train":
        train(args.config_file)
    elif args.action == "check":
        # Check verifies that the model compiles by training for one epoch
        override_dotmap(["epochs", "i", 1], args.config_file)
        train(args.config_file)
    elif args.action == "evaluate":
        evaluate(args.config_file)

    # Create the argument parser with command and example specific arguments


parser = ArgumentParser(description="Interact with your research model")
parser = add_common_arguments(parser)
parser = add_ggnn_specific_arguments(parser)

args = parser.parse_args()

if args.override:
    override_dotmap(args.override, args.config_file)

model_name = args.model or args.config_file.model.name

if not model_name:
    raise (
        ArgumentTypeError(
            "Model identifier must either be supplied in config file or command line argument --model"
        )
    )
else:
    args.model = model_name
    args.config_file.model.name = model_name

# Establish run_id and log files
run_id = get_run_id()
args.run_id = run_id
args.log_file = os.path.join(args.log_dir, "{}_log.json".format(run_id))
args.best_model_file = os.path.join(args.log_dir, "{}_model_best.pkl".format(run_id))

try:
    take_action(args)
except:
    typ, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

