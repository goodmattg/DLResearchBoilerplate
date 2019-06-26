import argparse
import yaml
import sys
import traceback
import os
import pdb

from train import train
from evaluate import evaluate

from scripts.common_args import CommonArgParser


def take_action(args):
    if args.action == "train":
        train(args.config_file)
    elif args.action == "check":
        # Check verifies that the model compiles by training for one epoch
        override_dotmap(["epochs", "i", 1], args.config_file)
        train(args.config_file)
    elif args.action == "evaluate":
        evaluate(args.config_file)


parser = CommonArgParser()
args = parser.parse_args()

if args.override:
    override_dotmap(args.override, args.config_file)

try:
    take_action(args)
except:
    typ, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

