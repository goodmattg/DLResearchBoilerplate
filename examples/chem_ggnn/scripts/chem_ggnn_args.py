import argparse

from utils.argparse import *


def add_example_specific_arguments(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    optional = parser.add_argument_group("GGNN example specific arguments")
    optional.add_argument(
        "--restrict-data",
        type=int,
        help="Stop index to truncate to after loading data JSON"
    )
    return parser
