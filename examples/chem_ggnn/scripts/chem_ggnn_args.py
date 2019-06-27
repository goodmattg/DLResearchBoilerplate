from argparse import ArgumentParser

from utils.argparse import *


def add_ggnn_specific_arguments(parser: ArgumentParser) -> ArgumentParser:

    optional = parser.add_argument_group("GGNN example specific arguments")

    optional.add_argument(
        "--restrict-data",
        type=int,
        help="Stop index to truncate to after loading data JSON",
    )

    return parser
