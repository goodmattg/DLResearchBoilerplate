from argparse import ArgumentParser

from utils.file import load_training_config_file
from utils.argparse import *

AVAILABLE_ACTIONS = ["train", "check", "test", "evaluate"]


def add_common_arguments(parser: ArgumentParser) -> ArgumentParser:

    core = parser.add_argument_group("core arguments")

    core.add_argument(
        "action",
        choices=AVAILABLE_ACTIONS,
        help="Take an action with your model (e.g. 'train')",
    )

    core.add_argument(
        "--model",
        type=str,
        help="Identifier of model to work with. Overrides config defined model",
    )

    core.add_argument(
        "--config-file",
        type=load_training_config_file,
        default="config/default.yaml",
        help="Configuration file absolute path",
    )

    core.add_argument(
        "--log-dir",
        type=dir_exists_write_privileges,
        default=".",
        help="Log file storage directory path",
    )

    core.add_argument(
        "--data-dir",
        type=dir_exists_read_privileges,
        default="data",
        help="Data file storage directory path",
    )

    core.add_argument(
        "--restore-weights",
        type=file_exists,
        help="Restore model with pre-trained weights",
    )

    core.add_argument("--freeze-graph-model", action="store_true")

    core.add_argument(
        "--override",
        "-o",
        nargs="*",
        help="Override key value pairs in training configuration file for ad-hoc testing",
    )

    return parser
