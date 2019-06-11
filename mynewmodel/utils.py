import os
import sys
import yaml
import pickle as pkl

from tensorflow.python.lib.io import file_io
from tensorflow.python.framework.errors_impl import NotFoundError

# Base Utilities (standard to boilerplate repository)


def load_file(filepath, load_func, **kwargs):
    try:
        print("Loading data file from: {0}".format(filepath))
        return load_func(filepath, **kwargs)
    except NotFoundError as e:
        print("Data file not found: {0}".format(filepath))
        exit(2)
    except Exception as e:
        print("Unable to load data file: {0}".format(filepath))
        exit(3)


def pickle_loader(filepath, encoding=None):
    """Load pickle file"""
    with file_io.FileIO(filepath, mode="rb") as stream:
        if encoding:
            return pkl.load(stream, encoding=encoding)
        else:
            return pkl.load(stream)


def yaml_loader(filepath, use_dotmap=True):
    """Load a yaml file into a dictionary. Optionally wrap with DotMap"""
    with file_io.FileIO(filepath, mode="r") as stream:
        if use_dotmap:
            return DotMap(yaml.load(stream))
        else:
            return yaml.load(stream)
