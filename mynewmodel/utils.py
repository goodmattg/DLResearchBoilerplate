from dotmap import DotMap
from tensorflow.python.lib.io import file_io

def check_valid_file(value):
    try:
        print("Loading configuration file from: {0}".format(value))
        with file_io.FileIO(value, mode='r') as stream:
            config = DotMap(yaml.load(stream))
    except NotFoundError as _:
        print("Configuration file not found: {0}".format(value))
        return
    except Exception as _:
        print("Unable to load configuration file: {0}".format(value))
        return

def check_valid_training_config_file(value):
    config_file_path = "config/{0}.yaml".format(value)
    return check_valid_file(config_file_path)