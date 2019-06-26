def get_from_dict(dictionary, key_list):
    """Get value from dictionary with arbitrary depth via descending list of keys."""
    return reduce(operator.getitem, key_list, dictionary)


def set_in_dict(dictionary, key_list, value):
    """Set value from dictionary with arbitrary depth via descending list of keys."""
    get_from_dict(dictionary, key_list[:-1])[key_list[-1]] = value


def cast_to_type(type_abbrev, val):
    """Convert string input value to explicitly denoted type. Types are as follows:
    "f" -> float
    "i" -> integer
    "b" -> boolean
    "s" -> string
    """
    if type_abbrev == "f":
        return float(val)
    elif type_abbrev == "i":
        return int(val)
    elif type_abbrev == "b":
        return val.lower() in ("yes", "true", "t", "1")
    else:
        return val


def override_dotmap(overrides, config):
    """Override DotMap dictionary with explicitly typed values."""
    for i in range(len(overrides) // 3):
        key, type_abbrev, val = overrides[i * 3 : (i + 1) * 3]
        set_in_dict(config, key.split("."), cast_to_type(type_abbrev, val))


def file_exists(prospective_file):
    """Check if the prospective file exists"""
    if not os.path.exists(prospective_file):
        raise argparse.ArgumentTypeError("File: '{0}' does not exist".format(prospective_file))
    return prospective_file


def dir_exists_write_privileges(prospective_dir):
    """Check if the prospective directory exists with write priveliges."""
    if not os.path.isdir(prospective_dir):
        raise argparse.ArgumentTypeError("Directory: '{0}' does not exist".format(prospective_dir))
    elif not os.access(prospective_dir, os.W_OK):
        raise argparse.ArgumentTypeError("Directory: '{0}' is not writable".format(prospective_dir))
    return prospective_dir

   
def dir_exists_read_privileges(prospective_dir):
    """Check if the prospective directory exists with read priveliges."""
    if not os.path.isdir(prospective_dir):
        raise argparse.ArgumentTypeError("Directory: '{0}' does not exist".format(prospective_dir))
    elif not os.access(prospective_dir, os.R_OK):
        raise argparse.ArgumentTypeError("Directory: '{0}' is not readable".format(prospective_dir))
    return prospective_dir
