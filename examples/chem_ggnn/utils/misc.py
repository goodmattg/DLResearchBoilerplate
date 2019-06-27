import time
import os


def get_run_id():
    date = time.strftime("%Y-%m-%d-%H-%M-%S")
    pid = str(os.getpid())
    return "{0}_{1}".format(date, pid)
