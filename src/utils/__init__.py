import sys
import math
import json
import yaml
import logging


DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"


logging.basicConfig(
    datefmt=DATE_FORMAT,
    format=LOG_FORMAT,
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("crisis-tapt-hmc")


def load_config(file):
    config = yaml.load(
        open(file, 'r'), 
        Loader=yaml.FullLoader
    )
    return config


def save_config(file, params):
    yaml.dump(
        params,
        open(file, 'w')
    )


def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))