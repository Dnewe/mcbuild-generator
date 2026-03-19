import argparse
import os
import sys
import yaml
from collections import defaultdict


def parse_config():
    """
    Parse --config argument
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="yaml config file path to run pipeline.",
    )
    return parser.parse_args()


def check_config(args):
    """
    Check if argument are valid
    """
    if not os.path.isfile(args.config) or not args.config.split(".")[-1] == "yaml":
        print(f"Error: Config '{args.config}' does not exist or is not .yaml.")
        sys.exit(1)


def get_config():
    """
    Parse argument --config (yaml_fp) to return config dict
    """
    try:
        args = parse_config()
        check_config(args)
        config = yaml.load(open(args.config, "r"), Loader=yaml.SafeLoader)
        return config
    except Exception as e:
        print(f"Error while loading config: {e}")
        return defaultdict(str)
