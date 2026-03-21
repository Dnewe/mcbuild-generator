import os
import numpy as np
from typing import List, Dict

from mcbuild_generator.utils.fs_io import write_json
from mcbuild_generator.constants.paths import RAW_BUILDS_FP_JSON


def get_row(dir, fn):
    _id = fn.split(".")[0]
    _fp = os.path.join(dir, fn)
    return {"id": _id, "filepath": _fp}


def extract_filepaths(
    datadir: str, max_files: int = -1, extensions: List[str] = ["schem"]
) -> List[Dict[str, str]]:
    """
    Create CSV file containing metadata of build files.

    Args:
        datadir (str): directory where build files are located
        csvfp (str): resulting CSV file path
        max_files (int): Maximum number of build files, -1: no limit
    """
    filenames = os.listdir(datadir)

    # filter extensions
    filenames = [fn for fn in filenames if fn.split(".")[-1] in extensions]

    # shuffle
    np.random.shuffle(filenames)

    # slice
    if max_files > 0 and max_files < len(filenames):
        filenames = filenames[:max_files]

    builds_fp = [get_row(datadir, fn) for fn in filenames]
    write_json(RAW_BUILDS_FP_JSON, builds_fp)

    return builds_fp
