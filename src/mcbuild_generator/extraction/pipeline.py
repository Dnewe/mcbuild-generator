import os
import pandas as pd

from mcbuild_generator.extraction.extract_filepaths import extract_filepaths
from mcbuild_generator.extraction.extract_builds_data import extract_builds_data
from mcbuild_generator.utils.fs_io import write_json
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.constants.paths import BUILDS_METADATA_CSV, RAW_BUILDS_FP_JSON


def pipeline_extraction(config):
    # extract filepaths
    print("\nExtracting filepaths...")
    raw_builds_fp = extract_filepaths(config["data_dir"], config["max_files"])
    write_json(RAW_BUILDS_FP_JSON, raw_builds_fp)

    # extract builds metadata
    if config["use_cache"] and os.path.isfile(BUILDS_METADATA_CSV):
        print("\nUsing cached builds metadata.")
        pd.read_csv(BUILDS_METADATA_CSV)  # to throw an error if it cannot read
    else:
        print("\nExtracting builds metadata...")
        builds_df = extract_builds_data(raw_builds_fp, config["multiprocessing"])
        builds_df.to_csv(BUILDS_METADATA_CSV)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_extraction(config["extraction"])
