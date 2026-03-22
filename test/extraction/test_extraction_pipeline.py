import pandas as pd

from mcbuild_generator.extraction.extract_filepaths import extract_filepaths
from mcbuild_generator.extraction.extract_builds_data import extract_builds_data


def test_pipeline_extraction(config, metadata_csv, max_files):
    # extract filepaths
    print("\nExtracting filepaths...")
    raw_builds_fp = extract_filepaths(config["data_dir"], config["max_files"])

    # test length
    assert len(raw_builds_fp) == max_files

    # extract filepaths
    print("\nExtracting builds metadata...")
    builds_df = extract_builds_data(raw_builds_fp, config["multiprocessing"])
    builds_df.to_csv(metadata_csv)

    # test length
    builds_df = pd.read_csv(metadata_csv)
    assert len(builds_df) == max_files
