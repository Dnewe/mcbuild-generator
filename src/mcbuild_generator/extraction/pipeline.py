from mcbuild_generator.extraction.extract_filepaths import extract_filepaths
from mcbuild_generator.extraction.extract_builds_data import extract_builds_data
from mcbuild_generator.utils.args import get_config


def pipeline_extraction(config):
    print("\nExtracting filepaths...")
    raw_builds_fp = extract_filepaths(config["data_dir"], config["max_files"])

    extract_builds_data(raw_builds_fp, config["multiprocessing"], config["use_cache"])


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_extraction(config["extraction"])
