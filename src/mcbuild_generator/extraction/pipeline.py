from mcbuild_generator.extraction.extract_filepaths import extract_filepaths
from mcbuild_generator.utils.args import get_config


def pipeline_extraction(config):
    print("\nextracting filepaths...")
    extract_filepaths(config["data_dir"], config["max_files"])


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_extraction(config["extraction"])
