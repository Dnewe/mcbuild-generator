from mcbuild_generator.extraction.extract_metadata import extract_metadata
from mcbuild_generator.utils.args import get_config


def pipeline(config):
    extract_metadata(config['data_dir'], config['max_files'])


if __name__=='__main__':
    # get args
    config = get_config()

    # run
    pipeline(config)