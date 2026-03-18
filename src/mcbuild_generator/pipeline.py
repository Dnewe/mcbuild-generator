from mcbuild_generator.extraction.pipeline import pipeline_extraction
from mcbuild_generator.processing.pipeline import pipeline_processing
from mcbuild_generator.training.pipeline import pipeline_training
from mcbuild_generator.utils.args import get_config


def pipeline_all(config):
    """
    Data processing pipeline
    """
    print("\n\nEXTRACTION PIPELINE")
    pipeline_extraction(config["extraction"])

    print("\n\nPROCESSING PIPELINE")
    pipeline_processing(config["processing"])

    print("\n\nTRAINING PIPELINE")
    pipeline_training(config["training"])


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_all(config)
