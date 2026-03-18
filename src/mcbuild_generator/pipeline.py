from mcbuild_generator.extraction.pipeline import pipeline_extraction
from mcbuild_generator.processing.pipeline import pipeline_processing
from mcbuild_generator.training.pipeline import pipeline_training
from mcbuild_generator.utils.args import get_config


def pipeline_all(config):
    """
    Data processing pipeline
    """
    print("\n\nExtracting data...")
    pipeline_extraction(config["extraction"])

    print("\n\nProcessing data...")
    pipeline_processing(config["processing"])

    print("\n\nTrainig...")
    pipeline_training(config["training"])


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_all(config)
