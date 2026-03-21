import pandas as pd

from mcbuild_generator.processing.filter_builds import filter_builds
from mcbuild_generator.processing.filter_blocks import filter_blocks
from mcbuild_generator.processing.transform_data import transform_data
from mcbuild_generator.processing.count_used_blocks import count_used_blocks
from mcbuild_generator.processing.index_block import index_block
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.constants.paths import BUILDS_METADATA_CSV


def pipeline_processing(config):
    """
    Data processing pipeline
    """
    builds_metadata_df = pd.read_csv(BUILDS_METADATA_CSV)

    print("\nFiltering builds...")
    filtered_builds_fp = filter_builds(
        builds_metadata_df, **config["build_filter"], use_cache=config["use_cache"]
    )

    print("\nCounting used blocks occurence...")
    counts = count_used_blocks(
        filtered_builds_fp,
        use_cache=config["use_cache"],
        multiproc=config["multiprocessing"],
    )

    print("\nFiltering block palette...")
    filter_blocks(counts, **config["block_filter"])

    print("\nIndexing blocks...")
    block_to_idx, idx_to_block = index_block(counts)

    print("\nTransforming data...")
    transform_data(
        filtered_builds_fp,
        block_to_idx,
        use_cache=config["use_cache"],
        multiproc=config["multiprocessing"],
    )


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_processing(config["processing"])
