import pandas as pd
import os

from mcbuild_generator.processing.filter_builds import filter_builds
from mcbuild_generator.processing.filter_blocks import filter_blocks
from mcbuild_generator.processing.transform_data import transform_data
from mcbuild_generator.processing.count_used_blocks import count_used_blocks
from mcbuild_generator.processing.index_block import index_block
from mcbuild_generator.utils.fs_io import read_json, write_json
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.constants.paths import (
    BUILDS_METADATA_CSV,
    CLEAN_BUILDS_FP_JSON,
    BLOCKS_COUNT_CSV,
    TRAIN_PROCESSED_BUILDS_DIR,
    BLOCK_TO_IDX_JSON,
    IDX_TO_BLOCK_JSON,
    RELEVANT_BLOCKS_JSON,
)


def pipeline_processing(config):
    """
    Data processing pipeline
    """
    builds_metadata_df = pd.read_csv(BUILDS_METADATA_CSV)

    # filter builds
    if config["use_cache"] and os.path.isfile(CLEAN_BUILDS_FP_JSON):
        print("\nUsing cached filtered builds filepaths.")
        filtered_builds_fp = list(read_json(CLEAN_BUILDS_FP_JSON))
    else:
        print("\nFiltering builds...")
        relevant_blocks = list(read_json(RELEVANT_BLOCKS_JSON))
        filtered_builds_fp = filter_builds(
            builds_metadata_df,
            relevant_blocks=relevant_blocks,
            **config["build_filter"],
        )
        write_json(CLEAN_BUILDS_FP_JSON, filtered_builds_fp)  # save for cache

    # count used blocks
    if config["use_cache"] and os.path.isfile(BLOCKS_COUNT_CSV):
        print("\nUsing cached block counts.")
        blocks_df = pd.read_csv(BLOCKS_COUNT_CSV)
    else:
        print("\nCounting used blocks occurence...")
        blocks_df = count_used_blocks(
            filtered_builds_fp, multiproc=config["multiprocessing"]
        )
        blocks_df.to_csv(BLOCKS_COUNT_CSV)  # save for cache

    # filter blocks
    print("\nFiltering block palette...")
    filter_blocks(blocks_df, **config["block_filter"])

    # index blocks
    print("\nIndexing blocks...")
    block_to_idx, idx_to_block = index_block(blocks_df)
    write_json(BLOCK_TO_IDX_JSON, block_to_idx)  # save for later
    write_json(IDX_TO_BLOCK_JSON, idx_to_block)  # save for later

    # transform schem -> tensor
    if config["use_cache"] and os.path.isdir(TRAIN_PROCESSED_BUILDS_DIR):
        schem_count = len(filtered_builds_fp)
        tensor_count = len(
            [
                fn
                for fn in os.listdir(TRAIN_PROCESSED_BUILDS_DIR)
                if fn.split(".")[-1] == "pt"
            ]
        )
        if schem_count == tensor_count:
            print("\nUsing cached transformed builds.")
            return
    print("\nTransforming data...")
    transform_data(
        TRAIN_PROCESSED_BUILDS_DIR,
        filtered_builds_fp,
        block_to_idx,
        multiproc=config["multiprocessing"],
    )


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_processing(config["processing"])
