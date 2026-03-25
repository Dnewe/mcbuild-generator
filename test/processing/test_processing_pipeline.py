import pandas as pd
import os

from mcbuild_generator.processing.filter_builds import filter_builds
from mcbuild_generator.processing.filter_blocks import filter_blocks
from mcbuild_generator.processing.transform_data import transform_data
from mcbuild_generator.processing.count_used_blocks import count_used_blocks
from mcbuild_generator.processing.index_block import index_block
from mcbuild_generator.utils.fs_io import write_json, read_json


def test_pipeline_processing(
    config,
    metadata_csv,
    idx_to_block_json,
    block_to_idx_json,
    processed_dir,
    relevant_blocks_json,
):
    """
    Data processing pipeline
    """
    builds_metadata_df = pd.read_csv(metadata_csv)
    relevant_blocks = list(read_json(relevant_blocks_json))

    # filter builds
    print("\nFiltering builds...")

    filtered_builds_fp = filter_builds(
        builds_metadata_df, relevant_blocks, **config["build_filter"]
    )

    # test

    # count used blocks
    print("\nCounting used blocks occurence...")
    blocks_df = count_used_blocks(
        filtered_builds_fp, multiproc=config["multiprocessing"]
    )

    # test unicity of block names
    assert len(blocks_df) == len(blocks_df["block"].unique())

    # filter blocks
    print("\nFiltering block palette...")
    filter_blocks(blocks_df, **config["block_filter"])

    # test less new blocks
    assert len(blocks_df["new_block"].unique()) <= len(blocks_df["block"].unique())

    # index blocks
    print("\nIndexing blocks...")
    block_to_idx, idx_to_block = index_block(blocks_df)
    write_json(block_to_idx_json, block_to_idx)  # save for later
    write_json(idx_to_block_json, idx_to_block)  # save for later

    # test lengths
    assert len(block_to_idx) == len(blocks_df)
    assert len(idx_to_block) == len(blocks_df["new_block"].unique())

    # transform schem -> tensor
    print("\nTransforming data...")
    transform_data(
        processed_dir,
        filtered_builds_fp,
        block_to_idx,
        multiproc=config["multiprocessing"],
    )

    # test files count
    schem_count = len(filtered_builds_fp)
    tensor_count = len(
        [fn for fn in os.listdir(processed_dir) if fn.split(".")[-1] == "pt"]
    )
    assert schem_count == tensor_count
