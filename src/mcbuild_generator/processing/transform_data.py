import os

from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.processing.index_block import index_block
from mcbuild_generator.constants.paths import (
    CLEAN_BUILDS_FP_JSON,
    IDX_TO_BLOCK_JSON,
    BLOCK_TO_IDX_JSON
)


def transform_data(filter=True, rare_variants_thresh=0.1, proportion_level='block', use_cache= True):
    clean_builds_fp = read_json(CLEAN_BUILDS_FP_JSON)

    if not use_cache or not os.path.isfile(BLOCK_TO_IDX_JSON) or not os.path.isfile(IDX_TO_BLOCK_JSON):
        print('indexing blocks...')
        index_block(clean_builds_fp, filter, rare_variants_thresh, proportion_level, use_cache)


