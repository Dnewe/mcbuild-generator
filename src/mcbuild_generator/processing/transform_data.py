import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

from mcbuild_generator.processing.schem import Schem
from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.processing.index_block import index_block
from mcbuild_generator.constants.paths import (
    CLEAN_BUILDS_FP_JSON,
    IDX_TO_BLOCK_JSON,
    BLOCK_TO_IDX_JSON,
    PROCESSED_BUILDS_DIR
)


def convert_schem(fp, blocks_to_idx):
    schem = Schem.load(fp)
    array = schem.to_array(blocks_to_idx)
    save_fp = os.path.join(PROCESSED_BUILDS_DIR, f'{schem.id}.pt')
    torch.save(torch.from_numpy(array.astype(np.int16)), save_fp)


def transform_data(filter=True, rare_variants_thresh=0.1, proportion_level='block', use_cache= True, multiproc=True):
    clean_builds_fp = read_json(CLEAN_BUILDS_FP_JSON)

    if not use_cache or not os.path.isfile(BLOCK_TO_IDX_JSON) or not os.path.isfile(IDX_TO_BLOCK_JSON):
        print('indexing blocks...')
        index_block(clean_builds_fp, filter, rare_variants_thresh, proportion_level, use_cache)

    blocks_to_idx = dict(read_json(BLOCK_TO_IDX_JSON))

    print('Transforming schem into tensors')

    convert_schem_partial = partial(convert_schem, blocks_to_idx=blocks_to_idx)
    processes = cpu_count() - 2 if multiproc else 1
    with Pool(processes) as p:
        for _ in tqdm(p.imap_unordered(convert_schem_partial, clean_builds_fp), total=len(clean_builds_fp)):
            pass
        

