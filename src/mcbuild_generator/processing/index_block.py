import os
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Literal, Tuple
import pandas as pd

from mcbuild_generator.utils.fs_io import read_json, write_json, write_csv
from mcbuild_generator.processing.schem import Schem
from mcbuild_generator.constants.paths import (
    ALL_BLOCKS_JSON,
    IDX_TO_BLOCK_JSON,
    BLOCK_TO_IDX_JSON,
    BLOCKS_COUNT_CSV
)


def get_all_blocks(all_blocks):
    """
    Returns a list of unique block names in format: minecraft:block[variants...]
    eg: minecraft:air, minecraft:torch[lit=true], minecraft:furnace[facing=east,lit=false]
    """
    block_names = []
    for block, data in all_blocks.items():
        if "variants" in data.keys() and len(data["variants"]) > 1:
            for variant in data["variants"].keys():
                block_names.append(f"{block}[{variant}]")
        else:
            block_names.append(block)
    return block_names


def process_build(fp) -> List[Dict[str, str|int]]:
    '''
    Helper function for get_all_used_blocks() to allow multiproc
    '''
    schem = Schem.load(fp)
    palette = schem.palette
    block_counts = schem.get_block_counts()

    local_used = []
    for b in palette:
        base_block = b.split('[')[0]
        local_used.append({
            "base_block": base_block,
            "block": b,
            "build_count": 1, 
            "block_count": block_counts[b]
        })
    return local_used


def merge_lists(lists: List[List[Dict[str, int|str]]]) -> Dict[str, Dict[str, int|str]]:
    '''
    Helper function for get_all_used_blocks() to allow multiproc
    '''
    merged = {}
    for l in lists:
        for d in l:
            block = d['block']
            if block not in merged:
                merged[block] = {"block": block, "base_block": d["base_block"], "build_count": 0, "block_count": 0}
            merged[block]["build_count"] += d["build_count"]
            merged[block]["block_count"] += d["block_count"]
    return merged


def count_used_blocks(builds_fp: List[str], use_cache=True, multiproc=True): # TODO useless use_cache (always true in this part)
    '''
    Similar to get_all_blocks but only retrieve used blocks in given build files.
    '''
    counts = []
    processes = cpu_count()-2 if multiproc else 1
    with Pool(processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_build, builds_fp), total=len(builds_fp)))

    counts = list(merge_lists(results).values())
    write_csv(BLOCKS_COUNT_CSV, counts)
    return counts


def ignore_rare_variants(
    count_df: pd.DataFrame, 
    threshold=0.1, 
    prop_level='block'
):
    '''
    Map rare variants to the most common variant
    '''
    count_df['variant_prop_build'] = (
        count_df['build_count'] /
        count_df.groupby('base_block')['build_count'].transform('sum')
    )
    count_df['variant_prop_block'] = (
        count_df['block_count']
        / count_df.groupby('base_block')['block_count'].transform('sum')
        #* usedblocks_df.groupby('base_block')['block'].transform('count')
    )

    most_common_map = (
        count_df.loc[count_df.groupby('base_block')[f'variant_prop_{prop_level}'].idxmax()]
        .set_index('base_block')['block']
    )

    count_df['new_block'] = count_df['block']

    mask = count_df[f'variant_prop_{prop_level}'] < threshold
    print(f'rare blocks count (<{threshold}): {mask.sum()}')

    count_df.loc[mask, 'new_block'] = (
        count_df.loc[mask, 'base_block'].map(most_common_map)
    )


def get_indexes(blocks_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    kept_blocks = list(blocks_df['new_block'].unique())
    kept_block_to_idx = {b: i for i,b in enumerate(kept_blocks)}

    idx_to_block = {i: b for i,b in enumerate(kept_blocks)}

    mapping = dict(zip(blocks_df['block'], blocks_df['new_block']))
    block_to_idx = {}
    for b in blocks_df['block']:
        idx = kept_block_to_idx[mapping[b]] 
        block_to_idx[b] = idx
    
    return block_to_idx, idx_to_block
    

def index_block(builds_fp, filter=True, rare_variants_thresh=0.1, proportion_level='block', use_cache=True):
    '''
    Generate json index map for blocks.
    Filter and merge irrelevant block variants into one,
    => multiple block variants -> id ; id -> single block variant
    '''
    if not use_cache or not os.path.isfile(BLOCKS_COUNT_CSV):
        counts = count_used_blocks(builds_fp, use_cache=use_cache)
        write_csv(BLOCKS_COUNT_CSV, counts)
    
    blocks_count = pd.read_csv(BLOCKS_COUNT_CSV)

    if filter:
        ignore_rare_variants(blocks_count, rare_variants_thresh, proportion_level)
    
    block_to_idx, idx_to_block = get_indexes(blocks_count)

    write_json(BLOCK_TO_IDX_JSON, block_to_idx)
    write_json(IDX_TO_BLOCK_JSON, idx_to_block)

    print(f'- used blocks total   : {len(block_to_idx)}')
    print(f'- used blocks filtered: {len(idx_to_block)}')
    print(f'  => removed: {len(block_to_idx) - len(idx_to_block)}')