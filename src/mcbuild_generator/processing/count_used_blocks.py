from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Dict
import pandas as pd

from mcbuild_generator.processing.schem import Schem


def get_all_blocks(all_blocks: Dict):
    """
    Returns a list of unique block names in format: minecraft:block[variants...]
    eg: minecraft:air, minecraft:torch[lit=true], minecraft:furnace[facing=east,lit=false]
    """
    block_names = []
    for block, data in all_blocks.items():
        if "variants" in data.keys() and len(data["variants"]) > 1:
            for variant in data["variants"].keys():
                block_names.append(f"minecraft:{block}[{variant}]")
        else:
            block_names.append(f"minecraft:{block}")
    return block_names


def process_build(fp) -> List[Dict[str, str | int]]:
    """
    Helper function for get_all_used_blocks() to allow multiproc
    """
    schem = Schem.load(fp)
    palette = schem.palette
    block_counts = schem.get_block_counts()

    local_used = []
    for b in palette:
        base_block = b.split("[")[0]
        local_used.append(
            {
                "base_block": base_block,
                "block": b,
                "build_count": 1,
                "block_count": block_counts[b],
            }
        )
    return local_used


def merge_lists(
    lists: List[List[Dict[str, int | str]]],
) -> Dict[str, Dict[str, int | str]]:
    """
    Helper function for get_all_used_blocks() to allow multiproc
    """
    merged = {}
    for list in lists:
        for d in list:
            block = d["block"]
            if block not in merged:
                merged[block] = {
                    "block": block,
                    "base_block": d["base_block"],
                    "build_count": 0,
                    "block_count": 0,
                }
            merged[block]["build_count"] += d["build_count"]
            merged[block]["block_count"] += d["block_count"]
    return merged


def count_used_blocks(builds_fp: List[str], multiproc=True) -> pd.DataFrame:
    """
    For each filtered build, count occurence of blocks in palette and total use in build.
    """
    counts = []
    processes = cpu_count() - 2 if multiproc else 1

    with Pool(processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_build, builds_fp),
                total=len(builds_fp),
                desc="counting",
            )
        )
    counts = list(merge_lists(results).values())

    return pd.DataFrame(counts)
