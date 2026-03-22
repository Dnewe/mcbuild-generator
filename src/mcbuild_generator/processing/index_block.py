from typing import Dict, Tuple
import pandas as pd


def index_block(blocks_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Generate json index map for blocks.
    Filter and merge irrelevant block variants into one,
    => multiple block variants -> id ; id -> single block variant
    """
    kept_blocks = list(blocks_df["new_block"].unique())
    kept_block_to_idx = {b: i for i, b in enumerate(kept_blocks)}

    idx_to_block = {i: b for i, b in enumerate(kept_blocks)}

    mapping = dict(zip(blocks_df["block"], blocks_df["new_block"]))
    block_to_idx = {}
    for b in blocks_df["block"]:
        idx = kept_block_to_idx[mapping[b]]
        block_to_idx[b] = idx

    return block_to_idx, idx_to_block
