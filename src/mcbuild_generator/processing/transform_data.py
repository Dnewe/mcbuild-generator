import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List

from mcbuild_generator.processing.schem import Schem
from mcbuild_generator.utils.fs_io import create_dir, del_dir
from mcbuild_generator.constants.paths import PROCESSED_BUILDS_DIR


""" TODO create patches of too big builds
def pad_to_multiple(x, patch_size): 
    H, L, W= x.shape[-3], x.shape[-2], x.shape[-1]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_l = (patch_size - L % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_l, 0, pad_h))
"""


def convert_schem(fp, block_to_idx):
    schem = Schem.load(fp)
    array = schem.to_array(block_to_idx)
    tensor = torch.from_numpy(array.astype(np.int16)).unsqueeze(0)  # shape (1, h, l, w)
    save_fp = os.path.join(PROCESSED_BUILDS_DIR, f"{schem.id}.pt")
    torch.save(tensor, save_fp)


def convert_schems(filepaths, block_to_idx, multiproc):
    convert_schem_partial = partial(convert_schem, block_to_idx=block_to_idx)
    processes = cpu_count() - 2 if multiproc else 1
    with Pool(processes) as p:
        for _ in tqdm(
            p.imap_unordered(convert_schem_partial, filepaths), total=len(filepaths)
        ):
            pass


def transform_data(
    builds_fp: List[str],
    block_to_idx,
    use_cache=True,
    multiproc=True,
):
    if os.path.isdir(PROCESSED_BUILDS_DIR):
        schem_count = len(builds_fp)
        tensor_count = len(
            [fn for fn in os.listdir(PROCESSED_BUILDS_DIR) if fn.split(".")[-1] == "pt"]
        )
        if use_cache and schem_count == tensor_count:
            return

    del_dir(PROCESSED_BUILDS_DIR)
    create_dir(PROCESSED_BUILDS_DIR)
    print("\nTransforming schem into tensors...")
    convert_schems(builds_fp, block_to_idx, multiproc)
