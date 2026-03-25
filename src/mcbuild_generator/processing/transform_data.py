import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List

from mcbuild_generator.processing.schem import Schem
from mcbuild_generator.utils.fs_io import create_dir, del_dir


""" TODO create patches of too big builds
def pad_to_multiple(x, patch_size): 
    H, L, W= x.shape[-3], x.shape[-2], x.shape[-1]
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_l = (patch_size - L % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_l, 0, pad_h))
"""


def convert_schem(fp, out_dir, block_to_idx):
    schem = Schem.load(fp)
    array = schem.to_array(block_to_idx)
    tensor = torch.from_numpy(array.astype(np.int16)).unsqueeze(0)  # shape (1, h, l, w)
    save_fp = os.path.join(out_dir, f"{schem.id}.pt")
    torch.save(tensor, save_fp)


def transform_data(out_dir, builds_fp: List[str], block_to_idx, multiproc=True):
    """
    Transforms schem files in builds_fp into tensors.
    """
    del_dir(out_dir)
    create_dir(out_dir)

    convert_schem_partial = partial(
        convert_schem, out_dir=out_dir, block_to_idx=block_to_idx
    )
    processes = cpu_count() - 2 if multiproc else 1
    with Pool(processes) as p:
        for _ in tqdm(
            p.imap_unordered(convert_schem_partial, builds_fp),
            total=len(builds_fp),
            desc="transforming",
        ):
            pass  # unload tqdm iterator
