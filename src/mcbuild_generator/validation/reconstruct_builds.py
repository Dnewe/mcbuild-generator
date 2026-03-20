import os

from mcbuild_generator.training.dataset import get_dataset
from mcbuild_generator.constants.paths import run_name, GENERATED_SCHEM_DIR, BLOCK_TO_IDX_JSON, IDX_TO_BLOCK_JSON
from mcbuild_generator.validation.create_schematic import create_schematic
from mcbuild_generator.utils.fs_io import read_json


def reconstruct_builds(model, device, n_builds=8):
    _, val_dataset = get_dataset()
    idx_to_block = read_json(IDX_TO_BLOCK_JSON)

    for i in range(n_builds):
        x = val_dataset[i].to(device) # type: ignore
        out = model.reconstruct(x, device=device)

        original = x[0].numpy()
        reconstructed = out[0].numpy()

        # original
        schem_name = f"{run_name}_original_{i+1}"
        create_schematic(GENERATED_SCHEM_DIR, schem_name, original, idx_to_block)
        # reconstructed
        schem_name = f"{run_name}_reconstructed_{i+1}"
        create_schematic(GENERATED_SCHEM_DIR, schem_name, reconstructed, idx_to_block)