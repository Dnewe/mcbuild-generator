
from mcbuild_generator.constants.paths import GENERATED_SCHEM_DIR, IDX_TO_BLOCK_JSON
from mcbuild_generator.validation.create_schematic import create_schematic
from mcbuild_generator.utils.fs_io import read_json


def generate_builds(model, device, n_builds=8):
    idx_to_block = read_json(IDX_TO_BLOCK_JSON)

    for i in range(n_builds):
        out = model.generate((32,32,32), device=device).cpu()
        array = out[0].numpy()
        schem_name = f"generated_{i+1}"
        create_schematic(GENERATED_SCHEM_DIR, schem_name, array, idx_to_block)
