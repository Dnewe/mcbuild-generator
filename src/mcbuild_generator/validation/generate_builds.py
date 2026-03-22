from tqdm import tqdm

from mcbuild_generator.validation.create_schematic import create_schematic


def generate_builds(out_dir, model, idx_to_block, device, n_builds=8):

    for i in tqdm(range(n_builds), desc="generating"):
        out = model.generate((32, 32, 32), device=device).cpu()
        array = out[0].numpy()
        schem_name = f"generated_{i + 1}"
        create_schematic(out_dir, schem_name, array, idx_to_block)
