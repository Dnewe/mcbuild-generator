from tqdm import tqdm

from mcbuild_generator.training.dataset import get_dataset
from mcbuild_generator.validation.create_schematic import create_schematic


def reconstruct_builds(out_dir, model, data_dir, idx_to_block, device, n_builds=8):
    _, val_dataset = get_dataset(data_dir)

    for i in tqdm(range(n_builds)):
        x = val_dataset[i].to(device)  # type: ignore
        out = model.reconstruct(x, device=device)

        original = x[0].numpy()
        reconstructed = out[0].numpy()

        # original
        schem_name = f"original_{i + 1}"
        create_schematic(out_dir, schem_name, original, idx_to_block)
        # reconstructed
        schem_name = f"reconstructed_{i + 1}"
        create_schematic(out_dir, schem_name, reconstructed, idx_to_block)
