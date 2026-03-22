import torch

from mcbuild_generator.training.vae.vae import get_model
from mcbuild_generator.validation.reconstruct_builds import reconstruct_builds
from mcbuild_generator.validation.generate_builds import generate_builds
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.utils.fs_io import read_json
from mcbuild_generator.constants.paths import (
    MODEL_FP,
    GENERATED_SCHEM_DIR,
    IDX_TO_BLOCK_JSON,
    TEST_PROCESSED_BUILDS_DIR,
)


def pipeline_validation(config, model_config):
    """
    Validation pipeline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_block = read_json(IDX_TO_BLOCK_JSON)

    # load trained model
    model_config["use_pretrained"] = True  # force pretrained model
    model_config["pretrained_fp"] = MODEL_FP
    model = get_model(**model_config, device=device)

    # reconstruct builds
    print("\nComputing reconstructed builds...")
    reconstruct_builds(
        GENERATED_SCHEM_DIR,
        model,
        TEST_PROCESSED_BUILDS_DIR,
        idx_to_block,
        **config["reconstruct"],
        device=device,
    )

    # generate builds
    print("\nComputing generated builds...")
    generate_builds(
        GENERATED_SCHEM_DIR, model, idx_to_block, **config["generate"], device=device
    )


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_validation(config["validation"], config["training"]["model"])  # type: ignore
