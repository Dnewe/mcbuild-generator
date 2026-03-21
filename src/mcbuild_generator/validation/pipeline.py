import torch

from mcbuild_generator.training.vae.vae import get_model
from mcbuild_generator.validation.reconstruct_builds import reconstruct_builds
from mcbuild_generator.validation.generate_builds import generate_builds
from mcbuild_generator.utils.args import get_config
from mcbuild_generator.constants.paths import MODEL_FP


def pipeline_validation(config, model_config):
    """
    Validation pipeline
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        embed_dim=model_config["embed_dim"],
        latent_channels=model_config["latent_channels"],
        use_pretrained=True,
        pretrained_fp=MODEL_FP,
        device=device,
    )

    print("\nComputing reconstructed builds...")
    reconstruct_builds(model, device=device)

    print("\nComputing generated builds...")
    generate_builds(model, device=device)


if __name__ == "__main__":
    # get args
    config = get_config()

    # run
    pipeline_validation(config["validation"], config["training"]["model"])  # type: ignore
