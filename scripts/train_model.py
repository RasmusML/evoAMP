import logging
import os

import hydra
import matplotlib.pyplot as plt
import pandas as pd
import torch
from evoamp.models import EvoAMP
from omegaconf import OmegaConf

import wandb

logger = logging.getLogger(__name__)


def _plot_loss(history, output_dir):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(history["train_losses"], label="train")
    if "val_losses" in history:
        ax.plot(history["val_losses"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "loss.png"), bbox_inches="tight")


@hydra.main(config_path="cfg", config_name="config", version_base="1.2")
def main(config):
    """Train a VAE model on AMP sequences."""
    logger.info(f"config: \n {OmegaConf.to_yaml(config)}")

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    relative_output_dir = output_dir.split("outputs/")[-1]

    # Set seed
    cfg = config["experiment"]
    torch.manual_seed(cfg.seed)

    # Load data
    cfg_data = cfg["data"]
    df = pd.read_csv(cfg_data["dataset_path"])
    if cfg_data["n_seqs"] is not None:
        df = df.iloc[: cfg_data["n_seqs"]]

    # Setup logging
    cfg_logging = cfg["logging"]
    if cfg_logging["use_wandb"]:
        dict_config = OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            project="evoamp",
            name=relative_output_dir,
            config=dict_config,
        )
        callback = wandb.log
    else:
        callback = None

    # Train model
    cfg_model = cfg["model"]
    cfg_encoder = cfg_model["encoder"]
    cfg_decoder = cfg_model["decoder"]

    if cfg_model["observation_model"] == "mue":
        max_sequence_length = df["sequence"].str.len().max()
        mue_max_latent_sequence_length = int(max_sequence_length * 1.1)
    else:
        mue_max_latent_sequence_length = None

    model = EvoAMP(
        encoder_embedding_dim=cfg_encoder["embedding_dim"],
        encoder_gru_dim=cfg_encoder["gru_dim"],
        latent_dim=cfg_model["latent_dim"],
        decoder_lstm_dim=cfg_decoder["lstm_dim"],
        observation_model=cfg_model["observation_model"],
        mue_max_latent_sequence_length=mue_max_latent_sequence_length,
    )
    history = model.train(df, cfg["train"], log_callback=callback)

    # Save results
    cfg_visualize = cfg["visualize"]
    if cfg_visualize["plot_loss"]:
        _plot_loss(history, output_dir)

    # Save model
    model_dir = os.path.join(output_dir, "pretrained_model")
    model.save(model_dir)

    logger.info(f"Saving results to {relative_output_dir}")


if __name__ == "__main__":
    main()
