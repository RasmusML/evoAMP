import os

import hydra
import pandas as pd
import torch
from evoamp.models import EvoAMP
from omegaconf import OmegaConf


@hydra.main(config_path="cfg", config_name="default_config.yaml", version_base="1.2")
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")

    cfg = config["experiment"]
    torch.manual_seed(cfg.seed)

    cfg_data = cfg["data"]
    df = pd.read_csv(cfg_data["dataset_path"])
    if cfg_data["n_samples"]:
        df = df.iloc[: cfg_data["n_samples"]]

    model = EvoAMP(**cfg["model"])
    model.train(df, cfg["train"])

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    model_dir = os.path.join(output_dir, "pretrained_model")

    model.save(model_dir)


if __name__ == "__main__":
    main()
