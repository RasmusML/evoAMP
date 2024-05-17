import json
import logging
import os

import pandas as pd
import torch
from evoamp.data.data_loading import AMPDataLoader, AMPDataset
from evoamp.models._globals import (
    END_TOKEN,
    START_TOKEN,
    TOKEN_TO_ID,
    ids_to_sequence,
    prepare_sequence,
    sequence_to_ids,
)

# from torch.nn.utils.rnn import pack_padded_sequence, unpack_padded_sequence
from evoamp.models._module import VAE
from torch.distributions import Categorical, kl_divergence
from torch.nn.functional import one_hot

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
MODEL_NAME = "model.pt"


class EvoAMP:
    def __init__(self, latent_dim: int, hidden_dim: int):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_dim = len(TOKEN_TO_ID)
        seq_start_embedding = one_hot(torch.tensor(TOKEN_TO_ID[START_TOKEN]), self.input_dim)
        self.module = VAE(self.input_dim, latent_dim, hidden_dim, seq_start_embedding.to(torch.float32))

    def train(self, df: pd.DataFrame, train_kwargs: dict = None):
        if train_kwargs is None:
            train_kwargs = {}

        batch_size = train_kwargs.get("batch_size", 64)
        val_percent = train_kwargs.get("val_percent", 0.0)
        lr = train_kwargs.get("lr", 0.01)
        epochs = train_kwargs.get("epochs", 10)
        beta = train_kwargs.get("beta", 1.0)
        sequence_padding = train_kwargs.get("sequence_padding", 0)

        dataset = AMPDataset(df)
        train_set_len = int(len(dataset) * (1 - val_percent))
        val_set_len = len(dataset) - train_set_len
        train_set, val_set = torch.utils.data.random_split(dataset, [train_set_len, val_set_len])

        train_loader = AMPDataLoader(train_set, batch_size=batch_size, shuffle=True)
        # val_loader = AMPDataLoader(val_set, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

        def elbo(xs, xs_length, xs_hat, qz, pz, beta):
            xs_hat = xs_hat.permute(1, 0, 2)  # (batch_size, seq_len, input_dim)

            kl = kl_divergence(qz, pz).sum(dim=-1)
            weighted_kl = kl * beta

            for i, length in enumerate(xs_length):
                xs_hat[i, length:] = 0

            px = Categorical(logits=xs_hat)

            recon_loss = -px.log_prob(xs).sum(dim=-1)
            loss = (recon_loss + weighted_kl).mean()

            return loss

        for epoch in range(epochs):
            train_loss = []

            self.module.train()
            for batch in train_loader:
                seqs, seq_lengths, is_amps = batch
                seqs_one_hot = one_hot(seqs, self.input_dim).permute(1, 0, 2).to(torch.float32)
                max_sequence_length = max(seq_lengths) + sequence_padding

                inference_output, generative_output = self.module(seqs_one_hot, max_sequence_length)

                optimizer.zero_grad()
                loss = elbo(
                    seqs,
                    seq_lengths,
                    generative_output["xs"],
                    inference_output["qz"],
                    inference_output["pz"],
                    beta,
                )
                loss.backward()
                optimizer.step()

                train_loss += [loss.item()]

            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {sum(train_loss) / len(train_loss)}")

    def sample(
        self, reference_sequence: str, is_amp: int, n_samples: int = 1, sequence_padding: int = 0
    ) -> list[list[str]]:
        self.module.eval()

        seq = prepare_sequence(reference_sequence)
        seq_ids = torch.tensor(sequence_to_ids(seq))
        seq_one_hot = one_hot(seq_ids, self.input_dim).unsqueeze(0).permute(1, 0, 2).to(torch.float32)
        max_sequence_length = len(seq_ids) + sequence_padding

        sample_ids = []
        for _ in range(n_samples):
            inference_output, generative_output = self.module(seq_one_hot, max_sequence_length)
            sample = generative_output["xs"].permute(1, 0, 2).argmax(dim=-1).squeeze(0).tolist()

            try:
                end_index = sample.index(TOKEN_TO_ID[END_TOKEN]) + 1
            except ValueError:
                end_index = len(sample)

            sample = sample[:end_index]
            sample_ids += [sample]

        samples = [ids_to_sequence(sample) for sample in sample_ids]

        return samples

    def save(self, path_dir: str):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        cfg_path = os.path.join(path_dir, CONFIG_NAME)
        model_path = os.path.join(path_dir, MODEL_NAME)

        if os.path.exists(cfg_path):
            raise FileExistsError(f"Configuration file already exists at {cfg_path}")

        if os.path.exists(model_path):
            raise FileExistsError(f"Model file already exists at {model_path}")

        cfg = {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }

        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        torch.save(self.module.state_dict(), model_path)

    @staticmethod
    def load(path_dir: str) -> "EvoAMP":
        cfg_path = os.path.join(path_dir, CONFIG_NAME)
        model_path = os.path.join(path_dir, MODEL_NAME)

        with open(cfg_path) as f:
            cfg = json.load(f)

        model = EvoAMP(**cfg)
        model.module.load_state_dict(torch.load(model_path))
        return model
