import json
import logging
import os
from typing import Callable

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

DEFAULT_MAX_SEQUENCE_LENGTH_FOR_SAMPLING = 50


class EvoAMP:
    def __init__(self, embedding_dim: int, latent_dim: int, hidden_dim: int):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_dim = len(TOKEN_TO_ID)
        seq_start_embedding = one_hot(torch.tensor(TOKEN_TO_ID[START_TOKEN]), self.input_dim)
        self.module = VAE(self.input_dim, embedding_dim, latent_dim, hidden_dim, seq_start_embedding.to(torch.float32))

    def train(self, df: pd.DataFrame, train_kwargs: dict = None, log_callback: Callable = None) -> dict:
        if train_kwargs is None:
            train_kwargs = {}

        batch_size = train_kwargs.get("batch_size", 64)
        val_split = train_kwargs.get("val_split", 0.0)
        lr = train_kwargs.get("lr", 0.001)
        epochs = train_kwargs.get("epochs", 10)
        kl_weight = train_kwargs.get("kl_weight", 1.0)
        sequence_padding = train_kwargs.get("sequence_padding", 0)

        dataset = AMPDataset(df)
        train_set_len = int(len(dataset) * (1 - val_split))
        val_set_len = len(dataset) - train_set_len
        train_set, val_set = torch.utils.data.random_split(dataset, [train_set_len, val_set_len])

        train_loader = AMPDataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = AMPDataLoader(val_set, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

        def _elbo(xs, xs_length, xs_hat, qz, pz, kl_weight=1.0):
            xs_hat = xs_hat.permute(1, 0, 2)  # (batch_size, seq_len, input_dim)

            kl = kl_divergence(qz, pz).sum(dim=-1)
            weighted_kl = kl * kl_weight

            # In order to ignore padding tokens when computing the loss we could:
            #   1. harcode the aminoacids logits for padding tokensn to the same (e.g., 0)
            #   2. set the reconstruction loss to 0 for padding tokens
            # Option 2. would make the loss independent of number of padding tokens and therefore easier to interpret.
            # However, MuE takes logits and assumed there probs sum to 1, so to keep the implementation similar we opt for option 1.
            mask = torch.arange(xs_hat.shape[1]).unsqueeze(0) < torch.tensor(xs_length).unsqueeze(1)
            xs_hat = xs_hat * mask.unsqueeze(-1).float()

            px = Categorical(logits=xs_hat)
            recon_loss = -px.log_prob(xs).sum(dim=-1)

            loss = (recon_loss + weighted_kl).mean()

            return loss

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            train_loss = []

            self.module.train()
            for batch in train_loader:
                seqs, seq_lengths, is_amps = batch
                max_sequence_length = seqs.shape[-1] + sequence_padding

                inference_output, generative_output = self.module(seqs, max_sequence_length)

                optimizer.zero_grad()
                loss = _elbo(
                    seqs,
                    seq_lengths,
                    generative_output["xs"],
                    inference_output["qz"],
                    generative_output["pz"],
                    kl_weight,
                )
                loss.backward()
                optimizer.step()

                train_loss += [loss.item()]

            train_losses += [sum(train_loss) / len(train_loss)]

            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]}")
            _trigger_callback(log_callback, {"train_loss": train_losses[-1]})

            if val_set_len > 0:
                val_loss = []

                self.module.eval()

                for batch in val_loader:
                    seqs, seq_lengths, is_amps = batch
                    max_sequence_length = max(seq_lengths) + sequence_padding

                    inference_output, generative_output = self.module(seqs, max_sequence_length)

                    loss = _elbo(
                        seqs,
                        seq_lengths,
                        generative_output["xs"],
                        inference_output["qz"],
                        generative_output["pz"],
                        kl_weight,
                    )

                    val_loss += [loss.item()]

                val_losses += [sum(val_loss) / len(val_loss)]

                logger.info(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_losses[-1]}")
                _trigger_callback(log_callback, {"val_loss": val_losses[-1]})

        results = {}
        results["train_losses"] = train_losses

        if val_set_len > 0:
            results["val_losses"] = val_losses

        return results

    def sample(
        self, n_samples: int = 1, is_amp: int = 1, reference_sequence: str = None, max_sequence_length: int = None
    ) -> list[list[str]]:
        self.module.eval()

        if reference_sequence is not None:
            seq = prepare_sequence(reference_sequence)
            seq_ids = torch.tensor(sequence_to_ids(seq)).unsqueeze(0)

        if max_sequence_length is None:
            if reference_sequence is None:
                max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH_FOR_SAMPLING
            else:
                max_sequence_length = len(seq)

        sample_ids = []
        for _ in range(n_samples):
            if reference_sequence is None:
                # sample using prior z
                pz = self.module.decoder.pz
                z = pz.sample().unsqueeze(0)

                generative_output = self.module.decoder(z, max_sequence_length)
                sample = generative_output["xs"].permute(1, 0, 2).argmax(dim=-1).squeeze(0).tolist()
            else:
                # sample using posterior z given reference sequence
                _, generative_output = self.module(seq_ids, max_sequence_length)
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
            "embedding_dim": self.embedding_dim,
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


def _trigger_callback(callback: Callable, data: dict):
    if callback is None:
        return

    callback(data)
