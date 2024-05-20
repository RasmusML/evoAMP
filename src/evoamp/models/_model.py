import json
import logging
import os
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import torch
from evoamp.data._data_loading import AMPDataLoader, AMPDataset
from evoamp.models._globals import (
    END_TOKEN,
    PAD_TOKEN,
    TOKEN_TO_ID,
    ids_to_sequence,
    prepare_sequence,
    sequence_to_ids,
)

# from torch.nn.utils.rnn import pack_padded_sequence, unpack_padded_sequence
from evoamp.models._module import VAE
from evoamp.scoring_matrices import SCORING_MATRICES
from torch.distributions import kl_divergence

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
MODEL_NAME = "model.pt"

DEFAULT_MAX_SEQUENCE_LENGTH_FOR_SAMPLING = 35


class EvoAMP:
    def __init__(
        self,
        encoder_embedding_dim: int,
        encoder_gru_dim: int,
        latent_dim: int,
        decoder_lstm_dim: int,
        observation_model: Literal["categorical", "mue"] = "categorical",
        mue_max_latent_sequence_length: int = None,
        scoring_matrix: Literal["PAM30", "BLOSUM62"] = None,
    ):
        self.configs = _extract_params(locals())
        self.input_dim = len(TOKEN_TO_ID)

        self.module = VAE(
            self.input_dim,
            encoder_embedding_dim,
            encoder_gru_dim,
            latent_dim,
            decoder_lstm_dim,
            observation_model=observation_model,
            mue_max_latent_sequence_length=mue_max_latent_sequence_length,
            pad_token_id=TOKEN_TO_ID[PAD_TOKEN],
            scoring_matrix_probabilities=_prepare_scoring_matrix(scoring_matrix),
        )

    def train(
        self,
        df: pd.DataFrame,
        train_kwargs: dict[str, Any] = None,
        log_callback: Callable[[dict[str, Any]], None] = None,
    ) -> dict:
        if train_kwargs is None:
            train_kwargs = {}

        epochs = train_kwargs.get("epochs", 10)
        batch_size = train_kwargs.get("batch_size", 64)
        val_split = train_kwargs.get("val_split", 0.0)
        lr = train_kwargs.get("lr", 0.001)
        kl_weight = train_kwargs.get("kl_weight", 1.0)
        shuffle_train_set = train_kwargs.get("shuffle_train_set", False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.module.to(device)

        logger.info(f"Training on {device}")

        dataset = AMPDataset(df)
        train_set_len = int(len(dataset) * (1 - val_split))
        val_set_len = len(dataset) - train_set_len
        train_set, val_set = torch.utils.data.random_split(dataset, [train_set_len, val_set_len])

        train_loader = AMPDataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train_set)
        val_loader = AMPDataLoader(val_set, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

        def _elbo(xs, px, qz, pz, kl_weight=1.0):
            kl = kl_divergence(qz, pz).sum(dim=-1)  # (batch_size)
            weighted_kl = kl * kl_weight

            recon_loss = -px.log_prob(xs)  # (batch_size)
            loss = (recon_loss + weighted_kl).mean()

            return loss

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            epoch_results = {}

            train_loss = []

            self.module.train()
            for batch in train_loader:
                seqs, is_amps = batch

                seqs = seqs.to(device)
                is_amps = is_amps.to(device)

                max_sequence_length = seqs.shape[-1]
                inference_output, generative_output = self.module(seqs, max_sequence_length)

                optimizer.zero_grad()
                loss = _elbo(
                    seqs,
                    generative_output["px"],
                    inference_output["qz"],
                    generative_output["pz"],
                    kl_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.module.parameters(), 1.0)
                optimizer.step()

                train_loss += [loss.item()]

            train_losses += [sum(train_loss) / len(train_loss)]
            epoch_results["train_loss"] = train_losses[-1]

            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.2f}")

            if val_set_len > 0:
                val_loss = []

                self.module.eval()

                for batch in val_loader:
                    seqs, is_amps = batch

                    seqs = seqs.to(device)
                    is_amps = is_amps.to(device)

                    max_sequence_length = seqs.shape[-1]
                    inference_output, generative_output = self.module(seqs, max_sequence_length)

                    loss = _elbo(
                        seqs,
                        generative_output["px"],
                        inference_output["qz"],
                        generative_output["pz"],
                        kl_weight,
                    )

                    val_loss += [loss.item()]

                val_losses += [sum(val_loss) / len(val_loss)]
                epoch_results["val_loss"] = val_losses[-1]

                logger.info(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_losses[-1]:.2f}")

            if log_callback is not None:
                log_callback(epoch_results)

        results = {}
        results["train_losses"] = train_losses

        if val_set_len > 0:
            results["val_losses"] = val_losses

        self.module.cpu()

        return results

    @torch.inference_mode()
    def sample(
        self, n_samples: int = 1, is_amp: int = 1, reference_sequence: str = None, max_sequence_length: int = None
    ) -> list[list[str]]:
        self.module.eval()

        if reference_sequence is None:
            # sample using prior z
            pz = self.module.decoder._get_prior_latent_distribution().expand([1, -1])
        else:
            # sample using posterior z given reference sequence
            seq = prepare_sequence(reference_sequence)
            seq_ids = torch.tensor(sequence_to_ids(seq)).unsqueeze(0)

            inference_output = self.module.encoder(seq_ids)
            pz = inference_output["qz"]

        if max_sequence_length is None:
            max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH_FOR_SAMPLING if reference_sequence is None else len(seq)

        sample_ids = []
        for _ in range(n_samples):
            z = pz.sample()

            generative_output = self.module.decoder(z, max_sequence_length)
            sample = generative_output["xs"].argmax(dim=-1).squeeze(0).tolist()

            try:
                end_index = sample.index(TOKEN_TO_ID[END_TOKEN])
                sample = sample[:end_index]
            except ValueError:
                pass

            sample_ids += [sample]

        samples = [ids_to_sequence(sample) for sample in sample_ids]

        return samples

    @torch.inference_mode()
    def get_latent_representation(self, sequence: str, is_amp: int) -> np.ndarray:
        seq = prepare_sequence(sequence)
        seq_ids = torch.tensor(sequence_to_ids(seq)).unsqueeze(0)

        inference_output = self.module.encoder(seq_ids)
        z = inference_output["qz"].mean.squeeze(0)

        return z.numpy()

    def save(self, path_dir: str):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

        cfg_path = os.path.join(path_dir, CONFIG_NAME)
        model_path = os.path.join(path_dir, MODEL_NAME)

        if os.path.exists(cfg_path):
            raise FileExistsError(f"Configuration file already exists at {cfg_path}")

        if os.path.exists(model_path):
            raise FileExistsError(f"Model file already exists at {model_path}")

        with open(cfg_path, "w") as f:
            json.dump(self.configs, f)

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


def _extract_params(params: dict) -> dict:
    params = params.copy()
    del params["self"]
    return params


def _prepare_scoring_matrix(scoring_matrix: Literal["PAM30", "BLOSUM62"]) -> torch.Tensor:
    if scoring_matrix is None:
        return None

    try:
        scoring_matrix_prob = torch.tensor(SCORING_MATRICES[scoring_matrix])
    except KeyError as e:
        raise ValueError(f"Invalid scoring matrix: {scoring_matrix}") from e

    return scoring_matrix_prob
