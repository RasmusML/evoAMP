from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from evoamp.distributions import MuE, SequentialCategorical
from evoamp.distributions._mue import Profile
from torch.distributions import Normal


class AutoregressiveRNNBase(nn.Module):
    def __init__(self, rnn: nn.RNNBase):
        super().__init__()
        self.rnn = rnn

    def forward(self, x0: torch.Tensor, h0: torch.Tensor, n_steps: int):
        xs, h = [x0], h0
        for t in range(1, n_steps):
            outs, h = self.rnn(xs[t - 1], h)
            xs += [outs]
        return torch.cat(xs, dim=1 if self.rnn.batch_first else 0), h


class Encoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, latent_dim: int, dropout_prob: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1
        self.dropout_prob = dropout_prob

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, bidirectional=self.bidirectional, batch_first=True)
        self.gru2 = nn.GRU(self.directions * hidden_dim, hidden_dim, bidirectional=self.bidirectional, batch_first=True)

        self.z_log_var = nn.Linear(self.directions * hidden_dim, latent_dim)
        self.z_mean = nn.Linear(self.directions * hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        h0 = torch.zeros(self.directions, x.size(0), self.hidden_dim).to(x.device)
        s, _ = self.gru1(emb, h0)  # (batch_size, seq_len, hidden_dim * directions)

        s = F.relu(s)
        s = F.dropout(s, p=self.dropout_prob)

        _, h = self.gru2(s, h0)  # (directions, batch_size, hidden_dim)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)  # (batch_size, directions * hidden_dim)

        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z_var = torch.exp(z_log_var)

        qz = Normal(z_mean, z_var)  # (batch_size, latent_dim)
        z = qz.rsample()

        return {"z": z, "qz": qz}


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        lstm_dim: int,
        output_dim: int,
        dropout_prob: float,
        observation_model: Literal["categorical", "mue"],
        mue_max_latent_sequence_length: int = None,
        pad_token_id: int = None,
        scoring_matrix_probabilities: torch.Tensor = None,
    ):
        super().__init__()

        self.dropout_prob = dropout_prob
        self.ar_gru = AutoregressiveRNNBase(nn.GRU(latent_dim, latent_dim, batch_first=True))
        self.lstm = nn.LSTM(latent_dim, lstm_dim, batch_first=True)
        self.fc = nn.Linear(lstm_dim, output_dim)
        self.observation_model = observation_model
        self.mue_max_latent_sequence_length = mue_max_latent_sequence_length
        self.pad_token_id = pad_token_id

        self.register_buffer("pz_mean", torch.zeros(latent_dim))
        self.register_buffer("pz_var", torch.ones(latent_dim))

        if self.observation_model == "mue":
            if mue_max_latent_sequence_length is None:
                raise ValueError(
                    "mue_max_latent_sequence_length must be set for MuE observation model, recommended value is `1.1 * max_sequence_length`"
                )

            M = mue_max_latent_sequence_length
            D, B = output_dim, output_dim  # latent vocab size, obs. vocab size

            self.insert_seq_logits = nn.Parameter(torch.zeros((M + 1, D)))

            indel_scalar = 1.0
            self.insert_logits = nn.Parameter(torch.ones((M, 3, 2)) * indel_scalar)
            self.delete_logits = nn.Parameter(torch.ones((M, 3, 2)) * indel_scalar)

            if scoring_matrix_probabilities is None:
                substitution_matrix_prob = None
            else:
                S = scoring_matrix_probabilities.shape[0]
                substitution_matrix_prob = torch.eye(D, B)
                substitution_matrix_prob[:S, :S] = scoring_matrix_probabilities

                def _convert_probability_to_logit(prob_matrix: torch.Tensor, eps=1e-6) -> torch.Tensor:
                    prob_matrix = torch.clamp(prob_matrix, eps, 1 - eps)
                    logit_matrix = torch.log(prob_matrix / (1 - prob_matrix))
                    return logit_matrix

                substitute_logits = _convert_probability_to_logit(substitution_matrix_prob)

            self.register_buffer("substitute_logits", substitute_logits)

            self.mue_state_arrange = Profile(M)

    def forward(self, z: torch.Tensor, batch_sequence_length: int):
        x0 = torch.zeros(z.shape[0], 1, z.shape[-1]).to(z.device)

        sequence_length = (
            self.mue_max_latent_sequence_length if self.observation_model == "mue" else batch_sequence_length
        )
        out, _ = self.ar_gru(x0, z.unsqueeze(0), sequence_length)

        out = F.relu(out)
        out = F.dropout(out, p=self.dropout_prob)

        out, _ = self.lstm(out)

        xs = self.fc(out)

        return {
            "xs": xs,
            "px": self._get_observation_distribution(xs),
            "pz": self._get_prior_latent_distribution(),
        }

    def _get_prior_latent_distribution(self):
        return Normal(self.pz_mean, self.pz_var)

    def _get_observation_distribution(self, xs: torch.Tensor):
        if self.observation_model == "categorical":
            return SequentialCategorical(logits=xs, pad_token_id=self.pad_token_id)
        elif self.observation_model == "mue":
            norm_precursor_seq_logits = xs - xs.logsumexp(dim=-1, keepdim=True)
            norm_insert_seq_logits = self.insert_seq_logits - self.insert_seq_logits.logsumexp(dim=-1, keepdim=True)
            norm_insert_logits = self.insert_logits - self.insert_logits.logsumexp(dim=-1, keepdim=True)
            norm_delete_logits = self.delete_logits - self.delete_logits.logsumexp(dim=-1, keepdim=True)

            if self.substitute_logits is not None:
                norm_substitute_logits = self.substitute_logits - self.substitute_logits.logsumexp(dim=-1, keepdim=True)
            else:
                norm_substitute_logits = None

            return MuE(
                precursor_seq_logits=norm_precursor_seq_logits,
                insert_seq_logits=norm_insert_seq_logits,
                insert_logits=norm_insert_logits,
                delete_logits=norm_delete_logits,
                substitute_logits=norm_substitute_logits,
                state_arrange=self.mue_state_arrange,
                pad_token_id=self.pad_token_id,
            )
        else:
            raise ValueError(f"Invalid observation model: {self.observation_model}")


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_embedding_dim: int,
        encoder_gru_dim: int,
        latent_dim: int,
        decoder_lstm_dim: int,
        dropout_prob: float = 0.3,
        observation_model: Literal["categorical", "mue"] = "categorical",
        mue_max_latent_sequence_length: int = None,
        pad_token_id: int = None,
        scoring_matrix_probabilities: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_embedding_dim, encoder_gru_dim, latent_dim, dropout_prob)
        self.decoder = Decoder(
            latent_dim,
            decoder_lstm_dim,
            input_dim,
            dropout_prob,
            observation_model,
            mue_max_latent_sequence_length,
            pad_token_id,
            scoring_matrix_probabilities,
        )

    def forward(self, x: torch.Tensor, batch_sequence_length: int):
        inference_outputs = self.encoder(x)
        generative_outputs = self.decoder(inference_outputs["z"], batch_sequence_length)
        return inference_outputs, generative_outputs
