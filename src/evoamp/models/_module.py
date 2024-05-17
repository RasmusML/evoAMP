import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class AutoregressiveModule(nn.Module):
    def __init__(
        self, rnn_cell_module: type[nn.Module], out_module: type[nn.Module], input_output_dim: int, hidden_dim: int
    ):
        super().__init__()
        self.input_output_dim = input_output_dim

        self.rnn = rnn_cell_module(input_output_dim, hidden_dim)
        self.onn = out_module(hidden_dim, input_output_dim)

    def forward(self, x0: torch.Tensor, h0: torch.Tensor, n_steps: int):
        xs, h = [x0], h0
        for t in range(1, n_steps):
            h = self.rnn(xs[t - 1], h)
            xs += [self.onn(h)]
        return torch.stack(xs), h


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim)

        self.var = nn.Linear(hidden_dim, latent_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        # x: (seq_length, batch_size, input_dim)
        h0 = torch.zeros(1, x.size(1), self.hidden_dim)
        _, h = self.gru(x, h0)
        h = F.relu(h.squeeze(0))  # (batch_size, hidden_dim)

        z_mean = self.mean(h)
        z_log_var = self.var(h)
        z_var = torch.exp(z_log_var)

        pz = Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        qz = Normal(z_mean, z_var)  # (batch_size, latent_dim)
        z = qz.rsample()

        return {"z": z, "qz": qz, "pz": pz}


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_start_embedding: torch.Tensor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_start_embedding = seq_start_embedding

        self.ar_gru = AutoregressiveModule(nn.GRUCell, nn.Linear, output_dim, hidden_dim)
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim)

    def forward(self, z: torch.Tensor, sequence_length: int):
        x0 = self.seq_start_embedding.repeat(z.size(0), 1)
        h0 = F.relu(self.z_to_hidden(z))
        xs, _ = self.ar_gru(x0, h0, sequence_length)
        return {"xs": xs}


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, seq_start_embedding: torch.Tensor):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_start_embedding)

    def forward(self, x: torch.Tensor, max_sequence_length: int):
        inference_outputs = self.encoder(x)
        generative_outputs = self.decoder(inference_outputs["z"], max_sequence_length)
        return inference_outputs, generative_outputs
