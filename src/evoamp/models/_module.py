import torch
import torch.nn as nn
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
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.directions = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.gru1 = nn.GRU(embedding_dim, hidden_dim, bidirectional=self.bidirectional, batch_first=True)
        self.gru2 = nn.GRU(self.directions * hidden_dim, hidden_dim, bidirectional=self.bidirectional, batch_first=True)

        self.z_log_var = nn.Linear(self.directions * hidden_dim, latent_dim)
        self.z_mean = nn.Linear(self.directions * hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        emb = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        h0 = torch.zeros(self.directions, x.size(0), self.hidden_dim)
        s, _ = self.gru1(emb, h0)  # (batch_size, seq_len, hidden_dim * directions)
        _, h = self.gru2(s, h0)  # (directions, batch_size, hidden_dim)
        h = h.permute(1, 0, 2).reshape(x.size(0), -1)  # (batch_size, directions * hidden_dim)

        z_mean = self.z_mean(h)
        z_log_var = self.z_log_var(h)
        z_var = torch.exp(z_log_var)

        qz = Normal(z_mean, z_var)  # (batch_size, latent_dim)
        z = qz.rsample()

        return {"z": z, "qz": qz}


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, lstm_dim: int, output_dim: int):
        super().__init__()

        self.ar_gru = AutoregressiveRNNBase(nn.GRU(latent_dim, latent_dim, batch_first=True))
        self.lstm = nn.LSTM(latent_dim, lstm_dim, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(lstm_dim, output_dim)
        self.pz = Normal(torch.zeros(latent_dim), torch.ones(latent_dim))

    def forward(self, z: torch.Tensor, sequence_length: int):
        x0 = torch.zeros(z.shape[0], 1, z.shape[-1])
        out, _ = self.ar_gru(x0, z.unsqueeze(0), sequence_length)
        out, _ = self.lstm(out)
        xs = self.fc(out)
        return {"xs": xs, "pz": self.pz}


class VAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        encoder_embedding_dim: int,
        encoder_gru_dim: int,
        latent_dim: int,
        decoder_lstm_dim: int,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, encoder_embedding_dim, encoder_gru_dim, latent_dim)
        self.decoder = Decoder(latent_dim, decoder_lstm_dim, input_dim)

    def forward(self, x: torch.Tensor, max_sequence_length: int):
        inference_outputs = self.encoder(x)
        generative_outputs = self.decoder(inference_outputs["z"], max_sequence_length)
        return inference_outputs, generative_outputs
