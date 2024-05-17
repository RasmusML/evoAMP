import torch
from evoamp.models._globals import prepare_sequence, sequence_to_ids
from torch.utils.data import DataLoader, Dataset


def _check_data(df):
    if "is_amp" not in df.columns:
        raise ValueError("Column 'is_amp' not found in DataFrame")

    if "sequence" not in df.columns:
        raise ValueError("Column 'sequence' not found in DataFrame")


class AMPDataset(Dataset):
    def __init__(self, df):
        self.df = df

        _check_data(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = prepare_sequence(row["sequence"])
        seq_ids = torch.tensor(sequence_to_ids(seq))
        length = seq_ids.shape[0]
        is_amp = row["is_amp"]

        return seq_ids, length, is_amp


def _amp_collate_fn(batch):
    seqs, lengths, is_amps = zip(*batch)

    max_length = max(lengths)

    padded_seqs = torch.zeros(len(seqs), max_length, dtype=torch.int64)
    for i, seq in enumerate(seqs):
        padded_seqs[i, : lengths[i]] = seq

    return padded_seqs, lengths, is_amps


class AMPDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _amp_collate_fn