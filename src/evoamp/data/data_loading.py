import torch
from evoamp.models._globals import PAD_TOKEN, TOKEN_TO_ID, prepare_sequence, sequence_to_ids
from torch.utils.data import DataLoader, Dataset

SEQUENCE_COLUMN = "sequence"
IS_AMP_COLUMN = "is_amp"

TEMP_SEQUENCE_LENGTH_COLUMN = "__sequence_length"


def _check_data(df):
    if IS_AMP_COLUMN not in df.columns:
        raise ValueError(f"Column '{IS_AMP_COLUMN}' not found in DataFrame")

    if SEQUENCE_COLUMN not in df.columns:
        raise ValueError(f"Column '{SEQUENCE_COLUMN}' not found in DataFrame")


class AMPDataset(Dataset):
    def __init__(self, df, sort=True):
        _check_data(df)

        self.df = df.copy()

        if sort:
            self.df[TEMP_SEQUENCE_LENGTH_COLUMN] = self.df[SEQUENCE_COLUMN].apply(len)
            self.df = self.df.sort_values(TEMP_SEQUENCE_LENGTH_COLUMN).reset_index(drop=True)
            self.df.drop(columns=[TEMP_SEQUENCE_LENGTH_COLUMN], inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = prepare_sequence(row[SEQUENCE_COLUMN])
        seq_ids = torch.tensor(sequence_to_ids(seq))
        is_amp = row[IS_AMP_COLUMN]

        return seq_ids, is_amp


def _amp_collate_fn(batch):
    seqs, is_amps = zip(*batch)

    seq_lengths = [len(seq) for seq in seqs]
    max_length = max(seq_lengths)

    pad_id = TOKEN_TO_ID[PAD_TOKEN]
    padded_seqs = torch.full((len(seqs), max_length), pad_id, dtype=torch.int64)
    for i, seq in enumerate(seqs):
        padded_seqs[i, : seq_lengths[i]] = seq

    is_amps = torch.tensor(is_amps)

    return padded_seqs, is_amps


class AMPDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = _amp_collate_fn
