PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


def _get_mappings():
    # Important: Don't change this ordering! The order of the tokens must match the order of the amino acids in the substitution matrices.
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # Important: The amino acids *must* be the first tokens.
    # We use this assumption when creating the "extended" substitution matrix in the model.
    tokens = list(amino_acids) + [PAD_TOKEN, START_TOKEN, END_TOKEN]

    id_to_token = {}
    token_to_id = {}

    for i, token in enumerate(tokens):
        id_to_token[i] = token
        token_to_id[token] = i

    return id_to_token, token_to_id


ID_TO_TOKEN, TOKEN_TO_ID = _get_mappings()


def prepare_sequence(sequence: str) -> list[str]:
    return [START_TOKEN] + list(sequence) + [END_TOKEN]


def sequence_to_ids(sequence: list[str]) -> list[int]:
    return [TOKEN_TO_ID[token] for token in sequence]


def ids_to_sequence(ids: list[int]) -> list[str]:
    return [ID_TO_TOKEN[i] for i in ids]
