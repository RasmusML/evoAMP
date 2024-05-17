def _get_mappings(start_token, end_token):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    tokens = [start_token] + list(amino_acids) + [end_token]

    id_to_token = {}
    token_to_id = {}

    for i, token in enumerate(tokens):
        id_to_token[i] = token
        token_to_id[token] = i

    return id_to_token, token_to_id


START_TOKEN = "<start>"
END_TOKEN = "<end>"

ID_TO_TOKEN, TOKEN_TO_ID = _get_mappings(START_TOKEN, END_TOKEN)


def prepare_sequence(sequence: str) -> list[str]:
    return [START_TOKEN] + list(sequence) + [END_TOKEN]


def sequence_to_ids(sequence: list[str]) -> list[int]:
    return [TOKEN_TO_ID[token] for token in sequence]


def ids_to_sequence(ids: list[int]) -> list[str]:
    return [ID_TO_TOKEN[i] for i in ids]
