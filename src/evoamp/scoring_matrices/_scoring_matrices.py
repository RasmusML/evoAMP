from typing import NamedTuple

import numpy as np
from evoamp.scoring_matrices._blosum import BLOSUM62_SCORING_MATRIX, _convert_blosum_to_probability
from evoamp.scoring_matrices._pam import PAM30_SCORING_MATRIX, _convert_pam_to_probability


class _SCORING_MATRICES(NamedTuple):
    PAM30_PROBABILITY: np.ndarray = _convert_pam_to_probability(PAM30_SCORING_MATRIX)
    BLOSUM62_PROBABILITY: np.ndarray = _convert_blosum_to_probability(BLOSUM62_SCORING_MATRIX)


SCORING_MATRICES = _SCORING_MATRICES()


def _convert_probability_to_logit(prob_matrix: np.ndarray, eps=1e-10) -> np.ndarray:
    prob_matrix = np.clip(prob_matrix, eps, 1 - eps)
    logit_matrix = np.log(prob_matrix / (1 - prob_matrix))
    return logit_matrix


# @TODO: Replace <pad> with stop as input, but keep <pad> in the obs sequence when computing the loss.
# This way, <pad> does not need to be an extra entry in the scoring matrix (only <end> is needed).
# This encourages not keep generating amino acid letters. We are ignoring the loss of pad tokens anyway!
