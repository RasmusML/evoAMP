from typing import NamedTuple

import numpy as np
from evoamp.scoring_matrices._blosum import BLOSUM62_SCORING_MATRIX, _convert_blosum_to_probability
from evoamp.scoring_matrices._pam import PAM30_SCORING_MATRIX, _convert_pam_to_probability


def _convert_probability_to_logit(prob_matrix: np.ndarray, eps=1e-10) -> np.ndarray:
    prob_matrix = np.clip(prob_matrix, eps, 1 - eps)
    logit_matrix = np.log(prob_matrix / (1 - prob_matrix))
    return logit_matrix


class _SCORING_MATRICES(NamedTuple):
    PAM30_LOGIT: np.ndarray = _convert_probability_to_logit(_convert_pam_to_probability(PAM30_SCORING_MATRIX))
    BLOSUM62_LOGIT: np.ndarray = _convert_probability_to_logit(_convert_blosum_to_probability(BLOSUM62_SCORING_MATRIX))


SCORING_MATRICES = _SCORING_MATRICES()
