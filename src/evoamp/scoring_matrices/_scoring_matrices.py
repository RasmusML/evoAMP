import numpy as np
from evoamp.scoring_matrices._blosum import BLOSUM62_SCORING_MATRIX, _convert_blosum_to_probability
from evoamp.scoring_matrices._pam import PAM30_SCORING_MATRIX, _convert_pam_to_probability

SCORING_MATRICES = {
    "PAM30": _convert_pam_to_probability(PAM30_SCORING_MATRIX),
    "BLOSUM62": _convert_blosum_to_probability(BLOSUM62_SCORING_MATRIX),
}


def _convert_probability_to_logit(prob_matrix: np.ndarray, eps=1e-10) -> np.ndarray:
    prob_matrix = np.clip(prob_matrix, eps, 1 - eps)
    logit_matrix = np.log(prob_matrix / (1 - prob_matrix))
    return logit_matrix
