from evoamp.scoring_matrices._blosum import (
    BLOSUM62_SCORING_MATRIX,
    BLOSUM80_SCORING_MATRIX,
    _convert_blosum_to_probability,
)
from evoamp.scoring_matrices._pam import PAM30_SCORING_MATRIX, _convert_pam_to_probability

SCORING_MATRICES = {
    "PAM30": _convert_pam_to_probability(PAM30_SCORING_MATRIX),
    "BLOSUM62": _convert_blosum_to_probability(BLOSUM62_SCORING_MATRIX),
    "BLOSUM80": _convert_blosum_to_probability(BLOSUM80_SCORING_MATRIX),
}
