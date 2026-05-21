"""RFT-Inf: Data Influence Estimator for Reinforcement Fine-Tuning.

Reference:
    Tan et al., "Understanding Data Influence in Reinforcement Finetuning",
    NeurIPS 2025.

The estimator implements Eq. (5) of the paper:

    D_hat(z) = sum_t  (2 * eta_t / N) * <G_z^(t), G_Z^(t)>

where G_z^(t) is the per-sample policy gradient at checkpoint t and
G_Z^(t) = sum_{z' in Z} G_{z'}^(t) is the cumulative gradient over the full
training set.
"""

from .config import InfluenceConfig
from .estimator import RFTInfEstimator
from .rewards import gsm8k_reward, extract_answer_gsm8k

__all__ = [
    "InfluenceConfig",
    "RFTInfEstimator",
    "gsm8k_reward",
    "extract_answer_gsm8k",
]

__version__ = "0.1.0"
