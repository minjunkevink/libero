"""Data quality (DemInf) modules for LIBERO.

Implements Demonstration Information Estimation (DemInf):
- VAE encoders for states and actions
- Batched k-NN mutual information estimator (KSG-style)
- Scoring and dataset filtering utilities
"""

from .vae import StateVAE, ActionVAE, JointVAE  # noqa: F401
from .mi_estimators import BatchedKnnMIEstimator  # noqa: F401
from .quality_scorer import DemInfScorer  # noqa: F401
from .dataset_filter import filter_hdf5_by_scores, add_quality_mask  # noqa: F401


