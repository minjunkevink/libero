import json
from typing import Dict, Tuple, List

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .vae import JointVAE
from .mi_estimators import BatchedKnnMIEstimator


class DemInfScorer:
    def __init__(
        self,
        state_mode: str,  # "image" or "state"
        state_latent: int,
        state_beta: float,
        action_dim: int,
        action_latent: int,
        action_beta: float,
        device: str = "cuda",
        recon_weight: float = 1.0,
        action_chunk_size: int = 1,
        mi_k_values: Tuple[int, ...] = (5, 6, 7),
        mi_batch_size: int = 1024,
        mi_iterations: int = 4,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.recon_weight = recon_weight
        self.action_chunk_size = action_chunk_size

        self.model = JointVAE(
            state_input_mode=state_mode,
            state_latent_dim=state_latent,
            state_beta=state_beta,
            action_dim=action_dim,
            action_latent_dim=action_latent,
            action_beta=action_beta,
        ).to(self.device)

        self.mi = BatchedKnnMIEstimator(
            k_values=mi_k_values, batch_size=mi_batch_size, n_iterations=mi_iterations
        )

    def _gather_pairs_from_hdf5(self, hdf5_path: str, obs_keys: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (obs_array, act_array, episode_ids) where obs_array may be low-dim or dict-encoded.
        For images, this function assumes pre-extracted features upstream; initial version focuses on low-dim.
        """
        obs_list = []
        act_list = []
        epi_list = []
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(list(f["data"].keys()))
            for di, ep in enumerate(demos):
                g = f["data"][ep]
                # Expect low-dim observations concatenated under provided keys
                # This is a minimal implementation; image pipelines can be added later.
                obs_parts = []
                for key in obs_keys:
                    if key in g:
                        obs_parts.append(g[key][()])  # [T, Dk]
                if not obs_parts:
                    continue
                obs = np.concatenate(obs_parts, axis=1)
                acts = g["actions"][()]
                T = min(len(obs), len(acts))
                obs_list.append(obs[:T])
                act_list.append(acts[:T])
                epi_list.append(np.full(T, di, dtype=np.int32))
        if len(obs_list) == 0:
            return np.empty((0,)), np.empty((0,)), np.empty((0,))
        return (
            np.concatenate(obs_list, axis=0),
            np.concatenate(act_list, axis=0),
            np.concatenate(epi_list, axis=0),
        )

    @torch.no_grad()
    def _encode_batches(self, obs_np: np.ndarray, act_np: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        z_s_all: List[np.ndarray] = []
        z_a_all: List[np.ndarray] = []
        for start in range(0, len(obs_np), batch_size):
            o = torch.from_numpy(obs_np[start : start + batch_size]).float().to(self.device)
            a = torch.from_numpy(act_np[start : start + batch_size]).float().to(self.device)
            z_s, z_a = self.model.encode(o, a)
            z_s_all.append(z_s.cpu().numpy())
            z_a_all.append(z_a.cpu().numpy())
        return np.concatenate(z_s_all, axis=0), np.concatenate(z_a_all, axis=0)

    def score_task_hdf5(self, hdf5_path: str, obs_keys: List[str]) -> Dict[str, float]:
        obs_np, act_np, epi_ids = self._gather_pairs_from_hdf5(hdf5_path, obs_keys)
        if obs_np.size == 0:
            return {}
        z_s, z_a = self._encode_batches(obs_np, act_np)

        # Batched MI estimates over encoded pairs
        mi_scores = self.mi.estimate(z_s, z_a)

        # Normalize scores (clip 1%-99%, z-score)
        if mi_scores.size == 0:
            return {}
        low, high = np.percentile(mi_scores, 1), np.percentile(mi_scores, 99)
        clipped = np.clip(mi_scores, low, high)
        norm = (clipped - clipped.mean()) / (clipped.std() + 1e-8)

        # Aggregate per-trajectory
        demo_scores: Dict[int, List[float]] = {}
        for s, epi in zip(norm, epi_ids[: len(norm)]):
            demo_scores.setdefault(int(epi), []).append(float(s))

        # Map back to demo names
        scores: Dict[str, float] = {}
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(list(f["data"].keys()))
            for i, name in enumerate(demos):
                vals = demo_scores.get(i, [])
                if vals:
                    scores[name] = float(np.mean(vals))
        return scores

    @staticmethod
    def save_scores(path: str, scores: Dict[str, float]):
        with open(path, "w") as f:
            json.dump(scores, f, indent=2)


