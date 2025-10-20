import numpy as np
from scipy.special import digamma


def _pairwise_l2(x: np.ndarray) -> np.ndarray:
    # x: [N, D]
    # returns [N, N]
    # (x - y)^2 = x^2 + y^2 - 2xy
    x2 = np.sum(x * x, axis=1, keepdims=True)  # [N, 1]
    d2 = x2 + x2.T - 2.0 * (x @ x.T)
    # numerical floor at zero
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2, dtype=np.float64)


class BatchedKnnMIEstimator:
    """
    KSG-style k-NN mutual information estimator using randomized batches.

    - Batch size default 1024
    - Repeat 4 passes with new shuffles
    - Average over k in {5,6,7}
    - Joint distance = max(dist_state, dist_action) (Inf-norm on Z)
    """

    def __init__(self, k_values=(5, 6, 7), batch_size=1024, n_iterations=4):
        self.k_values = tuple(k_values)
        self.batch_size = int(batch_size)
        self.n_iterations = int(n_iterations)

    def estimate(self, z_state: np.ndarray, z_action: np.ndarray) -> np.ndarray:
        assert z_state.shape[0] == z_action.shape[0]
        n = z_state.shape[0]
        mi_values = []

        for _ in range(self.n_iterations):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                if batch_idx.size < 4:
                    continue

                s = z_state[batch_idx]
                a = z_action[batch_idx]

                s_dist = _pairwise_l2(s)
                a_dist = _pairwise_l2(a)
                joint = np.maximum(s_dist, a_dist)

                # sort distances along rows
                # joint_knn[:, k] gives kth neighbor radius
                joint_knn = np.partition(joint, kth=max(self.k_values), axis=1)

                mi_k = []
                for k in self.k_values:
                    # kth neighbor distance per point
                    radii = joint_knn[:, k]
                    # counts of neighbors within radius in marginals
                    s_count = (s_dist < radii[:, None]).sum(axis=1) - 1  # exclude self
                    a_count = (a_dist < radii[:, None]).sum(axis=1) - 1

                    # prevent zeros
                    s_count = np.clip(s_count, 1, None)
                    a_count = np.clip(a_count, 1, None)

                    # KSG estimator core term (constant terms cancel in ranking)
                    mi = -np.mean(digamma(s_count) + digamma(a_count))
                    mi_k.append(mi)

                mi_values.append(float(np.mean(mi_k)))

        return np.asarray(mi_values, dtype=np.float64)


