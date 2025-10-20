import argparse
import json
import os

from libero.lifelong.data_quality import DemInfScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, help="Path to task hdf5 file")
    parser.add_argument("--obs_keys", type=str, default="robot0_gripper_qpos,robot0_joint_pos")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="deminf_scores.json")
    args = parser.parse_args()

    obs_keys = [k.strip() for k in args.obs_keys.split(",") if k.strip()]

    scorer = DemInfScorer(
        state_mode="state",
        state_latent=12,
        state_beta=0.05,
        action_dim=7,  # default 7-DoF actions; adjust per dataset if needed
        action_latent=6,
        action_beta=0.05,
        device=args.device,
        recon_weight=1.0,
        action_chunk_size=1,
        mi_k_values=(5, 6, 7),
        mi_batch_size=1024,
        mi_iterations=4,
    )

    scores = scorer.score_task_hdf5(args.hdf5, obs_keys)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[info] saved scores to {args.out} with {len(scores)} demos scored")


if __name__ == "__main__":
    main()


