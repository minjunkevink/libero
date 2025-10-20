import argparse
import json
import os

from libero.lifelong.data_quality import filter_hdf5_by_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input task hdf5")
    parser.add_argument("--scores", type=str, required=True, help="JSON score file")
    parser.add_argument("--out", type=str, required=True, help="Output filtered hdf5 path")
    parser.add_argument("--top_ratio", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.scores, "r") as f:
        scores = json.load(f)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    filter_hdf5_by_scores(args.input, args.out, scores, top_ratio=args.top_ratio)
    print(f"[info] wrote filtered dataset to {args.out}")


if __name__ == "__main__":
    main()


