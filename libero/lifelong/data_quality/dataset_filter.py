from typing import Dict, Iterable, Optional

import h5py
import numpy as np


def add_quality_mask(hdf5_path: str, scores: Dict[str, float], mask_name: str) -> None:
    with h5py.File(hdf5_path, "a") as f:
        demos = sorted(list(f["data"].keys()))
        # sort demos by score descending (higher is better)
        order = sorted(((d, scores.get(d, -np.inf)) for d in demos), key=lambda x: x[1], reverse=True)
        kept = np.array([d.encode("utf-8") for d, _ in order], dtype="S")
        mask_group = f.require_group("mask")
        if mask_name in mask_group:
            del mask_group[mask_name]
        mask_group.create_dataset(mask_name, data=kept)


def filter_hdf5_by_scores(
    input_path: str,
    output_path: str,
    scores: Dict[str, float],
    top_ratio: Optional[float] = 0.8,
) -> None:
    """Create a filtered copy of input hdf5 keeping top_ratio of demos by score."""
    with h5py.File(input_path, "r") as fin:
        demos = sorted(list(fin["data"].keys()))
        ranked = sorted(((d, scores.get(d, -np.inf)) for d in demos), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(ranked) * float(top_ratio))) if top_ratio is not None else len(ranked)
        keep = set(d for d, _ in ranked[:k])

        with h5py.File(output_path, "w") as fout:
            # copy root groups and attributes
            g_in = fin["data"]
            g_out = fout.create_group("data")
            for attr in g_in.attrs:
                g_out.attrs[attr] = g_in.attrs[attr]

            for d in demos:
                if d not in keep:
                    continue
                src = g_in[d]
                dst = g_out.create_group(d)
                # copy datasets
                for key in src.keys():
                    src.copy(src[key], dst, name=key)
                # copy attrs
                for attr in src.attrs:
                    dst.attrs[attr] = src.attrs[attr]


