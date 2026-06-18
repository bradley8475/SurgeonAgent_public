"""按 panel 数 + topology 多样性切 train/test。

输出：
  splits/test.txt          每行一个 pattern.json 绝对路径
  splits/train.txt         同上
  splits/test_meta.json    每个测试样本的 path / n_panels / signature
  splits/split_stats.json  分布统计、topology 重叠率等

用法：
  uv run python utils/build_split.py
"""

import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

from cad.api import load_pattern

DATA_ROOTS = [
    "../garment_data/gcd/garments_5000_0",
    "../garment_data/template/1015",
    "../garment_data/clo/1015",
]

# (low, high) inclusive; quota: how many test samples to draw from this bin.
BINS = [
    ("easy", 2, 5, 25),
    ("medium", 6, 10, 25),
    ("hard", 11, 18, 25),
    ("very_hard", 19, 999, 25),
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "splits")
SEED = 42


def collect_records():
    records = []
    for root in DATA_ROOTS:
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if not os.path.isdir(full):
                continue
            p = os.path.join(full, "pattern.json")
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                continue
            try:
                pat = load_pattern(p)
                sig = tuple(sorted(panel.name for panel in pat.panels))
                records.append({
                    "path": p,
                    "n_panels": len(pat.panels),
                    "sig": sig,
                })
            except Exception as e:
                print(f"  ! load_pattern failed: {p}: {e}", file=sys.stderr)
    return records


def split_bin(records_in_bin, quota, rng):
    """Within a bin, sample `quota` paths maximizing topology diversity.

    Pass 1: shuffle unique signatures, pick one sample per signature until quota
            filled or signatures exhausted.
    Pass 2 (overflow): shuffle remaining samples globally and fill the gap.
    """
    by_sig = defaultdict(list)
    for r in records_in_bin:
        by_sig[r["sig"]].append(r)

    sigs = list(by_sig.keys())
    rng.shuffle(sigs)

    picked = []
    for sig in sigs:
        bucket = by_sig[sig]
        rng.shuffle(bucket)
        picked.append(bucket.pop(0))
        if len(picked) >= quota:
            break

    if len(picked) < quota:
        leftovers = [r for bucket in by_sig.values() for r in bucket]
        rng.shuffle(leftovers)
        picked.extend(leftovers[: quota - len(picked)])

    return picked


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"loading patterns from {len(DATA_ROOTS)} roots...")
    t0 = time.time()
    records = collect_records()
    print(f"  loaded {len(records)} valid patterns in {time.time()-t0:.1f}s")

    test_records = []
    for name, lo, hi, quota in BINS:
        in_bin = [r for r in records if lo <= r["n_panels"] <= hi]
        n_sigs = len(set(r["sig"] for r in in_bin))
        picked = split_bin(in_bin, quota, rng)
        actual_sigs = len(set(r["sig"] for r in picked))
        print(
            f"  bin={name:9s} range=[{lo},{hi}] pool={len(in_bin):4d} "
            f"unique_sig={n_sigs:3d} picked={len(picked)} sigs_in_test={actual_sigs}"
        )
        for r in picked:
            r["bin"] = name
        test_records.extend(picked)

    test_paths = {r["path"] for r in test_records}
    train_records = [r for r in records if r["path"] not in test_paths]

    test_sigs = set(r["sig"] for r in test_records)
    train_sigs = set(r["sig"] for r in train_records)
    overlap_sigs = test_sigs & train_sigs
    train_samples_with_test_sig = sum(1 for r in train_records if r["sig"] in overlap_sigs)

    with open(os.path.join(OUT_DIR, "test.txt"), "w") as f:
        f.write("\n".join(r["path"] for r in test_records) + "\n")
    with open(os.path.join(OUT_DIR, "train.txt"), "w") as f:
        f.write("\n".join(r["path"] for r in train_records) + "\n")
    with open(os.path.join(OUT_DIR, "test_meta.json"), "w") as f:
        json.dump(
            [
                {
                    "path": r["path"],
                    "n_panels": r["n_panels"],
                    "signature": list(r["sig"]),
                    "bin": r["bin"],
                }
                for r in test_records
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    stats = {
        "seed": SEED,
        "total": len(records),
        "test_count": len(test_records),
        "train_count": len(train_records),
        "bins": [
            {
                "name": name,
                "panel_range": [lo, hi if hi < 999 else None],
                "pool_size": sum(1 for r in records if lo <= r["n_panels"] <= hi),
                "test_picked": sum(1 for r in test_records if r["bin"] == name),
            }
            for (name, lo, hi, _) in BINS
        ],
        "topology_signatures": {
            "total_unique": len(set(r["sig"] for r in records)),
            "in_test": len(test_sigs),
            "in_train": len(train_sigs),
            "shared_with_train": len(overlap_sigs),
            "train_samples_sharing_topology_with_test": train_samples_with_test_sig,
            "leakage_rate": round(
                train_samples_with_test_sig / len(train_records), 4
            ),
        },
        "panel_count_distribution_test": dict(
            sorted(Counter(r["n_panels"] for r in test_records).items())
        ),
    }
    with open(os.path.join(OUT_DIR, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\ntest:  {len(test_records)} → {OUT_DIR}/test.txt")
    print(f"train: {len(train_records)} → {OUT_DIR}/train.txt")
    print(f"\ntopology overlap (loose-split disclosure):")
    print(f"  test signatures: {len(test_sigs)}")
    print(f"  shared with train: {len(overlap_sigs)} ({len(overlap_sigs)/len(test_sigs)*100:.1f}% of test sigs)")
    print(f"  train samples sharing test topology: {train_samples_with_test_sig} ({stats['topology_signatures']['leakage_rate']*100:.2f}% of train)")


if __name__ == "__main__":
    main()
