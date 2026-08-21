#!/usr/bin/env python3
"""依据滚动验证的稳健 epoch，为多个种子重拟合最终模型。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default="model/walk_forward/aggregate.csv")
    parser.add_argument("--top-seeds", type=int, default=3)
    parser.add_argument("--output-root", default="model/final_ensemble")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    aggregate = pd.read_csv(args.aggregate).head(args.top_seeds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for row in aggregate.itertuples(index=False):
        seed = int(row.seed)
        epochs = max(1, int(round(row.median_best_epoch)))
        output_dir = output_root / f"seed_{seed}"
        env = os.environ.copy()
        env.update(
            {
                "TQDM_DISABLE": "1",
                "BDC_SEED": str(seed),
                "BDC_NUM_EPOCHS": str(epochs),
                "BDC_BATCH_SIZE": str(args.batch_size),
                "BDC_OUTPUT_DIR": str(output_dir),
                "BDC_EARLY_STOPPING_PATIENCE": "0",
            }
        )
        print(f"[REFIT] seed={seed} epochs={epochs} -> {output_dir}")
        subprocess.run([sys.executable, "-u", "code/src/refit.py"], env=env, check=True)


if __name__ == "__main__":
    main()
