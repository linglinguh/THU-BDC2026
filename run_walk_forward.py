#!/usr/bin/env python3
"""运行非重叠时间滚动验证和多随机种子实验。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="沪深300模型滚动验证")
    parser.add_argument("--seeds", default="42,2026,3407")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-root", default="model/walk_forward")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def derive_fold_dates(data_path: str, count: int) -> list[str]:
    data = pd.read_csv(data_path, usecols=["日期"])
    calendar = pd.Index(sorted(pd.to_datetime(data["日期"]).unique()))
    latest = pd.Timestamp(calendar[-1])
    folds = []
    for offset in reversed(range(count)):
        target = latest - pd.DateOffset(months=2 * offset)
        position = max(0, int(calendar.searchsorted(target, side="right")) - 1)
        folds.append(pd.Timestamp(calendar[position]).strftime("%Y-%m-%d"))
    return folds


def find_resume_checkpoint(run_dir: Path, force: bool = False) -> Path | None:
    """Return an unfinished run's checkpoint unless a clean rerun was requested."""
    if force or (run_dir / "training_summary.json").exists():
        return None
    checkpoint = run_dir / "training_checkpoint.pth"
    return checkpoint if checkpoint.exists() else None


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    fold_dates = derive_fold_dates("data/stock_data.csv", args.folds)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for as_of_date in fold_dates:
        for seed in seeds:
            run_dir = output_root / f"asof_{as_of_date.replace('-', '')}" / f"seed_{seed}"
            summary_path = run_dir / "training_summary.json"
            if summary_path.exists() and not args.force:
                print(f"[SKIP] {as_of_date} seed={seed}")
            else:
                env = os.environ.copy()
                env.update(
                    {
                        "TQDM_DISABLE": "1",
                        "BDC_AS_OF_DATE": as_of_date,
                        "BDC_SEED": str(seed),
                        "BDC_NUM_EPOCHS": str(args.max_epochs),
                        "BDC_EARLY_STOPPING_PATIENCE": str(args.patience),
                        "BDC_BATCH_SIZE": str(args.batch_size),
                        "BDC_VALIDATION_MONTHS": "2",
                        "BDC_OUTPUT_DIR": str(run_dir),
                    }
                )
                resume_checkpoint = find_resume_checkpoint(run_dir, force=args.force)
                if resume_checkpoint is not None:
                    env["BDC_RESUME_CHECKPOINT"] = str(resume_checkpoint)
                    print(
                        f"\n[RESUME] as_of={as_of_date} seed={seed} "
                        f"checkpoint={resume_checkpoint}"
                    )
                else:
                    print(f"\n[RUN] as_of={as_of_date} seed={seed} -> {run_dir}")
                subprocess.run([sys.executable, "-u", "code/src/train.py"], env=env, check=True)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(summary)
            pd.DataFrame(rows).to_csv(output_root / "results.csv", index=False)

    results = pd.DataFrame(rows).sort_values(["seed", "as_of_date"])
    aggregate = results.groupby("seed", as_index=False).agg(
        mean_submission_return=("best_submission_return", "mean"),
        std_submission_return=("best_submission_return", "std"),
        median_best_epoch=("best_epoch", "median"),
        min_submission_return=("best_submission_return", "min"),
    )
    aggregate = aggregate.sort_values(
        ["mean_submission_return", "min_submission_return"], ascending=False
    )
    results.to_csv(output_root / "results.csv", index=False)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    print("\n滚动验证明细:\n", results.to_string(index=False))
    print("\n随机种子汇总:\n", aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
