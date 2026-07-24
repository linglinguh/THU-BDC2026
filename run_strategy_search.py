#!/usr/bin/env python3
"""Search portfolio post-processing parameters on saved walk-forward models."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code" / "src"))

from config import config as model_config
from model import StockTransformer
from predict import normalize_cross_section
from strategy import select_portfolio
from train import AlignedRankingDataset, collate_fn, preprocess_val_data, split_train_val_by_last_month


FACTOR_VARIANTS = {
    "model_only": (1.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    "current": (0.80, 0.10, 0.00, 0.05, 0.02, 0.03),
    "momentum": (0.60, 0.30, 0.00, 0.05, 0.02, 0.03),
    "reversal": (0.65, 0.05, 0.20, 0.05, 0.02, 0.03),
    "low_vol": (0.65, 0.10, 0.00, 0.20, 0.02, 0.03),
    "balanced": (0.50, 0.20, 0.10, 0.10, 0.05, 0.05),
}


def parse_args():
    parser = argparse.ArgumentParser(description="已训练模型的组合策略滚动校准")
    parser.add_argument("model_dirs", nargs="+", help="同一历史截面的模型目录")
    parser.add_argument("--data", default="data/stock_data.csv")
    parser.add_argument("--output", default="model/strategy_search/results.csv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cutoff", help="模型配置未记录 as_of_date 时显式指定历史截止日")
    return parser.parse_args()


def load_run_cutoff(model_dirs: list[Path], explicit_cutoff: str | None = None) -> pd.Timestamp:
    if explicit_cutoff:
        return pd.Timestamp(explicit_cutoff)
    cutoffs = set()
    for model_dir in model_dirs:
        run_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        cutoff = run_config.get("as_of_date")
        if not cutoff:
            raise ValueError(f"模型目录缺少 as_of_date: {model_dir}")
        cutoffs.add(cutoff)
    if len(cutoffs) != 1:
        raise ValueError(f"只能搜索同一历史截面的模型，当前为: {sorted(cutoffs)}")
    return pd.Timestamp(cutoffs.pop())


def collect_predictions(raw, model_dirs, batch_size):
    stock_ids = sorted(raw["股票代码"].unique())
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}
    idx2stock = {index: stock_id for stock_id, index in stockid2idx.items()}
    _, validation, val_start = split_train_val_by_last_month(
        raw, model_config["sequence_length"], validation_months=model_config["validation_months"]
    )
    processed, features = preprocess_val_data(validation, stockid2idx=stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    targets_by_date = None
    for model_dir in model_dirs:
        scaled = processed.copy()
        scaler = joblib.load(model_dir / "scaler.pkl")
        scaled[features] = scaler.transform(scaled[features])
        dataset = AlignedRankingDataset(
            scaled,
            features,
            model_config["sequence_length"],
            min_window_end_date=val_start.strftime("%Y-%m-%d"),
        )
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
        model = StockTransformer(len(features), model_config, len(stock_ids)).to(device)
        model.load_state_dict(
            torch.load(model_dir / "best_model.pth", map_location=device, weights_only=True)
        )
        model.eval()
        scores_by_date = {}
        current_targets = {}
        with torch.no_grad():
            for batch in loader:
                masks = batch["masks"].bool()
                outputs = model(batch["sequences"].to(device), mask=batch["masks"].to(device)).cpu()
                for row, date in enumerate(batch["dates"]):
                    valid = masks[row]
                    ids = [idx2stock[int(value)] for value in batch["stock_indices"][row][valid]]
                    scores_by_date[pd.Timestamp(date)] = dict(zip(ids, outputs[row][valid].numpy()))
                    current_targets[pd.Timestamp(date)] = dict(
                        zip(ids, batch["targets"][row][valid].numpy())
                    )
        predictions.append(scores_by_date)
        targets_by_date = current_targets if targets_by_date is None else targets_by_date
        print(f"模型验证预测完成: {model_dir}，{len(scores_by_date)} 个截面")
    return predictions, targets_by_date


def score_sources(predictions):
    sources = [(f"model_{index + 1}", values) for index, values in enumerate(predictions)]
    if len(predictions) > 1:
        ensemble = {}
        for date in sorted(set.intersection(*(set(values) for values in predictions))):
            ids = sorted(set.intersection(*(set(values[date]) for values in predictions)))
            normalized = [
                normalize_cross_section([values[date][stock_id] for stock_id in ids])
                for values in predictions
            ]
            ensemble[date] = dict(zip(ids, np.mean(np.stack(normalized), axis=0)))
        sources.append(("ensemble", ensemble))
    return sources


def strategy_grid():
    for factor_name, weights in FACTOR_VARIANTS.items():
        for pool_size, industry_cap, temperature in itertools.product(
            (5, 10, 15, 30), (0, 1, 2), (0.2, 0.5, 1.0, 2.0, 100.0)
        ):
            yield {
                "factor_variant": factor_name,
                "candidate_pool_size": pool_size,
                "enable_multi_factor": True,
                "enable_industry_diversify": industry_cap > 0,
                "industry_max_per_sector": max(industry_cap, 1),
                "predict_temperature": temperature,
                "multi_factor_model_weight": weights[0],
                "multi_factor_momentum_weight": weights[1],
                "multi_factor_reversal_weight": weights[2],
                "multi_factor_volatility_weight": weights[3],
                "multi_factor_liquidity_weight": weights[4],
                "multi_factor_volume_weight": weights[5],
            }


def evaluate(raw, sources, targets_by_date):
    rows = []
    for source_name, scores_by_date in sources:
        for search_config in strategy_grid():
            returns = []
            for date, score_map in scores_by_date.items():
                ranked = sorted(score_map, key=score_map.get, reverse=True)
                selected, weights = select_portfolio(
                    raw,
                    ranked,
                    [score_map[stock_id] for stock_id in ranked],
                    date,
                    search_config,
                )
                targets = targets_by_date[date]
                returns.append(sum(weight * targets[stock_id] for stock_id, weight in zip(selected, weights)))
            rows.append(
                {
                    "score_source": source_name,
                    **{key: search_config[key] for key in (
                        "factor_variant", "candidate_pool_size", "enable_industry_diversify",
                        "industry_max_per_sector", "predict_temperature"
                    )},
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "min_return": float(np.min(returns)),
                    "win_rate": float(np.mean(np.asarray(returns) > 0)),
                    "validation_days": len(returns),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["mean_return", "min_return", "win_rate"], ascending=False
    )


def main():
    args = parse_args()
    model_dirs = [Path(value) for value in args.model_dirs]
    cutoff = load_run_cutoff(model_dirs, args.cutoff)
    raw = pd.read_csv(args.data, dtype={"股票代码": str})
    raw["股票代码"] = raw["股票代码"].astype(str).str.zfill(6)
    raw["日期"] = pd.to_datetime(raw["日期"])
    raw = raw[raw["日期"] <= cutoff].copy()
    predictions, targets = collect_predictions(raw, model_dirs, args.batch_size)
    results = evaluate(raw, score_sources(predictions), targets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print("\n最佳策略:\n", results.head(20).to_string(index=False))
    print(f"\n完整结果已写入: {output}")


if __name__ == "__main__":
    main()
