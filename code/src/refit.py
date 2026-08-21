"""使用全部可用标签按固定 epoch 重新拟合最终模型。"""

from __future__ import annotations

import json
import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import config
from model import StockTransformer
from train import (
    AlignedRankingDataset,
    WeightedRankingLoss,
    collate_fn,
    preprocess_data,
    set_seed,
    train_ranking_model,
)


def main():
    set_seed(config.get("seed", 42))
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    data_file = os.path.join(config["data_path"], "stock_data.csv")
    if not os.path.exists(data_file):
        data_file = os.path.join(config["data_path"], "train.csv")

    raw = pd.read_csv(data_file, dtype={"股票代码": str})
    raw["股票代码"] = raw["股票代码"].astype(str).str.zfill(6)
    raw["日期"] = pd.to_datetime(raw["日期"])
    if config.get("as_of_date"):
        raw = raw[raw["日期"] <= pd.Timestamp(config["as_of_date"])].copy()
    stock_ids = sorted(raw["股票代码"].unique())
    stockid2idx = {stock_id: index for index, stock_id in enumerate(stock_ids)}

    processed, features = preprocess_data(raw, is_train=True, stockid2idx=stockid2idx)
    processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
    processed = processed.dropna(subset=features)
    scaler = StandardScaler()
    processed[features] = scaler.fit_transform(processed[features])
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    dataset = AlignedRankingDataset(processed, features, config["sequence_length"])
    if not len(dataset):
        raise ValueError("全量重拟合没有可用样本")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids)).to(device)
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config["top5_weight"],
        pairwise_weight=config["pairwise_weight"],
        base_weight=config.get("base_weight", 1.0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=config["num_epochs"]
    )
    use_amp = config.get("use_amp", False) and device.type == "cuda"
    amp_scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for epoch in range(config["num_epochs"]):
        loss, metrics = train_ranking_model(
            model, loader, criterion, optimizer, device, epoch, None, scaler=amp_scaler
        )
        scheduler.step()
        print(f"[REFIT] epoch {epoch + 1}/{config['num_epochs']} loss={loss:.4f} metrics={metrics}")
        checkpoint_tmp = os.path.join(output_dir, "refit_checkpoint.pth.tmp")
        torch.save(
            {
                "completed_epochs": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "seed": config.get("seed"),
            },
            checkpoint_tmp,
        )
        os.replace(checkpoint_tmp, os.path.join(output_dir, "refit_checkpoint.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    with open(os.path.join(output_dir, "config.json"), "w") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "refit_summary.json"), "w") as file:
        json.dump(
            {
                "seed": config.get("seed"),
                "epochs": config["num_epochs"],
                "samples": len(dataset),
                "as_of_date": str(raw["日期"].max().date()),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(f"全量重拟合完成: {output_dir}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
