#!/usr/bin/env python3
"""从中证指数与新浪财经下载当前沪深300成分股的后复权日线数据。

该脚本是 Baostock 暂时不可用时的公开备用数据源。下载过程按股票保存分片，
中断后再次运行同一命令会自动跳过已经成功完成的股票。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "股票代码",
    "日期",
    "开盘",
    "收盘",
    "最高",
    "最低",
    "成交量",
    "成交额",
    "振幅",
    "涨跌额",
    "换手率",
    "涨跌幅",
]


def parse_args():
    today = pd.Timestamp.today().normalize()
    parser = argparse.ArgumentParser(
        description="从中证指数和新浪财经下载当前沪深300最近三年行情"
    )
    parser.add_argument(
        "--start-date",
        default=(today - pd.DateOffset(years=3)).strftime("%Y-%m-%d"),
    )
    parser.add_argument("--end-date", default=today.strftime("%Y-%m-%d"))
    parser.add_argument("--output", default="data/stock_data.csv")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-interval", type=float, default=0.5)
    return parser.parse_args()


def market_symbol(stock_code: str) -> str:
    stock_code = str(stock_code).zfill(6)
    if stock_code.startswith(("5", "6", "9")):
        return f"sh{stock_code}"
    if stock_code.startswith(("0", "1", "2", "3")):
        return f"sz{stock_code}"
    raise ValueError(f"无法判断股票所属市场: {stock_code}")


def get_current_constituents() -> pd.DataFrame:
    constituents = ak.index_stock_cons_csindex(symbol="000300").copy()
    required = {"成分券代码", "成分券名称"}
    if not required.issubset(constituents.columns):
        raise ValueError(f"中证指数成分股字段异常: {list(constituents.columns)}")
    constituents["成分券代码"] = constituents["成分券代码"].astype(str).str.zfill(6)
    constituents = constituents.drop_duplicates("成分券代码").sort_values("成分券代码")
    if len(constituents) != 300:
        raise ValueError(f"沪深300成分股数量应为300，当前为 {len(constituents)}")
    return constituents.reset_index(drop=True)


def convert_history(raw: pd.DataFrame, stock_code: str, start_date: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError("返回数据为空")

    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "turnover",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    previous_close = frame["close"].shift(1)

    result = pd.DataFrame(
        {
            "股票代码": stock_code,
            "日期": frame["date"],
            "开盘": frame["open"],
            "收盘": frame["close"],
            "最高": frame["high"],
            "最低": frame["low"],
            "成交量": frame["volume"],
            "成交额": frame["amount"],
            "振幅": (frame["high"] - frame["low"]) / previous_close * 100,
            "涨跌额": frame["close"] - previous_close,
            # 新浪 turnover 是比例，比赛基准 CSV 使用百分数。
            "换手率": frame["turnover"] * 100,
            "涨跌幅": frame["close"].pct_change(fill_method=None) * 100,
        }
    )
    result = result[result["日期"] >= pd.Timestamp(start_date)].copy()
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=OUTPUT_COLUMNS)
    result["日期"] = result["日期"].dt.strftime("%Y-%m-%d")
    return result[OUTPUT_COLUMNS]


def fetch_stock(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    # 多取十个自然日，为目标区间第一天计算涨跌幅和振幅提供前收盘价。
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=10)).strftime("%Y%m%d")
    raw = ak.stock_zh_a_daily(
        symbol=market_symbol(stock_code),
        start_date=fetch_start,
        end_date=pd.Timestamp(end_date).strftime("%Y%m%d"),
        adjust="hfq",
    )
    return convert_history(raw, stock_code, start_date)


def validate_output(frame: pd.DataFrame, expected_codes: set[str], end_date: str) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"输出字段异常: {list(frame.columns)}")
    codes = set(frame["股票代码"].astype(str).str.zfill(6))
    if codes != expected_codes:
        missing = sorted(expected_codes - codes)
        extra = sorted(codes - expected_codes)
        raise ValueError(f"股票集合不一致，缺失={missing}，多余={extra}")
    if frame.duplicated(["股票代码", "日期"]).any():
        raise ValueError("输出中存在重复的股票-日期记录")
    numeric = frame.drop(columns=["股票代码", "日期"]).to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("输出中存在 NaN 或 Inf")
    latest = pd.to_datetime(frame["日期"]).max()
    if latest < pd.Timestamp(end_date) - pd.Timedelta(days=7):
        raise ValueError(f"最新行情日期异常: {latest.date()}")


def main():
    args = parse_args()
    start_date = pd.Timestamp(args.start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(args.end_date).strftime("%Y-%m-%d")
    if start_date > end_date:
        raise ValueError(f"开始日期晚于结束日期: {start_date} > {end_date}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = output_path.with_suffix(output_path.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    print("获取中证指数官网的当前沪深300成分股...")
    constituents = get_current_constituents()
    list_path = output_path.parent / "hs300_stock_list_akshare.csv"
    constituents.to_csv(list_path, index=False, encoding="utf-8-sig")
    print(f"成分股日期: {constituents.iloc[0].get('日期', '未知')}，共 {len(constituents)} 只")

    failures = []
    for position, row in constituents.iterrows():
        stock_code = row["成分券代码"]
        stock_name = row["成分券名称"]
        part_path = parts_dir / f"{stock_code}.csv"
        if part_path.exists():
            try:
                existing = pd.read_csv(part_path, dtype={"股票代码": str})
                if not existing.empty and list(existing.columns) == OUTPUT_COLUMNS:
                    print(f"[{position + 1:03d}/300] {stock_code} {stock_name}: 已完成，跳过")
                    continue
            except Exception:
                pass

        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                history = fetch_stock(stock_code, start_date, end_date)
                # 60 日滚动特征 + 60 日模型窗口至少需要约 120 条记录；
                # 新上市成分股可能没有完整三年历史，但 130 条已足够形成样本。
                if len(history) < 130:
                    raise ValueError(f"历史记录过少: {len(history)}")
                history.to_csv(part_path, index=False, encoding="utf-8-sig")
                print(
                    f"[{position + 1:03d}/300] {stock_code} {stock_name}: "
                    f"{len(history)} 条，至 {history['日期'].max()}"
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[{position + 1:03d}/300] {stock_code} {stock_name}: "
                    f"第 {attempt} 次失败: {exc}"
                )
                time.sleep(min(2**attempt, 10))
        if last_error is not None:
            failures.append((stock_code, stock_name, str(last_error)))
        time.sleep(max(args.request_interval, 0))

    if failures:
        failure_path = output_path.parent / "failed_stocks_akshare.csv"
        pd.DataFrame(failures, columns=["股票代码", "股票名称", "错误"]).to_csv(
            failure_path, index=False, encoding="utf-8-sig"
        )
        raise RuntimeError(f"仍有 {len(failures)} 只股票下载失败，详情见 {failure_path}")

    frames = [
        pd.read_csv(parts_dir / f"{code}.csv", dtype={"股票代码": str})
        for code in constituents["成分券代码"]
    ]
    result = pd.concat(frames, ignore_index=True)
    result["股票代码"] = result["股票代码"].astype(str).str.zfill(6)
    result = result.sort_values(["股票代码", "日期"]).reset_index(drop=True)
    validate_output(result, set(constituents["成分券代码"]), end_date)

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    result.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(output_path)
    print(
        f"下载完成: {output_path}\n"
        f"股票={result['股票代码'].nunique()}，记录={len(result)}，"
        f"日期={result['日期'].min()} 至 {result['日期'].max()}"
    )


if __name__ == "__main__":
    main()
