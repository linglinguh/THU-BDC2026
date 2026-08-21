#!/usr/bin/env python3
"""生成当前沪深300的中证一级行业静态映射，供离线训练和 Docker 使用。"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="更新当前沪深300中证一级行业映射")
    parser.add_argument("--output", default="code/src/industry_map.json")
    parser.add_argument("--cache-dir", default="temp/industry_map_parts")
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def choose_industry(records: pd.DataFrame) -> str | None:
    if records is None or records.empty:
        return None
    records = records.copy()
    standards = records["分类标准"].fillna("").astype(str)
    preferred = records[standards.eq("中证行业分类标准")]
    if preferred.empty:
        preferred = records[standards.str.contains("中证行业分类标准") & ~standards.str.contains("旧")]
    if preferred.empty:
        preferred = records[standards.eq("巨潮行业分类标准")]
    if preferred.empty:
        return None
    preferred = preferred.copy()
    preferred["变更日期"] = pd.to_datetime(preferred["变更日期"], errors="coerce")
    row = preferred.sort_values("变更日期").iloc[-1]
    for column in ("行业门类", "行业大类", "行业中类"):
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return None


def main():
    args = parse_args()
    output = Path(args.output)
    cache_dir = Path(args.cache_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    constituents = ak.index_stock_cons_csindex(symbol="000300").copy()
    constituents["成分券代码"] = constituents["成分券代码"].astype(str).str.zfill(6)
    constituents = constituents.drop_duplicates("成分券代码").sort_values("成分券代码")
    if len(constituents) != 300:
        raise ValueError(f"沪深300成分股数量异常: {len(constituents)}")

    mapping = {}
    failures = []
    for position, row in constituents.reset_index(drop=True).iterrows():
        code = row["成分券代码"]
        name = row["成分券名称"]
        cache_path = cache_dir / f"{code}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            industry = cached.get("industry")
        else:
            industry = None
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    records = ak.stock_industry_change_cninfo(
                        symbol=code,
                        start_date="19900101",
                        end_date=date.today().strftime("%Y%m%d"),
                    )
                    industry = choose_industry(records)
                    if not industry:
                        raise ValueError("没有可用的中证/巨潮行业记录")
                    cache_path.write_text(
                        json.dumps({"code": code, "name": name, "industry": industry}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"[{position + 1:03d}/300] {code} {name}: 第{attempt}次失败: {exc}")
                    time.sleep(min(2**attempt, 10))
            if last_error is not None:
                failures.append((code, name, str(last_error)))
                continue
        mapping[code] = industry
        print(f"[{position + 1:03d}/300] {code} {name}: {industry}")
        time.sleep(0.2)

    if failures:
        print("行业获取失败:", failures)
    if len(mapping) < 290:
        raise RuntimeError(f"有效行业映射不足: {len(mapping)}/300")

    payload = {
        "source": "巨潮资讯-中证行业分类标准（行业门类）",
        "as_of": date.today().isoformat(),
        "stocks": dict(sorted(mapping.items())),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"已写入 {output}: {len(mapping)}/300")


if __name__ == "__main__":
    main()
