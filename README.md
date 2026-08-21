# THU Big Data Challenge 2026

面向沪深 300 成分股的排序学习选股方案。模型读取每只股票过去 60 个公共交易日的量价与技术特征，同时建模单股时序和股票间关系，最终输出不超过 5 只股票及总和不超过 1 的权重。

## 核心流程

1. 从 `data/stock_data.csv` 读取历史行情；本地兼容回退到 `data/train.csv`。
2. 按股票计算 39 或 `158+39` 组技术特征。股票代码仅作为元数据，不作为连续数值特征输入模型。
3. 按公共交易日历构造 T+1 至 T+5 的收益标签；股票缺少任一目标交易日时，该标签无效。
4. 以交易日为样本构造严格对齐的紧凑数据集，并训练 `StockTransformer` 排序模型。
5. 验证和推理共用 `code/src/strategy.py`：多因子排序、行业分散、稳定 softmax 与组合合法性校验。
6. 以验证期 `submission_return` 保存最佳模型，推理结果写入 `output/result.csv`。

## 主要文件

- `code/src/train.py`：特征预处理、交易日标签、对齐数据集、训练与验证。
- `code/src/predict.py`：对齐推理序列、模型打分与结果输出。
- `code/src/strategy.py`：训练验证和正式推理共用的组合策略。
- `code/src/utils.py`：39/158 类技术特征工程。
- `code/src/config.py`：模型、训练和组合策略参数。
- `get_stock_data.py`：从 Baostock 下载沪深 300 历史行情。
- `test/test_regressions.py`：交易日对齐、标签、特征和组合策略回归测试。

训练产物位于 `model/`：

- `best_model.pth`
- `scaler.pkl`
- `config.json`
- `final_score.txt`
- `log/`

## 本地运行

依赖 Python 3.10–3.12、uv 和 TA-Lib 系统库。

```bash
uv sync --frozen
sh train.sh
sh test.sh
```

下载指定时间范围的数据：

```bash
.venv/bin/python get_stock_data.py \
  --start-date 2023-07-23 \
  --end-date 2026-07-23 \
  --output data/stock_data.csv
```

不传日期时，结束日默认为当天，开始日默认为结束日前 3 年。

## 测试

快速回归：

```bash
.venv/bin/python -m unittest discover -s test -p 'test_regressions.py' -v
.venv/bin/python test/score_self.py
```

`test/score_self.py` 仅适用于 `data/test.csv` 覆盖预测组合中每只股票后续完整
5 个交易日的历史回放；对尚无未来标签的正式提交结果只执行格式与组合合法性校验。

真实数据小规模训练—预测烟测：

```bash
BDC_NUM_EPOCHS=1 \
BDC_SMOKE_MAX_SAMPLES=2 \
BDC_SMOKE_MAX_STOCKS=20 \
  .venv/bin/python code/src/train.py
.venv/bin/python code/src/predict.py
```

烟测变量仅用于本地验证；不设置时默认执行全量股票、全量日期和最多 15 个 epoch，
并按验证组合收益早停。
在 8GB Apple Silicon 上可设置 `BDC_BATCH_SIZE=1` 避免 MPS 统一内存换页；
它只改变梯度更新的批大小，不会裁剪股票、日期或 epoch。

每个完整 epoch 会写入 `training_checkpoint.pth`。中断后可精确续训：

```bash
BDC_BATCH_SIZE=1 \
BDC_RESUME_CHECKPOINT=model/60_158+39/training_checkpoint.pth \
  .venv/bin/python code/src/train.py
```

旧版本若只有 `best_model.pth`，可以通过 `BDC_RESUME_FROM_BEST=1` 和
`BDC_RESUME_EPOCH` 从最佳权重继续，但优化器动量会重新初始化。

滚动验证会自动跳过已完成实验，并从未完成目录中的
`training_checkpoint.pth` 续跑：

```bash
.venv/bin/python run_walk_forward.py
```

保存模型后可在同一历史截面上校准组合后处理参数。工具会分别评估各模型和
截面标准化集成，并输出完整搜索结果：

```bash
.venv/bin/python run_strategy_search.py \
  model/walk_forward/asof_20260323/seed_42 \
  model/walk_forward/asof_20260323/seed_2026 \
  --output model/strategy_search/asof_20260323.csv
```

组合策略参数改变后，旧检查点中的最佳验证分数已不可比较；续训应使用新的
`BDC_OUTPUT_DIR`，或对滚动实验显式使用 `--force` 从头训练。

## Docker 全流程烟测

CPU 镜像适合本地或 Apple Silicon 验证：

```bash
docker build \
  --build-arg CPU_ONLY=1 \
  -t bdc2026:latest .
docker compose -f docker-compose.smoke.yml up --abort-on-container-exit
docker compose -f docker-compose.smoke.yml down
```

正式镜像不传 `CPU_ONLY=1`，将使用 `uv.lock` 中冻结的 CUDA 依赖。网络受限时可通过 `BASE_IMAGE` 和 `DEBIAN_MIRROR` 构建参数指定镜像源。

## 提交前检查

- 使用赛事最新挂载数据，或先更新 `data/stock_data.csv`；不要用过期行情生成正式结果。
- 取消所有 `BDC_SMOKE_*` 环境变量并完成正式训练。
- 在与比赛一致的 Docker 环境中执行 `data/run.sh`，以 Docker 生成的 `output/result.csv` 为准。
- 确认结果股票代码唯一、权重为有限非负数、股票数不超过 5、权重和不超过 1。
