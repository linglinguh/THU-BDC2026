# 股票排序预测系统 — 规格说明 (SDD)

> **版本**: 1.0 | **最后更新**: 2026-07-22 | **基线版本**: THU-BDC2026-baseline

---

## 1. 系统概述

### 1.1 目标
基于沪深300成分股的过去60个交易日量价与技术特征，预测未来5个交易日收益最高的股票组合（≤5只，权重之和≤1）。

### 1.2 整体数据流

```
[原始股票数据 CSV]
       │
       ▼
  ┌─────────────┐
  │ 数据获取     │  get_stock_data.py  /  data/split_train_test.py
  └─────────────┘
       │  train.csv
       ▼
  ┌─────────────┐
  │ 特征工程     │  utils.py (engineer_features_*)
  │ 标准化       │  StandardScaler
  │ 排序数据集   │  create_ranking_dataset_vectorized()
  └─────────────┘
       │  sequences, targets, relevance, stock_indices
       ▼
  ┌─────────────┐     config.py: 超参数
  │ 模型训练     │  ──▶ StockTransformer (model.py)
  │ 评估保存     │  ──▶ WeightedRankingLoss (train.py)
  └─────────────┘
       │  best_model.pth, scaler.pkl
       ▼
  ┌─────────────┐
  │ 推理预测     │  predict.py → output/result.csv
  └─────────────┘
       │  stock_id + weight
       ▼
  ┌─────────────┐
  │ 评分         │  score_self.py / score_docker.py
  └─────────────┘
       │  Final Score (加权收益率)
```

---

## 2. 模块规格

### 2.1 `code/src/config.py` — 配置中心

**职责**: 全局超参数与路径定义的**单一数据源**。所有模块都从这里读取配置，不得硬编码。

**接口**:
```python
config: dict  # 全局配置字典
sequence_length: int = 60
feature_num: str = '158+39'
```

**必须包含的键**:

| 键 | 类型 | 默认值 | 必须 | 说明 |
|---|---|---|---|---|
| `sequence_length` | int | 60 | ✅ | 输入序列长度（交易日） |
| `feature_num` | str | `'158+39'` | ✅ | 特征集标识 |
| `d_model` | int | 256 | ✅ | Transformer 嵌入维度 |
| `nhead` | int | 4 | ✅ | 注意力头数 |
| `num_layers` | int | 3 | ✅ | Transformer 层数 |
| `dim_feedforward` | int | 512 | ✅ | 前馈网络维度 |
| `batch_size` | int | 4 | ✅ | 训练批次大小 |
| `num_epochs` | int | 50 | ✅ | 训练轮数 |
| `learning_rate` | float | 1e-5 | ✅ | 初始学习率 |
| `dropout` | float | 0.1 | ✅ | Dropout 比率 |
| `max_grad_norm` | float | 5.0 | ✅ | 梯度裁剪阈值 |
| `pairwise_weight` | float | 1 | ✅ | 配对损失权重 |
| `base_weight` | float | 1.0 | ✅ | 非 top-k 样本权重 |
| `top5_weight` | float | 2.0 | ✅ | top-5 样本权重 |
| `use_amp` | bool | True | ✅ | 启用混合精度 |
| `predict_temperature` | float | 1.0 | ✅ | 推理 softmax 温度 |
| `output_dir` | str | 自动生成 | ✅ | 模型产物输出目录 |
| `data_path` | str | `'./data'` | ✅ | 数据文件目录 |

**约束**:
- `nhead` 必须整除 `d_model`
- `feature_num` 只能是 `'39'` 或 `'158+39'`
- 新增配置项必须对此文件修改

---

### 2.2 `code/src/model.py` — 模型定义

**职责**: 定义 `StockTransformer` 模型，输入多只股票的时序特征，输出每只股票的排序分数。

**核心类**: `StockTransformer(nn.Module)`

| 项目 | 规格 |
|---|---|
| **输入形状** | `[batch, num_stocks, seq_len, feature_dim]` |
| **输出形状** | `[batch, num_stocks]` — 每只股票的排序分数（越大越好） |
| **参数量** | 约 4M（197特征、d_model=256） |

**子模块序列**:
1. `input_proj` (Linear): `feature_dim → d_model`
2. `PositionalEncoding`: 正弦位置编码
3. `temporal_encoder` (TransformerEncoder): 提取单股票时序模式
4. `FeatureAttention`: 时间维特征加权聚合
5. `CrossStockAttention` (MultiheadAttention): 同一日股票间交互建模
6. `ranking_layers` (Sequential): d_model → d_model → d_model/2
7. `score_head` (Sequential): d_model/2 → d_model/4 → 1

**约束**:
- 修改网络结构时，必须保持输入/输出形状一致
- `_init_weights()` 使用 Xavier 初始化
- 支持 `batch_first=True`

---

### 2.3 `code/src/utils.py` — 特征工程与数据集

**职责**: 提供特征计算函数和排序数据集构建。

**函数接口**:

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `engineer_features_39(df_group)` | `DataFrame`（单股票数据） | `DataFrame`（含39特征） | 39个技术指标 |
| `engineer_features_158plus39(df_group)` | `DataFrame`（单股票数据） | `DataFrame`（含197特征） | 158 Alpha + 39 技术指标 |
| `create_ranking_dataset_vectorized(data, features, seq_len, ...)` | 预处理后的完整DataFrame | `(sequences, targets, relevance, stock_indices)` | 按日切分排序样本 |

**`create_ranking_dataset_vectorized` 返回值规格**:
```python
sequences:      List[np.ndarray]   # 每元素形状 [num_stocks_day, seq_len, len(features)]
targets:        List[np.ndarray]   # 每元素形状 [num_stocks_day]  — 真实5日收益率
relevance:      List[np.ndarray]   # 每元素形状 [num_stocks_day]  — 排序标签（整数0..N-1）
stock_indices:  List[np.ndarray]   # 每元素形状 [num_stocks_day]  — 股票全局ID
```

**约束**:
- 特征工程依赖 `TA-Lib` C 库
- 所有特征函数必须接收单股票分组数据
- 新增特征时，必须同步更新 `feature_cloums_map`（在 train.py 和 predict.py 中各有一份）

---

### 2.4 `code/src/train.py` — 训练流程

**职责**: 完整的训练 pipeline，从数据加载到模型保存。

**关键函数签名**:

```python
# 训练入口
def train_ranking_model(
    model: StockTransformer,
    dataloader: DataLoader,
    criterion: WeightedRankingLoss,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    scaler: GradScaler | None = None  # AMP 混合精度
) -> tuple[float, dict]

# 评估
def evaluate_ranking_model(
    model, dataloader, criterion, device, writer, epoch,
    use_amp: bool = False
) -> tuple[float, dict]
```

**损失函数**: `WeightedRankingLoss`
- `listwise_loss`: 加权 KL散度 + CrossEntropy
- `pairwise_loss`: 加权 sigmoid 配对比较
- 对真实 Top-5 样本施加 `top5_weight` 倍权重

**评估指标** (`calculate_ranking_metrics`):
```python
return {
    'pred_return_sum': float,   # 预测Top5的真实收益和
    'max_return_sum': float,    # 理论最大Top5收益和
    'random_return_sum': float, # 随机选股的期望收益
    'ratio_pred': float,        # pred_return / max_return
    'ratio_random': float,      # random_return / max_return
    'final_score': float,       # (pred - random) / (max - random)
}
```

**产物**:
| 文件 | 内容 |
|---|---|
| `best_model.pth` | 最佳模型权重（按 eval final_score） |
| `scaler.pkl` | StandardScaler 实例 |
| `config.json` | 训练时配置快照 |
| `final_score.txt` | 最佳分数记录 |
| `log/` | TensorBoard 日志 |

**约束**:
- 训练/验证集按时间顺序划分（`split_train_val_by_last_month`）
- 入口使用 `mp.set_start_method('spawn')` 保护多进程
- `drop_small_open=True` 过滤开盘价 < 1e-4 的异常样本

---

### 2.5 `code/src/predict.py` — 推理预测

**职责**: 加载训练好的模型，对未来5个交易日进行预测，输出 stock_id + weight。

**输入**:
- `data/train.csv` — 历史数据（包含到最新交易日）
- `model/{seq}_{feat}/best_model.pth` — 训练好的模型
- `model/{seq}_{feat}/scaler.pkl` — 标准化器

**输出**: `output/result.csv`

| 列名 | 类型 | 约束 |
|---|---|---|
| `stock_id` | str (6位) | 最多5行 |
| `weight` | float | 每行0~1，所有行之和≤1 |

**权重分配**: softmax(score / temperature)，自动归一化。

**约束**:
- 输出格式必须严格匹配（列名 `stock_id`、`weight`）
- ≤5 只股票，权重之和 ∈ [0, 1]，否则评分脚本返回 -999
- 特征工程必须与训练时一致（`feature_num` 相同）

---

## 3. 数据规格

### 3.1 原始数据 (`data/stock_data.csv`)

**必须包含的列**:
```
股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 换手率, 涨跌幅, ...
```
- `股票代码`: str, 6位数字
- `日期`: str, `YYYY-MM-DD` 格式
- 数值列: float

### 3.2 训练数据 (`data/train.csv`)

由 `split_train_test.py` 从 `stock_data.csv` 划分得到。格式与原始数据一致。

### 3.3 测试数据 (`data/test.csv`)

最后5个交易日数据。格式同上。**仅评分时使用，不能用于训练**。

### 3.4 特征集

**39 特征** (`feature_num='39'`):
```
instrument, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅,
sma_5, sma_20, ema_12, ema_26, rsi, macd, macd_signal, volume_change, obv,
volume_ma_5, volume_ma_20, volume_ratio, kdj_k, kdj_d, kdj_j, boll_mid, boll_std,
atr_14, ema_60, volatility_10, volatility_20, return_1, return_5, return_10,
high_low_spread, open_close_spread, high_close_spread, low_close_spread
```

**158+39 特征** (`feature_num='158+39'`):
上述39个 + 158个 Alpha 类特征（ROC, MA, STD, BETA, RSQR, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD，每种 × 5个时间窗口: 5, 10, 20, 30, 60 = 29类 × 5窗口 = 145个，加开盘价特征 KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2, OPEN0, HIGH0, LOW0, VWAP0 共13个）= 158个

---

## 4. Docker 部署规格

### 4.1 执行流程

```
docker compose up
  → data/run.sh 执行:
    1. /bin/bash /app/init.sh      (空)
    2. /bin/bash /app/test.sh      → python code/src/predict.py
  → output/result.csv 生成
```

### 4.2 资源限制

```
GPU: NVIDIA (runtime: nvidia)
CPU: 10核
内存: 16GB
超时: 8小时
```

### 4.3 评分流程

```python
# score_docker.py 逻辑
1. 读取 test/output/result.csv
2. 验证: stock_id/weight 列存在, ≤5只, 权重和∈[0,1]
3. 读取 data/test.csv (真实未来5日数据)
4. 计算: 收益率 = (第5日开盘价 - 第1日开盘价) / 第1日开盘价
5. 计算: Final Score = Σ(权重 × 收益率)
6. 输出: Team Name + Final Score
```

---

## 5. 开发规范

### 5.1 代码修改原则

1. **修改模型结构** → 只能改 `model.py`，保持输入/输出形状不变
2. **修改训练逻辑** → 只能改 `train.py`，保持产物格式不变
3. **修改推理逻辑** → 只能改 `predict.py`，保持 `output/result.csv` 格式不变
4. **新增/修改特征** → 必须同步更新 `train.py` 和 `predict.py` 中的 `feature_cloums_map`
5. **新增配置项** → 只能添加在 `config.py` 中，使用 `config.get('key', default)` 读取

### 5.2 禁止事项

- ❌ 修改 `output/result.csv` 的列名或结构
- ❌ 在 Docker 内依赖网络请求（离线运行）
- ❌ 训练时使用未来的测试数据
- ❌ 修改评分脚本 `score_self.py` / `score_docker.py`
- ❌ 硬编码路径（必须通过 config 读取）

### 5.3 分支策略

```
master          — 稳定版本
feature/xxx     — 功能开发分支
experiment/xxx  — 实验性改动
```

---

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| TA-Lib 安装失败 | 先装 C 库: `wget ta-lib-0.4.0-src.tar.gz && ./configure && make install` |
| 多进程报错 | 必须通过脚本入口运行（`spawn` 模式） |
| 显存不足 (OOM) | 启用 `use_amp=True` + 降低 `batch_size` |
| 预测结果格式不合法 | 检查 stock_id 是否为6位字符串、权重和是否在 [0,1] |
