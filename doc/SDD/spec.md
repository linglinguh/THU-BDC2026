# 股票排序预测系统 — 软件设计规格说明书 (SDD)

> **版本**: 2.0 | **最后更新**: 2026-07-22 | **适用比赛**: 2026 中国高校计算机大赛—大数据挑战赛
> **基线仓库**: https://github.com/Sherlock1956/THU-BDC2026

---

## 0. 比赛要求对标

### 0.1 赛题要求（来源：竞赛通知 + 赛题描述）

| 要求项 | 比赛规定 | 本系统实现 |
|---|---|---|
| **预测对象** | 沪深300指数成分股 | `data/hs300_stock_list.csv` |
| **预测目标** | 未来一周（5个交易日）收益最大的股票组合 | Top5 股票 + 权重 |
| **组合约束** | 不超过5只股票，累计权重 ≤ 1 | `predict.py` 输出 ≤5 行，weight 求和 ≤ 1 |
| **输出格式** | `result.csv`：`stock_id, weight` | ✅ 严格匹配 |
| **提交方式** | Docker 镜像 (.tar) + result.csv | ✅ Dockerfile + docker-compose.yml |
| **可复现性** | 组委会用 Docker 重跑 train+predict 验证 | ✅ `data/run.sh` 链式调用 |
| **成绩门槛** | 必须优于基准程序 | 改进点见 §5.4 |
| **算法要求** | 必须有机器学习算法贡献 | ✅ StockTransformer 排序学习 |

### 0.2 评分公式（来源：`test/score_docker.py`）

```
收益率_i = (test数据第5日开盘价 - test数据第1日开盘价) / test数据第1日开盘价

Final Score = Σ_i (weight_i × 收益率_i)
```

- 预测格式不合法（列名错误/股票数>5/权重和>1）→ **-999 分**
- 阶段排名按 Final Score 降序排列

### 0.3 提交时间线

| 阶段 | 时间 | 状态 |
|---|---|---|
| A阶段① | 4月25-26日 | ✅ 已过 |
| A阶段② | 5月30-31日 | ✅ 已过 |
| A阶段③ | 6月27-28日 | ✅ 已过 |
| **B阶段** | **8月1日8:00 ~ 8月2日23:59** | 🔴 即将到来 |
| 决赛答辩 | 8月中下旬 | — |

---

## 1. 系统概述

### 1.1 目标
基于沪深300成分股的过去60个交易日量价与技术特征，预测未来5个交易日收益最高的股票组合（≤5只，权重之和≤1），追求 `Final Score = Σ(权重 × 收益率)` 最大化。

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
  │ 评分         │  score_self.py / score_docker.py → Final Score
  └─────────────┘
```

### 1.3 Docker 执行链路（赛事方评测流程）

```
选手提交 .tar 镜像
    │
    ▼ 赛事方执行
docker load → docker compose up
    │
    ▼ 容器内执行 data/run.sh:
    ├── /bin/bash /app/init.sh      (环境初始化，当前为空)
    ├── /bin/bash /app/train.sh     → python code/src/train.py
    └── /bin/bash /app/test.sh      → python code/src/predict.py
    │
    ▼ 赛事方评分
score_docker.py 读取 test/output/result.csv + data/test.csv
    │
    ▼
Final Score = Σ(weight_i × 收益率_i)
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
    'final_score': float,       # (pred - random) / (max - random) — 训练监控指标
}
```

> **注意**: `final_score` 是训练时的代理指标（排序能力衡量），**不是**比赛最终得分。比赛得分 = `Σ(weight × 真实5日收益率)`，见 §0.2。

**产物**:
| 文件 | 内容 | 是否提交 |
|---|---|---|
| `best_model.pth` | 最佳模型权重（按 eval final_score） | ✅ Docker 内含 |
| `scaler.pkl` | StandardScaler 实例 | ✅ Docker 内含 |
| `config.json` | 训练时配置快照 | ✅ 复现验证用 |
| `final_score.txt` | 训练分数记录 | 仅供参考 |
| `log/` | TensorBoard 日志 | 不提交（.gitignore） |

**约束**:
- 训练/验证集按时间顺序划分（`split_train_val_by_last_month`）
- 入口使用 `mp.set_start_method('spawn')` 保护多进程
- `drop_small_open=True` 过滤开盘价 < 1e-4 的异常样本
- **严禁训练时使用 `data/test.csv`**（未来数据泄漏）

---

### 2.5 `code/src/predict.py` — 推理预测

**职责**: 加载训练好的模型，对未来5个交易日进行预测，输出 `stock_id + weight`。

**输入**:
- `data/train.csv` — 历史数据（包含到最新交易日）
- `model/{seq}_{feat}/best_model.pth` — 训练好的模型
- `model/{seq}_{feat}/scaler.pkl` — 标准化器

**输出**: `output/result.csv`

| 列名 | 类型 | 约束 | 违反后果 |
|---|---|---|---|
| `stock_id` | str (6位) | 最多5行 | -999 分 |
| `weight` | float | 每行0~1，所有行之和≤1 | -999 分 |

**权重分配**: softmax(score / temperature)，自动归一化，和为1。

**约束**:
- 输出格式必须严格匹配（列名 `stock_id`、`weight`，**不能写成中文**）
- ≤5 只股票，权重之和 ∈ [0, 1]，否则评分脚本返回 -999
- 特征工程必须与训练时一致（`feature_num` 相同）
- 输出路径固定为 `output/result.csv`（Docker 通过 volume 映射到 `test/output/`）

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
- 评分脚本只使用 `股票代码`、`日期`、`开盘` 三列计算收益率

### 3.2 训练数据 (`data/train.csv`)

由 `split_train_test.py` 从 `stock_data.csv` 划分得到。格式与原始数据一致。

### 3.3 测试数据 (`data/test.csv`)

最后5个交易日数据。格式同上。

> **⚠️ 严禁用于训练**。评分时 `score_docker.py` 会读取此文件计算真实收益率。

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

### 3.5 股票代码规范

- 6位数字字符串（如 `000001`）
- `predict.py` 中使用 `.str.zfill(6)` 补齐
- 评分时必须与 `data/test.csv` 中的 `股票代码` 完全匹配

---

## 4. Docker 部署规格

### 4.1 执行流程

```
docker compose up
  → data/run.sh 执行:
    1. /bin/bash /app/init.sh      (环境初始化，当前为空)
    2. /bin/bash /app/train.sh     → python code/src/train.py
    3. /bin/bash /app/test.sh      → python code/src/predict.py
  → output/result.csv 生成
```

> **注意**: 赛事方评测时会**完整执行训练+推理**，不能只提交预训练权重。

### 4.2 资源限制

```yaml
# docker-compose.yml
runtime: nvidia
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
mem_limit: 16g
cpus: 10.0
# 超时: 8小时5分钟（超时强制终止）
```

### 4.3 评分流程

```python
# score_docker.py 逻辑（不可修改）
1. 读取 test/output/result.csv          # 选手的预测
2. 验证: stock_id/weight 列存在
3. 验证: ≤5只股票
4. 验证: 权重和 ∈ [0, 1]
   → 任一验证失败 → Final Score = -999
5. 读取 data/test.csv (真实未来5日数据)
6. 计算: 收益率_i = (第5日开盘价 - 第1日开盘价) / 第1日开盘价
7. 计算: Final Score = Σ_i (weight_i × 收益率_i)
8. 输出: Team Name + Final Score
```

### 4.4 提交清单

| 提交物 | 说明 |
|---|---|
| `队伍名称.tar` | Docker 镜像导出文件，`docker save -o 队伍名称.tar bdc2026:latest` |
| `result.csv` | 最终预测结果（Docker 运行后自动生成） |

**提交方式**: 上传 .tar 到夸克网盘，生成永久有效分享链接（不加提取码），在 Heywhale 平台提交。

---

## 5. 开发规范

### 5.1 代码修改原则

1. **修改模型结构** → 只能改 `model.py`，保持输入/输出形状不变
2. **修改训练逻辑** → 只能改 `train.py`，保持产物格式不变
3. **修改推理逻辑** → 只能改 `predict.py`，保持 `output/result.csv` 格式不变
4. **新增/修改特征** → 必须同步更新 `train.py` 和 `predict.py` 中的 `feature_cloums_map`
5. **新增配置项** → 只能添加在 `config.py` 中，使用 `config.get('key', default)` 读取

### 5.2 禁止事项

- ❌ 修改 `output/result.csv` 的列名或结构（会导致 -999 分）
- ❌ 在 Docker 内依赖网络请求（离线运行）
- ❌ 训练时使用 `data/test.csv`（未来数据泄漏）
- ❌ 修改评分脚本 `score_self.py` / `score_docker.py`
- ❌ 硬编码路径（必须通过 config 读取）
- ❌ 提交不含机器学习算法的方案（会被取消资格）

### 5.3 分支策略

```
master          — 稳定版本（可直接提交比赛）
feature/xxx     — 功能开发分支（如 feature/industry-diversification）
experiment/xxx  — 实验性改动（如 experiment/different-loss）
```

### 5.4 超越基准程序的改进方向

| 方向 | 改动文件 | 是否需要重训 | 预期收益 |
|---|---|---|---|
| **权重不等权分配** | predict.py | ❌ 不需要 | ⭐⭐⭐⭐ |
| **行业分散约束** | predict.py | ❌ 不需要 | ⭐⭐⭐ |
| **扩大候选池+二次筛选** | predict.py | ❌ 不需要 | ⭐⭐ |
| **波动率风控** | predict.py | ❌ 不需要 | ⭐ |
| **新增特征** | utils.py + train.py + predict.py | ✅ 需要 | ⭐⭐⭐ |
| **模型架构改进** | model.py + train.py | ✅ 需要 | ⭐⭐⭐⭐ |
| **损失函数优化** | train.py | ✅ 需要 | ⭐⭐⭐ |
| **集成学习** | train.py + predict.py | ✅ 需要 | ⭐⭐⭐⭐ |
| **超参数搜索** | config.py + train.py | ✅ 需要 | ⭐⭐ |

### 5.5 可复现性要求

比赛要求 Docker 重跑结果与提交结果一致。因此：
- `train.py` 中使用了 `set_seed(42)` 固定随机种子
- `config.json` 保存训练时配置快照
- **修改代码后必须重新打包 Docker 镜像并验证**

---

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| TA-Lib 安装失败 | 先装 C 库: `wget ta-lib-0.4.0-src.tar.gz && ./configure && make install` |
| 多进程报错 | 必须通过脚本入口运行（`spawn` 模式） |
| 显存不足 (OOM) | 启用 `use_amp=True` + 降低 `batch_size` |
| 预测结果格式不合法 | 检查 stock_id 是否为6位字符串、权重和是否在 [0,1] |
| Docker 内训练超时 | 降低 `num_epochs` 或减小模型规模 |
| 评分返回 -999 | 检查 result.csv 列名是否为 `stock_id` 和 `weight`（不能是中文） |
| 成绩未超越基准 | 优先改进权重分配策略（无需重训） |

---

## 7. 附录

### 7.1 关键文件清单

| 文件 | 职责 | 可修改 |
|---|---|---|
| `code/src/config.py` | 全局配置 | ✅ |
| `code/src/model.py` | 模型结构 | ✅ |
| `code/src/train.py` | 训练流程 | ✅ |
| `code/src/predict.py` | 推理预测 | ✅ |
| `code/src/utils.py` | 特征工程 | ✅ |
| `get_stock_data.py` | 数据下载 | ⚠️ 改日期参数 |
| `data/split_train_test.py` | 数据划分 | ⚠️ 改日期参数 |
| `Dockerfile` | Docker 打包 | ⚠️ 确认依赖 |
| `docker-compose.yml` | Docker 编排 | ❌ 不建议改 |
| `data/run.sh` | 容器执行入口 | ❌ 不建议改 |
| `train.sh` / `test.sh` | 训练/推理脚本 | ❌ 不建议改 |
| `test/score_self.py` | 本地自评 | ❌ 不可修改 |
| `test/score_docker.py` | 赛事方评分 | ❌ 不可修改 |

### 7.2 官方资源

- **比赛平台**: https://www.heywhale.com/u/2026BDC
- **大赛官网**: https://nercbds.tsinghua.edu.cn/bdc.html
- **基准程序**: https://github.com/Sherlock1956/THU-BDC2026
- **大赛邮箱**: data@tsinghua.edu.cn
