# 股票排序预测系统 — 软件设计规格说明书 (SDD)

> **版本**: 3.0 | **最后更新**: 2026-07-22 | **适用比赛**: 2026 中国高校计算机大赛—大数据挑战赛
> **基线仓库**: https://github.com/Sherlock1956/THU-BDC2026
> **赛题内容**: https://www.heywhale.com/home/competition/69c0dfa34f302f8f0122e1bb

---

## 0. 比赛要求对标

### 0.1 赛题要求（赛题&数据页面）

| 要求项 | 比赛规定 | 本系统实现 |
|---|---|---|
| **预测对象** | 沪深300指数成分股 | `data/hs300_stock_list.csv` |
| **预测目标** | 未来一周（5个交易日）收益最大的股票组合 | Top5 股票 + 权重 |
| **组合约束** | 不超过5只股票，累计权重 ≤ 1（建议 = 1） | `predict.py` 输出 ≤5 行，weight 求和 = 1 |
| **输出格式** | `result.csv`：`stock_id, weight` | ✅ 严格匹配 |
| **股票代码** | 6位数字（000408, 000975...） | ✅ `zfill(6)` |

### 0.2 评估公式（赛题&数据页面 §四）

```
单只股票收益率: R_i = (P_open^{T+5} - P_open^{T+1}) / P_open^{T+1}

总收益率: R_total = Σ_{i=1}^{n} w_i × R_i
```

- n ≤ 5
- P_open^{T+1}: 第 i 只股票在 T+1 日的开盘价（买入价）
- P_open^{T+5}: 第 i 只股票在 T+5 日的开盘价（卖出价）
- 排名按 R_total 降序排列

### 0.3 提交时间线

| 阶段 | 时间 | 状态 |
|---|---|---|
| A阶段① | 4月25-26日 | ✅ 已过 |
| A阶段② | 5月30-31日 | ✅ 已过 |
| A阶段③ | 6月27-28日 | ✅ 已过 |
| **B阶段** | **8月1日8:00 ~ 8月2日23:59** | 🔴 即将到来 |
| 决赛答辩 | 8月中下旬 | — |

### 0.4 提交规范（提交规范页面）

| 规范项 | 比赛要求 | 本系统实现 |
|---|---|---|
| **基础镜像** | `nvidia/cuda` + Python 3.10+ | ✅ Dockerfile `FROM nvidia/cuda:12.2.0` + Python 3.10 |
| **容器服务名** | docker compose 中服务名必须叫 `client` | ✅ docker-compose.yml |
| **工作目录** | `/app` | ✅ WORKDIR `/app` |
| **数据路径** | 赛事方挂载到 `/app/data` | ✅ volume 映射 `./data:/app/data` |
| **输出路径** | `/app/output/result.csv` | ✅ 映射到 `./test/output/` |
| **网络** | **离线**（训练和推理均不能访问网络） | ✅ 无网络调用 |
| **GPU 支持** | NVIDIA RTX 20xx/30xx/40xx/50xx | ✅ `runtime: nvidia` |
| **训练时间** | **1 小时**（超时强制停止） | ✅ `num_epochs=10` + AMP |
| **执行流程** | `init.sh → train.sh → test.sh` | ✅ `data/run.sh` 链式调用 |
| **依赖框架** | 主用 PyTorch | ✅ |
| **提交方式** | `docker save` 导出 .tar | ✅ |
| **预训练模型** | 允许使用但需声明许可证 | ✅ 未用 |
| **Tushare API** | **禁止**（离线环境） | ✅ 未用 |

---

## 1. 系统概述

### 1.1 目标
基于沪深300成分股的过去60个交易日量价与技术特征，预测未来5个交易日收益最高的股票组合（≤5只，权重之和≤1），追求 `R_total = Σ w_i × (P_T+5 - P_T+1)/P_T+1` 最大化。

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
  │ 评分         │  score_docker.py → Final Score = R_total
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
    ├── /bin/bash /app/train.sh     → python code/src/train.py   ← 1小时超时
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

**关键配置项**:

| 键 | 默认值 | 必须 | 说明 |
|---|---|---|---|
| `sequence_length` | 60 | ✅ | 输入序列长度 |
| `feature_num` | `'158+39'` | ✅ | 特征集标识 |
| `d_model` | 256 | ✅ | Transformer 嵌入维度 |
| `nhead` | 4 | ✅ | 注意力头数 |
| `num_layers` | 3 | ✅ | Transformer 层数 |
| `dim_feedforward` | 512 | ✅ | 前馈网络维度 |
| `batch_size` | 4 | ✅ | 训练批次大小 |
| `num_epochs` | 10 | ✅ | 训练轮数（1小时预算） |
| `learning_rate` | 1e-5 | ✅ | 初始学习率 |
| `dropout` | 0.1 | ✅ | Dropout 比率 |
| `max_grad_norm` | 5.0 | ✅ | 梯度裁剪阈值 |
| `pairwise_weight` | 1 | ✅ | 配对损失权重 |
| `base_weight` | 1.0 | ✅ | 非 top-k 样本权重 |
| `top5_weight` | 2.0 | ✅ | top-5 样本权重 |
| `use_amp` | True | ✅ | 启用混合精度 |
| `predict_temperature` | 1.0 | ✅ | 推理 softmax 温度 |
| `enable_multi_factor` | True | ✅ | 多因子评分开关 |
| `multi_factor_*_weight` | 见 config | ✅ | 因子权重 |
| `output_dir` | 自动生成 | ✅ | 模型产物输出目录 |
| `data_path` | `'./data'` | ✅ | 数据文件目录 |

**约束**:
- `nhead` 必须整除 `d_model`
- `feature_num` 只能是 `'39'` 或 `'158+39'`
- 新增配置项必须对此文件修改

---

### 2.2 `code/src/model.py` — 模型定义

**职责**: 定义 `StockTransformer` 模型。

| 项目 | 规格 |
|---|---|
| **输入形状** | `[batch, num_stocks, seq_len, feature_dim]` |
| **输出形状** | `[batch, num_stocks]` |
| **参数量** | 约 4M |

**子模块序列**:
1. `input_proj` (Linear): `feature_dim → d_model`
2. `PositionalEncoding`: 正弦位置编码
3. `temporal_encoder` (TransformerEncoder): 提取单股票时序模式
4. `FeatureAttention`: 时间维特征加权聚合
5. `CrossStockAttention` (MultiheadAttention): 股票间交互
6. `ranking_layers` (Sequential): `d_model → d_model → d_model/2`
7. `score_head` (Sequential): `d_model/2 → d_model/4 → 1`

**约束**:
- 输入/输出形状不变
- `_init_weights()` 使用 Xavier 初始化

---

### 2.3 `code/src/utils.py` — 特征工程与数据集

**职责**: 提供特征计算函数和排序数据集构建。

**函数接口**:

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `engineer_features_39(df_group)` | `DataFrame` | `DataFrame` | 39个技术指标 |
| `engineer_features_158plus39(df_group)` | `DataFrame` | `DataFrame` | 158 Alpha + 39 技术指标 |
| `create_ranking_dataset_vectorized(data, features, seq_len, ...)` | `DataFrame` | `(sequences, targets, relevance, stock_indices)` | 按日切分排序样本 |

**约束**:
- 特征工程依赖 `TA-Lib` C 库
- 新增特征时，必须同步更新 `feature_cloums_map`

---

### 2.4 `code/src/train.py` — 训练流程

**职责**: 完整的训练 pipeline，从数据加载到模型保存。

**关键函数签名**:

```python
def train_ranking_model(
    model, dataloader, criterion, optimizer, device, epoch, writer,
    scaler: GradScaler | None = None  # AMP
) -> tuple[float, dict]

def evaluate_ranking_model(
    model, dataloader, criterion, device, writer, epoch,
    use_amp: bool = False
) -> tuple[float, dict]
```

**损失函数**: `WeightedRankingLoss` (listwise + pairwise)

**产物**:
| 文件 | 内容 | 是否提交 |
|---|---|---|
| `best_model.pth` | 最佳模型权重 | ✅ Docker 内含 |
| `scaler.pkl` | StandardScaler 实例 | ✅ Docker 内含 |
| `config.json` | 训练时配置快照 | ✅ 复现验证用 |

**约束**:
- 训练/验证集按时间顺序划分
- `mp.set_start_method('spawn')`
- `drop_small_open=True` 过滤开盘价 < 1e-4 的异常样本
- **严禁训练时使用 `data/test.csv`**（未来数据泄漏）
- **整体时间预算 1 小时**（含训练+验证+保存）

---

### 2.5 `code/src/predict.py` — 推理预测

**职责**: 加载训练好的模型，对未来5个交易日进行预测，输出 `stock_id + weight`。

**输入**:
- `data/train.csv` — 历史数据
- `model/{seq}_{feat}/best_model.pth` — 训练好的模型
- `model/{seq}_{feat}/scaler.pkl` — 标准化器

**输出**: `output/result.csv`

| 列名 | 类型 | 约束 | 违反后果 |
|---|---|---|---|
| `stock_id` | str (6位) | 最多5行 | -999 分 |
| `weight` | float | 每行0~1，所有行之和≤1 | -999 分 |

**权重分配**: softmax(score / temperature) + 浮点保护（4舍5入到6位 + 溢出裁剪）

**约束**:
- 输出格式严格匹配（列名 `stock_id`、`weight`）
- ≤5 只股票，权重之和 ∈ [0, 1]（建议 = 1）
- 特征工程必须与训练时一致

---

## 3. 数据规格

### 3.1 原始数据 (`data/stock_data.csv`)

**必须包含的列**:
```
股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅
```

### 3.2 训练数据 (`data/train.csv`)

由 `split_train_test.py` 从 `stock_data.csv` 划分得到。

### 3.3 测试数据 (`data/test.csv`)

最后5个交易日数据。**仅评分时使用**。

### 3.4 特征集

**39 特征** / **158+39 特征**：见 `feature_cloums_map`。

### 3.5 股票代码规范

- 6位数字字符串（如 `000001`）
- `predict.py` 中使用 `.str.zfill(6)` 补齐

---

## 4. Docker 部署规格（赛事规范对齐）

### 4.1 执行流程

```bash
# 容器内
data/run.sh:
  1. /bin/bash /app/init.sh      (环境初始化，当前为空)
  2. /bin/bash /app/train.sh     → python code/src/train.py   ← 1小时超时
  3. /bin/bash /app/test.sh      → python code/src/predict.py
  → /app/output/result.csv 生成
```

### 4.2 Docker 资源限制

```yaml
# docker-compose.yml
runtime: nvidia
services:
  client:                    # ← 比赛要求服务名必须是 client
    image: bdc2026:latest
    command: /bin/bash /app/data/run.sh
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
# 超时: 1 小时（赛事方强制停止）
```

### 4.3 Dockerfile 基础镜像

```dockerfile
FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04
# Python 3.10+
# 离线构建（不允许运行时联网）
```

### 4.4 评分流程

```python
# score_docker.py 逻辑（不可修改）
1. 读取 test/output/result.csv
2. 验证: stock_id/weight 列存在, ≤5只, 权重和 ∈ [0, 1]
   → 任一验证失败 → Final Score = -999
3. 读取 data/test.csv (真实未来5日数据)
4. 计算: R_i = (open_T+5 - open_T+1) / open_T+1
5. 计算: Final Score = Σ w_i × R_i
6. 输出: Team Name + Final Score
```

### 4.5 提交清单

| 提交物 | 说明 |
|---|---|
| `队伍名称.tar` | `docker save -o 队伍名称.tar bdc2026:latest` |
| 提交平台 | Heywhale 平台 / docker 镜像提交入口 |

---

## 5. 开发规范

### 5.1 代码修改原则

1. **修改模型结构** → 只能改 `model.py`，保持输入/输出形状不变
2. **修改训练逻辑** → 只能改 `train.py`，保持产物格式不变
3. **修改推理逻辑** → 只能改 `predict.py`，保持 `output/result.csv` 格式不变
4. **新增/修改特征** → 必须同步更新 `train.py` 和 `predict.py` 中的 `feature_cloums_map`
5. **新增配置项** → 只能添加在 `config.py` 中

### 5.2 禁止事项（比赛硬性约束）

- ❌ 修改 `output/result.csv` 的列名或结构（会导致 -999 分）
- ❌ 在 Docker 内依赖网络请求（**离线环境**）
- ❌ 训练时使用 `data/test.csv`（未来数据泄漏）
- ❌ 修改评分脚本 `score_self.py` / `score_docker.py`
- ❌ 硬编码路径（必须通过 config 读取）
- ❌ 提交不含机器学习算法的方案（会被取消资格）
- ❌ 训练总时间超过 1 小时（被强制停止）
- ❌ 使用 Tushare / Tushare Pro 在线数据 API

### 5.3 分支策略

```
master          — 稳定版本
feature/xxx     — 功能开发分支
experiment/xxx  — 实验性改动
```

### 5.4 超越基准程序的改进方向

| 方向 | 改动文件 | 是否需要重训 | 预期收益 |
|---|---|---|---|
| **权重不等权分配** | predict.py | ❌ 不需要 | ⭐⭐⭐⭐ |
| **多因子评分融合** | predict.py + config.py | ❌ 不需要 | ⭐⭐⭐⭐ |
| **行业分散约束** | predict.py | ❌ 不需要 | ⭐⭐⭐ |
| **扩大候选池+二次筛选** | predict.py | ❌ 不需要 | ⭐⭐ |
| **波动率风控** | predict.py | ❌ 不需要 | ⭐ |
| **新增特征** | utils.py + train.py + predict.py | ✅ 需要 | ⭐⭐⭐ |
| **模型架构改进** | model.py + train.py | ✅ 需要 | ⭐⭐⭐⭐ |
| **损失函数优化** | train.py | ✅ 需要 | ⭐⭐⭐ |
| **集成学习** | train.py + predict.py | ✅ 需要 | ⭐⭐⭐⭐ |
| **超参数搜索** | config.py + train.py | ✅ 需要 | ⭐⭐ |

### 5.5 可复现性要求

- `train.py` 中使用 `set_seed(42)` 固定随机种子
- `config.json` 保存训练时配置快照
- Docker 镜像必须包含完整运行环境
- **整体时间预算 1 小时**（含训练+推理+启动）

---

## 6. 常见问题

| 问题 | 解决 |
|---|---|
| TA-Lib 安装失败 | Dockerfile 内 wget + 编译安装 |
| 多进程报错 | 必须通过脚本入口运行（`spawn` 模式） |
| 显存不足 (OOM) | 启用 `use_amp=True` + 降低 `batch_size` |
| 预测结果格式不合法 | 检查 stock_id 6位、权重和 ≤ 1 |
| Docker 内训练超时 | 降低 `num_epochs` 或减小模型规模 |
| 评分返回 -999 | 列名必须是 `stock_id` 和 `weight`（不能是中文） |
| 训练超过 1 小时 | 默认 `num_epochs=10`，根据 GPU 性能微调 |
| 浮点精度导致 -999 | predict.py 已加 `np.round + 溢出裁剪` 保护 |

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
| `get_stock_data.py` | 数据下载 | ⚠️ 仅本地使用，不进入 Docker |
| `data/split_train_test.py` | 数据划分 | ⚠️ 仅本地使用 |
| `Dockerfile` | Docker 打包 | ⚠️ 需符合 nvidia/cuda 规范 |
| `docker-compose.yml` | Docker 编排 | ⚠️ 服务名必须叫 `client` |
| `data/run.sh` | 容器执行入口 | ❌ 必须包含 train.sh + test.sh |
| `train.sh` / `test.sh` | 训练/推理脚本 | ❌ |
| `test/score_self.py` | 本地自评 | ❌ 不可修改 |
| `test/score_docker.py` | 赛事方评分 | ❌ 不可修改 |

### 7.2 官方资源

- **比赛平台**: https://www.heywhale.com/u/2026BDC
- **大赛官网**: https://nercbds.tsinghua.edu.cn/bdc.html
- **基准程序**: https://github.com/Sherlock1956/THU-BDC2026
- **大赛邮箱**: data@tsinghua.edu.cn

### 7.3 关键规范差异 v2.0 → v3.0

| 项 | v2.0（错） | v3.0（对）|
|---|---|---|
| 训练时间 | 8小时 | **1小时** |
| 容器服务名 | app | **client** |
| 基础镜像 | python:3.12 | **nvidia/cuda** + Python 3.10 |
| run.sh 训练 | 注释掉 | **必须执行** |
| num_epochs | 50 | **10** |
