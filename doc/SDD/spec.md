# 股票排序预测系统 — 软件设计规格说明书 (SDD)

> **版本**: 3.0 | **最后更新**: 2026-07-23 | **适用比赛**: 2026 中国高校计算机大赛—大数据挑战赛
> **基线仓库**: https://github.com/Sherlock1956/THU-BDC2026
> **赛题内容**: https://www.heywhale.com/home/competition/69c0dfa34f302f8f0122e1bb

---

## 0. 比赛要求对标（来源：赛题&数据.md + 通知公告.md）

### 0.1 赛题要求

| 要求项 | 比赛规定 | 本系统实现 |
|---|---|---|
| **预测对象** | 沪深300指数成分股 | `data/hs300_stock_list.csv` |
| **预测目标** | 未来一周（5个交易日）收益最大的股票组合 | Top5 股票 + 权重 |
| **组合约束** | 不超过5只股票，累计权重 ≤ 1（不到1的部分持有现金） | `predict.py` 输出 ≤5 行，weight 求和 ≤ 1 |
| **输出格式** | `result.csv`（UTF-8编码）：`stock_id,weight` | ✅ 严格匹配 |
| **数据来源** | 不限，只要免费公开可下载 | baostock + 选手自有数据 |
| **预训练模型** | 允许（需满足开源+发布时间条件） | ✅ 未使用 |
| **可复现性+创新性** | 参考指标 | ✅ StockTransformer 排序学习 |
| **成绩门槛** | 总收益率须大于基准程序才能参与排名 | 改进点见 §5.4 |

### 0.2 评估公式

```
单只股票收益率: R_i = (P_open^{T+5} - P_open^{T+1}) / P_open^{T+1}

总投资组合收益率: R_total = Σ_{i=1}^{n} w_i × R_i
```

- n ≤ 5
- 现金部分（权重未用完部分）收益率为0
- 排名规则：
  1) R_total 须大于基准程序才能参与排名
  2) 按 R_total 从大到小排序

### 0.3 提交时间线

| 阶段 | 时间 | 状态 |
|---|---|---|
| A阶段① | 4月25-26日 | ✅ 已过 |
| A阶段② | 5月30-31日 | ✅ 已过 |
| A阶段③ | 6月27-28日 | ✅ 已过 |
| **B阶段** | **8月1日8:00 ~ 8月2日23:59** | 🔴 即将到来 |
| 决赛答辩 | 8月中下旬 | — |

### 0.4 Docker 提交规范（来源：通知公告.md）

| 规范项 | 比赛要求 | 本系统实现 |
|---|---|---|
| **提交方式** | 先提交 result.csv，排名靠前再提交 docker（B阶段选 120% 名额） | ✅ |
| **上传入口** | Heywhale 平台专属上传入口（不再用网盘） | ✅ |
| **复现要求** | Docker 运行结果必须与提交的 result.csv 完全一致 | ✅ set_seed + config.json |
| **代码审查** | 最终验证阶段会审查代码，确认结果由模型预测产生 | ✅ 无硬编码 |
| **禁止硬编码** | 不能将固定股票代码/权重写入代码绕过模型 | ✅ |

### 0.5 数据目录挂载规则（来源：通知公告.md，重要！）

| 规范项 | 比赛要求 | 本系统实现 |
|---|---|---|
| **data 目录** | 外部挂载目录，**赛事方数据会覆盖选手 data 内容** | ✅ |
| **赛事方提供** | `data/stock_data.csv`（截至提交最后一天往前3年数据） | ✅ |
| **赛事方不提供** | ❌ `train.csv`、`test.csv` | ✅ 代码已改为读 stock_data.csv |
| **自定义数据存放** | 必须放在 `model/` 等非挂载目录（不能放 data/output/temp） | ✅ |
| **选手需自行划分** | 代码中需完成数据划分（训练集/验证集等） | ✅ `split_train_val_by_last_month` 内置 |

---

## 1. 系统概述

### 1.1 目标
基于沪深300成分股的过去60个交易日量价与技术特征，预测未来5个交易日收益最高的股票组合（≤5只，权重之和≤1），追求 `R_total = Σ w_i × (P_T+5 - P_T+1)/P_T+1` 最大化。

### 1.2 整体数据流

```
[data/stock_data.csv（赛事方挂载）]
       │
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
    ├── /bin/bash /app/train.sh     → python code/src/train.py
    │                                   └─ 读取 data/stock_data.csv（赛事方挂载）
    └── /bin/bash /app/test.sh      → python code/src/predict.py
                                        └─ 读取 data/stock_data.csv
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
| `num_epochs` | 50 | ✅ | 训练轮数 |
| `learning_rate` | 1e-5 | ✅ | 初始学习率 |
| `dropout` | 0.1 | ✅ | Dropout 比率 |
| `max_grad_norm` | 5.0 | ✅ | 梯度裁剪阈值 |
| `use_amp` | True | ✅ | 启用混合精度 |
| `predict_temperature` | 1.0 | ✅ | 推理 softmax 温度 |
| `enable_multi_factor` | True | ✅ | 多因子评分开关 |
| `multi_factor_*_weight` | 见 config | ✅ | 因子权重 |
| `output_dir` | 自动生成 | ✅ | 模型产物输出目录 |
| `data_path` | `'./data'` | ✅ | 数据文件目录（赛事方挂载） |

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

**数据加载策略**:
```python
# 优先读 stock_data.csv（赛事方挂载），本地开发时 fallback 到 train.csv
data_file = os.path.join(data_path, 'stock_data.csv')
if not os.path.exists(data_file):
    data_file = os.path.join(data_path, 'train.csv')
```

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

def split_train_val_by_last_month(df, sequence_length)  # 内置数据划分
```

**损失函数**: `WeightedRankingLoss` (listwise + pairwise)

**产物**:
| 文件 | 内容 | 是否提交 |
|---|---|---|
| `best_model.pth` | 最佳模型权重 | ✅ Docker 内含 |
| `scaler.pkl` | StandardScaler 实例 | ✅ Docker 内含 |
| `config.json` | 训练时配置快照 | ✅ 复现验证用 |

**约束**:
- 训练/验证集按时间顺序划分（`split_train_val_by_last_month`）
- `mp.set_start_method('spawn')`
- `drop_small_open=True` 过滤开盘价 < 1e-4 的异常样本
- **严禁训练时使用 `data/test.csv`**（未来数据泄漏）
- **严禁硬编码股票代码/权重**（通知公告.md 规则）

---

### 2.5 `code/src/predict.py` — 推理预测

**职责**: 加载训练好的模型，对未来5个交易日进行预测，输出 `stock_id + weight`。

**数据加载策略**:
```python
# 优先读 stock_data.csv（赛事方挂载），本地开发时 fallback 到 train.csv
data_file = os.path.join(config['data_path'], 'stock_data.csv')
if not os.path.exists(data_file):
    data_file = os.path.join(config['data_path'], 'train.csv')
```

**输入**:
- `data/stock_data.csv` — 历史数据（赛事方挂载）
- `model/{seq}_{feat}/best_model.pth` — 训练好的模型
- `model/{seq}_{feat}/scaler.pkl` — 标准化器

**输出**: `output/result.csv`

| 列名 | 类型 | 约束 | 违反后果 |
|---|---|---|---|
| `stock_id` | str (6位) | 最多5行 | -999 分 |
| `weight` | float | 每行0~1，所有行之和≤1 | -999 分 |

**权重分配**: softmax(score / temperature) + 浮点精度保护（四舍五入到6位 + 溢出裁剪）

**约束**:
- 输出格式必须严格匹配（列名 `stock_id`、`weight`，**不能写成中文**）
- ≤5 只股票，权重之和 ∈ [0, 1]
- 特征工程必须与训练时一致
- **严禁硬编码股票代码/权重**（通知公告.md 规则）

---

## 3. 数据规格

### 3.1 赛事方数据 (`data/stock_data.csv`)

赛事方在最终验证阶段挂载，包含：
- 截至提交最后一天往前 3 年的数据
- 沪深300成分股历史股价
- **不提供 train.csv 和 test.csv**

**必须包含的列**:
```
股票代码, 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌额, 换手率, 涨跌幅
```
- `股票代码`: str, 6位数字
- `日期`: str, `YYYY-MM-DD` 格式
- 数值列: float

### 3.2 本地开发数据

`split_train_test.py` 从 `stock_data.csv` 划分得到 `train.csv` 和 `test.csv`，仅供本地测试，**不进入 Docker**。

### 3.3 测试数据 (`data/test.csv`)

仅本地评分时使用（`score_self.py`）。**赛事方不提供此文件**。

### 3.4 特征集

**39 特征** / **158+39 特征**：见 `feature_cloums_map`。

### 3.5 股票代码规范

- 6位数字字符串（如 `000001`）
- `predict.py` 中使用 `.str.zfill(6)` 补齐

### 3.6 目录挂载规则（重要！）

| 目录 | 是否被赛事方挂载 | 可存放内容 |
|---|---|---|
| `data/` | ✅ **会被覆盖** | 仅放赛事方期望的 stock_data.csv |
| `output/` | ✅ **会被覆盖** | result.csv 输出 |
| `temp/` | ✅ **会被覆盖** | 临时文件 |
| `model/` | ❌ 不挂载 | 训练权重、自定义数据 |
| `code/` | ❌ 不挂载 | 源代码 |

> **⚠️ 严禁将自定义数据放在 data/output/temp 目录内**

---

## 4. Docker 部署规格

### 4.1 执行流程

```
docker compose up
  → data/run.sh 执行:
    1. /bin/bash /app/init.sh      (环境初始化，当前为空)
    2. /bin/bash /app/train.sh     → python code/src/train.py
    │   └─ 读取 data/stock_data.csv（赛事方挂载）
    3. /bin/bash /app/test.sh      → python code/src/predict.py
        └─ 读取 data/stock_data.csv
  → output/result.csv 生成
```

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
```

### 4.3 评分流程

```python
# score_docker.py 逻辑（不可修改）
1. 读取 test/output/result.csv
2. 验证: stock_id/weight 列存在, ≤5只, 权重和 ∈ [0, 1]
   → 任一验证失败 → Final Score = -999
3. 读取 data/test.csv (真实未来5日数据)
4. 计算: R_i = (P_T+5 - P_T+1) / P_T+1
5. 计算: Final Score = Σ w_i × R_i
6. 输出: Team Name + Final Score
```

### 4.4 提交流程（来源：通知公告.md）

| 步骤 | 说明 |
|---|---|
| 1. 提交 result.csv | 在 Heywhale 平台提交结果文件 |
| 2. 排名筛选 | B阶段选 120% 名额进入 docker 提交 |
| 3. Docker 上传 | 通过平台专属上传入口（不再用网盘） |
| 4. 代码审查 | 最终验证阶段审查代码是否由模型预测产生 |

---

## 5. 开发规范

### 5.1 代码修改原则

1. **修改模型结构** → 只能改 `model.py`，保持输入/输出形状不变
2. **修改训练逻辑** → 只能改 `train.py`，保持产物格式不变
3. **修改推理逻辑** → 只能改 `predict.py`，保持 `output/result.csv` 格式不变
4. **新增/修改特征** → 必须同步更新 `train.py` 和 `predict.py` 中的 `feature_cloums_map`
5. **新增配置项** → 只能添加在 `config.py` 中，使用 `config.get('key', default)` 读取

### 5.2 禁止事项（比赛硬性约束）

- ❌ 修改 `output/result.csv` 的列名或结构（会导致 -999 分）
- ❌ 在 Docker 内依赖网络请求（离线运行）
- ❌ 训练时使用 `data/test.csv`（未来数据泄漏）
- ❌ 修改评分脚本 `score_self.py` / `score_docker.py`
- ❌ 硬编码路径（必须通过 config 读取）
- ❌ 提交不含机器学习算法的方案（会被取消资格）
- ❌ **硬编码股票代码/权重**绕过模型预测（通知公告.md 明令禁止）
- ❌ **将自定义数据放在 data/output/temp 目录**（会被赛事方挂载覆盖）
- ❌ **假设 data/train.csv 存在**（赛事方只提供 stock_data.csv）

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
- **Docker 运行结果必须与提交的 result.csv 完全一致**（通知公告.md 重申）

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
| 浮点精度导致 -999 | predict.py 已加 `np.round + 溢出裁剪` 保护 |
| 赛事方 data 目录覆盖 | 自定义数据放 `model/` 目录 |
| train.csv 不存在 | 代码已 fallback 到 stock_data.csv |

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
| `Dockerfile` | Docker 打包 | ⚠️ 确认依赖 |
| `docker-compose.yml` | Docker 编排 | ❌ 不建议改 |
| `data/run.sh` | 容器执行入口 | ❌ 不建议改 |
| `train.sh` / `test.sh` | 训练/推理脚本 | ❌ |
| `test/score_self.py` | 本地自评 | ❌ 不可修改 |
| `test/score_docker.py` | 赛事方评分 | ❌ 不可修改 |

### 7.2 官方资源

- **比赛平台**: https://www.heywhale.com/u/2026BDC
- **大赛官网**: https://nercbds.tsinghua.edu.cn/bdc.html
- **基准程序**: https://github.com/Sherlock1956/THU-BDC2026
- **大赛邮箱**: data@tsinghua.edu.cn
- **Tushare 数据平台**: tushare.pro（大赛提供免费权限，有效期至 2026-08-31）

### 7.3 v2.1 → v3.0 关键变更

| 项 | v2.1 | v3.0 |
|---|---|---|
| 数据来源假设 | 假设 data/train.csv 存在 | **改为读 stock_data.csv + fallback** |
| 目录挂载规则 | 未说明 | **新增 §3.6 挂载规则表** |
| 提交流程 | 网盘 | **先 result.csv 排名 → 再 docker 上传** |
| 硬编码禁止 | 未提及 | **通知公告.md 明令禁止** |
