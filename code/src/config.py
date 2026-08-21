import os

# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'seed': int(os.getenv('BDC_SEED', '42')),
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    # 8GB Apple Silicon 可通过 BDC_BATCH_SIZE=1 避免统一内存换页；正式默认仍为 4。
    'batch_size': int(os.getenv('BDC_BATCH_SIZE', '4')),
    # 50 epoch 已证实严重过拟合；默认最多15轮，并使用验证组合收益早停。
    'num_epochs': int(os.getenv('BDC_NUM_EPOCHS', '15')),
    'early_stopping_patience': int(os.getenv('BDC_EARLY_STOPPING_PATIENCE', '3')),
    'early_stopping_min_delta': float(os.getenv('BDC_EARLY_STOPPING_MIN_DELTA', '0.0001')),
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # --- MASTER 架构 (AAAI 2024) 三大核心创新 ---
    # 启用后使用 Intra-Day Attention + Inter-Day Attention + Market-Guided Gating 交替层
    # 关闭则回退到原始 StockTransformer 架构（向后兼容旧权重）
    'enable_master': True,
    # MASTER 交替层数：每层包含 日内注意力→日间注意力→市场门控
    # 4GB 显卡建议 1 层；显存充裕可调到 2~3
    'master_num_layers': 1,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': os.getenv('BDC_OUTPUT_DIR', f'./model/{sequence_length}_{feature_num}'),
    'data_path': './data',
    # 滚动验证可指定历史截面；空字符串表示使用数据中的最新日期。
    'as_of_date': os.getenv('BDC_AS_OF_DATE', ''),
    'validation_months': int(os.getenv('BDC_VALIDATION_MONTHS', '2')),

    # 混合精度训练 (AMP)，显存减半，4GB显卡也能跑batch=2~4
    'use_amp': True,

    # 三个非重叠 CUDA 滚动窗口的三模型集成结果显示，近等权配置
    # 比集中持仓具有更高的跨窗口均值和更小的最差窗口回撤。
    'predict_temperature': 100.0,

    # --- 推理后处理策略 ---
    # 候选池大小：从模型排名 Top-N 中做二次筛选，再取最终 Top5
    'candidate_pool_size': 30,

    # 多因子评分模式（启用后替代简单筛选，对候选池各因子 z-score 后加权融合）
    'enable_multi_factor': True,
    # 模型分数在综合评分中的权重（因子只做微调，ML 主导决策）
    'multi_factor_model_weight': 0.80,
    'multi_factor_momentum_weight': 0.10,   # 近5日动量
    'multi_factor_reversal_weight': 0.00,   # 反转（与动量重叠，关闭）
    'multi_factor_volatility_weight': 0.05, # 波动率（逆向，低波更优）
    'multi_factor_liquidity_weight': 0.02,  # 换手率（中性，极端值扣分）
    'multi_factor_volume_weight': 0.03,     # 成交量比（中性，极端值扣分）

    # 单因子筛选模式（enable_multi_factor=False 时生效，向后兼容）
    'enable_momentum_filter': True,
    'momentum_lookback_days': 5,
    'enable_volatility_filter': True,
    'volatility_lookback_days': 10,
    'volatility_max_threshold': 0.15,

    # 行业分散（避免 Top5 集中在同一行业，降低组合风险）
    'enable_industry_diversify': True,
    'industry_max_per_sector': 1,       # 同一行业最多入选 1 只
}
