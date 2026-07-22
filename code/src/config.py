# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    'learning_rate': 1e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',

    # 混合精度训练 (AMP)，显存减半，4GB显卡也能跑batch=2~4
    'use_amp': True,

    # 推理时权重分配的 softmax 温度：越小越集中（接近 max=1），越大越均匀（趋近等权0.2）
    'predict_temperature': 1.0,

    # --- 推理后处理策略 ---
    # 候选池大小：从模型排名 Top-N 中做二次筛选，再取最终 Top5
    'candidate_pool_size': 15,
    # 动量筛选：剔除近期下跌的股票（近 N 日累计涨跌幅 < 0）
    'enable_momentum_filter': True,
    'momentum_lookback_days': 5,
    # 波动率风控：剔除近期波动过大的股票（日均振幅超阈值）
    'enable_volatility_filter': True,
    'volatility_lookback_days': 10,
    'volatility_max_threshold': 0.15,  # 日均振幅上限 15%
}