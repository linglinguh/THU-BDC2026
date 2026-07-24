import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
import joblib
import os
import json
import multiprocessing as mp
import random
import contextlib
from strategy import select_portfolio

# 混合精度训练 (AMP)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

feature_cloums_map = {
    '39': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv','volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'],

    '158+39': ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅','KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0', 'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5', 'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10', 'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20', 'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30', 'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30', 'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60', 'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60', 'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60', 'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60', 'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60', 'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60', 'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5', 'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5', 'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60','sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv', 'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std', 'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',  'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread']
}
feature_engineer_func_map = {
    '39': engineer_features_39,
    '158+39': engineer_features_158plus39
}

STRATEGY_CONFIG_KEYS = (
    'predict_temperature',
    'candidate_pool_size',
    'enable_multi_factor',
    'multi_factor_model_weight',
    'multi_factor_momentum_weight',
    'multi_factor_reversal_weight',
    'multi_factor_volatility_weight',
    'multi_factor_liquidity_weight',
    'multi_factor_volume_weight',
    'enable_industry_diversify',
    'industry_max_per_sector',
)


def changed_strategy_config(checkpoint_config, current_config):
    """Return strategy keys that make a saved validation score incomparable."""
    checkpoint_config = checkpoint_config or {}
    return {
        key: (checkpoint_config.get(key), current_config.get(key))
        for key in STRATEGY_CONFIG_KEYS
        if checkpoint_config.get(key) != current_config.get(key)
    }


def _build_label_and_clean(processed, drop_small_open=True):
    """Build T+1-to-T+5 labels on the shared exchange trading calendar.

    Rows without a valid label are retained because they can still be used as
    causal history by a later prediction window.
    """
    processed = processed.sort_values(['股票代码', '日期']).copy()
    processed['_date'] = pd.to_datetime(processed['日期'])
    calendar = pd.Index(sorted(processed['_date'].unique()))
    date_to_position = pd.Series(np.arange(len(calendar)), index=calendar)
    processed['_date_position'] = processed['_date'].map(date_to_position).astype(np.int64)

    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
    processed['_position_t1'] = processed.groupby('股票代码')['_date_position'].shift(-1)
    processed['_position_t5'] = processed.groupby('股票代码')['_date_position'].shift(-5)

    valid_label = (
        (processed['_position_t1'] == processed['_date_position'] + 1)
        & (processed['_position_t5'] == processed['_date_position'] + 5)
    )

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        valid_label &= processed['open_t1'] > 1e-4

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed.loc[~valid_label, 'label'] = np.nan

    processed.drop(
        columns=['open_t1', 'open_t5', '_date', '_date_position', '_position_t1', '_position_t5'],
        inplace=True,
    )
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失 (KL散度 + Cross Entropy)"""
        
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)

        # 加权 Cross Entropy（原实现未使用 weights）
        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()
        
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        
        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()
        
        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)
        
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        
        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix
        
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        
        return loss
        
    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        _, top_indices = torch.topk(y_true, k, dim=1)
        
        # 2. 创建权重向量
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor
            
        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        
        # 组合两种损失
        total_loss = listwise + self.pairwise_weight * pairwise
        
        return total_loss

def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算新的评估指标：Top 5 收益之和，以及与理论最高值和随机值的比值"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    
    return metrics


def calculate_submission_return(
    y_pred,
    y_true,
    masks,
    stock_indices,
    dates,
    raw_df,
    idx2stock,
):
    """Evaluate the exact selection, diversification and weighting used in predict.py."""
    returns = []
    for batch_index, date in enumerate(dates):
        valid = masks[batch_index].bool()
        scores = y_pred[batch_index][valid].detach().cpu().numpy()
        targets = y_true[batch_index][valid].detach().cpu().numpy()
        indices = stock_indices[batch_index][valid].detach().cpu().numpy()
        stock_ids = [idx2stock[int(index)] for index in indices]
        order = np.argsort(scores, kind='stable')[::-1]
        ranked_ids = [stock_ids[index] for index in order]
        ranked_scores = scores[order]
        try:
            selected_ids, weights = select_portfolio(
                raw_df,
                ranked_ids,
                ranked_scores,
                pd.Timestamp(date),
                config,
                verbose=False,
            )
        except ValueError:
            continue
        target_by_stock = dict(zip(stock_ids, targets))
        returns.append(
            float(sum(weight * target_by_stock[stock_id] for stock_id, weight in zip(selected_ids, weights)))
        )
    return float(np.mean(returns)) if returns else -float('inf')

class AlignedRankingDataset(torch.utils.data.Dataset):
    """Compact ranking dataset aligned to one shared exchange calendar.

    Features are stored once in an [stock, date, feature] cube. Individual
    60-day windows are sliced on demand, avoiding several GB of duplicated
    arrays while guaranteeing that the same time position means the same date
    for every stock passed to cross-stock attention.
    """

    def __init__(self, data, features, sequence_length, min_window_end_date=None):
        frame = data.copy()
        frame['日期'] = pd.to_datetime(frame['日期'])
        duplicate_mask = frame.duplicated(['instrument', '日期'], keep=False)
        if duplicate_mask.any():
            raise ValueError(f'存在 {int(duplicate_mask.sum())} 行重复的 股票-日期 数据')

        self.sequence_length = int(sequence_length)
        self.stock_values = np.asarray(sorted(frame['instrument'].unique()), dtype=np.int64)
        self.calendar = np.asarray(sorted(frame['日期'].unique()), dtype='datetime64[ns]')
        stock_to_row = {stock: row for row, stock in enumerate(self.stock_values)}

        shape = (len(self.stock_values), len(self.calendar))
        self.feature_cube = np.full((*shape, len(features)), np.nan, dtype=np.float32)
        self.label_cube = np.full(shape, np.nan, dtype=np.float32)
        available = np.zeros(shape, dtype=np.bool_)

        stock_rows = frame['instrument'].map(stock_to_row).to_numpy(dtype=np.int64)
        date_cols = pd.Index(self.calendar).get_indexer(frame['日期'])
        if (date_cols < 0).any():
            raise ValueError('内部错误：部分日期无法映射到共享交易日历')
        self.feature_cube[stock_rows, date_cols] = frame[features].to_numpy(dtype=np.float32)
        self.label_cube[stock_rows, date_cols] = frame['label'].to_numpy(dtype=np.float32)
        available[stock_rows, date_cols] = True

        finite_features = np.isfinite(self.feature_cube).all(axis=2)
        available &= finite_features
        cumulative = np.pad(available.astype(np.int32).cumsum(axis=1), ((0, 0), (1, 0)))
        minimum_date = pd.to_datetime(min_window_end_date) if min_window_end_date is not None else None
        self.samples = []
        for end_col in range(self.sequence_length - 1, len(self.calendar)):
            end_date = pd.Timestamp(self.calendar[end_col])
            if minimum_date is not None and end_date < minimum_date:
                continue
            start_col = end_col - self.sequence_length + 1
            complete_window = (
                cumulative[:, end_col + 1] - cumulative[:, start_col] == self.sequence_length
            )
            valid_stocks = np.flatnonzero(
                complete_window & np.isfinite(self.label_cube[:, end_col])
            )
            if len(valid_stocks) >= 10:
                self.samples.append((end_col, valid_stocks))

        memory_mb = (self.feature_cube.nbytes + self.label_cube.nbytes) / 1024 ** 2
        print(
            f'对齐数据集: {len(self.samples)} 个日期样本, '
            f'{len(self.stock_values)} 只股票, 紧凑存储 {memory_mb:.1f} MB'
        )

    def __len__(self):
        return len(self.samples)

    def limit_for_smoke_test(self, max_samples=0, max_stocks=0):
        """Limit real-data work only when explicitly requested by test env vars."""
        samples = self.samples[-max_samples:] if max_samples > 0 else self.samples
        if max_stocks > 0:
            samples = [(end_col, stock_rows[:max_stocks]) for end_col, stock_rows in samples]
        self.samples = samples

    def __getitem__(self, idx):
        end_col, stock_rows = self.samples[idx]
        start_col = end_col - self.sequence_length + 1
        sequences = self.feature_cube[stock_rows, start_col:end_col + 1]
        targets = self.label_cube[stock_rows, end_col]
        order = np.argsort(targets, kind='stable')[::-1]
        relevance = np.empty(len(targets), dtype=np.int64)
        relevance[order] = np.arange(len(targets), 0, -1, dtype=np.int64)
        return {
            'sequences': torch.from_numpy(sequences.copy()),
            'targets': torch.from_numpy(targets.copy()),
            'relevance': torch.from_numpy(relevance),
            'stock_indices': torch.from_numpy(self.stock_values[stock_rows].copy()),
            'date': pd.Timestamp(self.calendar[end_col]),
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    dates = [item['date'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.long)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks),                     # [batch, max_stocks]
        'dates': dates,
    }

# 排序训练函数
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, scaler=None):
    """训练函数，支持混合精度 (AMP)。scaler 不为 None 时启用 AMP 模式。"""
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    use_amp = scaler is not None
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # --- 前向传播放在 autocast 中（AMP 模式则自动混合精度）---
        amp_context = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
        with amp_context:
            # 模型预测（传入 mask 供 MASTER 日内注意力排除 padding 股票）
            outputs = model(sequences, mask=masks)  # [batch, max_stocks] 预测分数
            
            # 应用mask，只考虑有效股票
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
            masked_targets = targets * masks
            masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
            
            # 计算损失（只对有效股票计算）
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                # 获取有效股票的预测值和预处理好的相关性得分
                valid_pred = masked_outputs[i][valid_indices]
                valid_relevance = masked_relevance[i][valid_indices]
                
                if len(valid_pred) > 1:
                    # 直接使用预处理好的相关性得分，无需重新计算
                    loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                    batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        # --- 反向传播与参数更新 ---
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            
            if use_amp:
                # AMP: scaler 管理梯度缩放
                scaler.scale(batch_loss).backward()
                scaler.unscale_(optimizer)
                # 梯度裁剪前必须先 unscale
                if not config.get('drop_clip', True):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 原始 FP32 训练路径
                batch_loss.backward()
                if not config.get('drop_clip', True):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(
    model,
    dataloader,
    criterion,
    device,
    writer,
    epoch,
    use_amp=False,
    raw_df=None,
    idx2stock=None,
):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    submission_return_sum = 0.0
    submission_return_days = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            stock_indices = batch['stock_indices'].to(device)
            
            # 前向传播（AMP 模式下使用 autocast 加速推理）
            amp_context = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
            with amp_context:
                outputs = model(sequences, mask=masks)
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                
                if len(valid_pred) > 1:
                    _, sorted_indices = torch.sort(valid_true, descending=True)
                    relevance_scores = torch.zeros_like(valid_true, requires_grad=False)
                    relevance_scores[sorted_indices] = torch.arange(len(valid_true), 0, -1, device=device, dtype=torch.float32)
                    relevance_scores = relevance_scores.detach()
                    
                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            if raw_df is not None and idx2stock is not None:
                for sample_index, date in enumerate(batch['dates']):
                    daily_return = calculate_submission_return(
                        masked_outputs[sample_index:sample_index + 1],
                        masked_targets[sample_index:sample_index + 1],
                        masks[sample_index:sample_index + 1],
                        stock_indices[sample_index:sample_index + 1],
                        [date],
                        raw_df,
                        idx2stock,
                    )
                    if np.isfinite(daily_return):
                        submission_return_sum += daily_return
                        submission_return_days += 1
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v
            
            num_batches += 1
    
    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches
    if raw_df is not None and idx2stock is not None:
        total_metrics['submission_return'] = (
            submission_return_sum / submission_return_days
            if submission_return_days > 0
            else -float('inf')
        )
    
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length, validation_months=2):
    """按最后若干月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=validation_months)).normalize()

    # Each model row may contain a 60-day rolling feature and the model itself
    # consumes 60 rows. Keep both lookbacks using actual exchange dates rather
    # than generic weekdays (which do not know Chinese market holidays).
    trading_dates = pd.Index(sorted(df['日期'].unique()))
    val_position = int(trading_dates.searchsorted(val_start, side='left'))
    context_position = max(0, val_position - (2 * sequence_length - 2))
    val_context_start = pd.Timestamp(trading_dates[context_position])

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后{validation_months}个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start

# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 1. 数据加载
    # 赛事方仅挂载 data/stock_data.csv，不提供 train.csv。
    # 优先读 stock_data.csv，本地开发时可仍用 split_train_test.py 生成的 train.csv
    data_file = os.path.join(data_path, 'stock_data.csv')
    if not os.path.exists(data_file):
        data_file = os.path.join(data_path, 'train.csv')
    print(f'读取数据文件: {data_file}')
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    full_df['日期'] = pd.to_datetime(full_df['日期'])
    if config.get('as_of_date'):
        as_of_date = pd.Timestamp(config['as_of_date'])
        full_df = full_df[full_df['日期'] <= as_of_date].copy()
        if full_df.empty:
            raise ValueError(f'as_of_date={as_of_date.date()} 之前没有数据')
        print(f'[WALK-FORWARD] 历史截面截止: {as_of_date.date()}')
    train_df, val_df, val_start = split_train_val_by_last_month(
        full_df,
        config['sequence_length'],
        validation_months=config.get('validation_months', 2),
    )
    
    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    idx2stock = {idx: sid for sid, idx in stockid2idx.items()}
    num_stocks = len(stockid2idx)
    
    # 2. 特征工程与预处理
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)
    
    # 3. 标准化
    scaler = StandardScaler()

    train_data[features] = train_data[features].replace([np.inf, -np.inf], np.nan)
    val_data[features] = val_data[features].replace([np.inf, -np.inf], np.nan)
    # 丢弃nan数据
    train_data = train_data.dropna(subset=features)
    val_data = val_data.dropna(subset=features)
    # 然后再缩放
    train_data[features] = scaler.fit_transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    
    # 4. 创建按共享交易日历对齐的紧凑排序数据集
    train_dataset = AlignedRankingDataset(
        train_data,
        features,
        config['sequence_length'],
    )
    val_dataset = AlignedRankingDataset(
        val_data,
        features,
        config['sequence_length'],
        min_window_end_date=val_start.strftime('%Y-%m-%d')
    )

    smoke_samples = int(os.getenv('BDC_SMOKE_MAX_SAMPLES', '0'))
    smoke_stocks = int(os.getenv('BDC_SMOKE_MAX_STOCKS', '0'))
    if smoke_samples > 0 or smoke_stocks > 0:
        train_dataset.limit_for_smoke_test(smoke_samples, smoke_stocks)
        val_dataset.limit_for_smoke_test(smoke_samples, smoke_stocks)
        print(
            f'[SMOKE] 限制真实数据样本: dates={smoke_samples or "all"}, '
            f'stocks={smoke_stocks or "all"}'
        )

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError('训练集或验证集没有可用的对齐日期样本')

    # 5. 创建排序数据加载器
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # 6. 模型初始化
    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    
    # 混合精度训练：仅 CUDA 设备 + 配置开启时生效
    use_amp = config.get('use_amp', False) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("[AMP] 混合精度训练已启用，显存占用约减半")
    else:
        print(f"[AMP] 未启用混合精度 (device={device.type}, config.use_amp={config.get('use_amp', False)})")
    
    # 8. 排序模型训练
    if is_train:
        best_score = -float('inf')
        best_epoch = -1
        start_epoch = 0
        epochs_without_improvement = 0
        last_completed_epoch = 0
        stopped_early = False
        checkpoint_path = os.path.join(output_dir, 'training_checkpoint.pth')

        resume_checkpoint = os.getenv('BDC_RESUME_CHECKPOINT')
        resume_from_best = os.getenv('BDC_RESUME_FROM_BEST', '0') == '1'
        if resume_checkpoint:
            if not os.path.exists(resume_checkpoint):
                raise FileNotFoundError(f'续训检查点不存在: {resume_checkpoint}')
            checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            strategy_changes = changed_strategy_config(checkpoint.get('config'), config)
            if strategy_changes:
                details = ', '.join(
                    f'{key}: {old!r} -> {new!r}'
                    for key, (old, new) in strategy_changes.items()
                )
                raise ValueError(
                    '续训检查点的组合策略与当前配置不同，最佳验证分数不可直接比较。'
                    f'请使用新的 BDC_OUTPUT_DIR 或以 --force 重新滚动训练。变更: {details}'
                )
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if scaler is not None and checkpoint.get('amp_scaler') is not None:
                scaler.load_state_dict(checkpoint['amp_scaler'])
            start_epoch = int(checkpoint['completed_epochs'])
            best_score = float(checkpoint['best_score'])
            best_epoch = int(checkpoint['best_epoch'])
            epochs_without_improvement = int(checkpoint.get('epochs_without_improvement', 0))
            last_completed_epoch = start_epoch
            print(
                f'[RESUME] 从完整检查点继续: 已完成 {start_epoch}/{config["num_epochs"]} epoch, '
                f'最佳 submission return={best_score:.4f}'
            )
        elif resume_from_best:
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f'最佳模型不存在: {best_model_path}')
            model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
            start_epoch = int(os.getenv('BDC_RESUME_EPOCH', '0'))
            best_epoch = int(os.getenv('BDC_RESUME_BEST_EPOCH', str(start_epoch)))
            best_score = float(os.getenv('BDC_RESUME_BEST_SCORE', '-inf'))
            print(
                f'[RESUME] 仅从模型权重继续: 已完成 {start_epoch}/{config["num_epochs"]} epoch, '
                '优化器状态将重新初始化'
            )

        patience = config.get('early_stopping_patience', 0)
        already_stopped_early = patience > 0 and epochs_without_improvement >= patience
        if start_epoch >= config['num_epochs']:
            print(
                f'[RESUME] 检查点已完成目标 {start_epoch}/{config["num_epochs"]} epoch，'
                '直接补写训练摘要'
            )
        elif already_stopped_early:
            stopped_early = True
            print(
                f'[RESUME] 检查点已满足早停条件（连续 {epochs_without_improvement} 个 epoch '
                '未提升），直接补写训练摘要'
            )
        
        for epoch in range(start_epoch, config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer, scaler=scaler
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")
            
            # 验证
            eval_loss, eval_metrics = evaluate_ranking_model(
                model,
                val_loader,
                criterion,
                device,
                writer,
                epoch,
                use_amp=use_amp,
                raw_df=full_df,
                idx2stock=idx2stock,
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 保存最佳模型（基于与最终提交完全一致的组合收益）
            current_submission_return = eval_metrics.get('submission_return', -float('inf'))
            improvement = current_submission_return - best_score
            if improvement > config.get('early_stopping_min_delta', 0.0):
                best_score = current_submission_return
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - submission return: {best_score:.4f}")
            else:
                epochs_without_improvement += 1

            # 每个完整 epoch 后保存可精确恢复的训练状态，先写临时文件再原子替换。
            checkpoint = {
                'completed_epochs': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp_scaler': scaler.state_dict() if scaler is not None else None,
                'best_score': best_score,
                'best_epoch': best_epoch,
                'epochs_without_improvement': epochs_without_improvement,
                'config': config,
            }
            checkpoint_tmp = f'{checkpoint_path}.tmp'
            torch.save(checkpoint, checkpoint_tmp)
            os.replace(checkpoint_tmp, checkpoint_path)
            last_completed_epoch = epoch + 1
            print(f'训练检查点已保存: epoch {last_completed_epoch}/{config["num_epochs"]}')

            if patience > 0 and epochs_without_improvement >= patience:
                stopped_early = True
                print(
                    f'[EARLY STOP] 连续 {epochs_without_improvement} 个 epoch 未提升 '
                    f'{config.get("early_stopping_min_delta", 0.0):.6f}，提前停止'
                )
                break
        print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 submission return: {best_score:.4f}")
        with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
            f.write(f"Best epoch: {best_epoch}\nBest submission_return: {best_score:.6f}\n")
        with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
            json.dump(
                {
                    'completed_epochs': last_completed_epoch,
                    'best_epoch': best_epoch,
                    'best_submission_return': best_score,
                    'stopped_early': stopped_early,
                    'seed': config.get('seed'),
                    'as_of_date': config.get('as_of_date') or str(full_df['日期'].max().date()),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 submission return: {best_score:.4f} ##########")
