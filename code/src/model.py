import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# MASTER 三大核心创新 (AAAI 2024)
# 参考: "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"
# ============================================================

class IntraDayAttention(nn.Module):
    """创新1：日内交互注意力 (Intra-Day Attention)

    在每个时间步内，让所有股票相互关注，捕捉同日截面关系（领涨/领跌效应）。
    输入: [B, N, L, D] → 每个时间步内做跨股票 self-attention → 输出: [B, N, L, D]
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(IntraDayAttention, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, N, L, D]
        B, N, L, D = x.shape
        # [B, N, L, D] -> [B*L, N, D]：每个时间步内股票间交互
        x_flat = x.permute(0, 2, 1, 3).reshape(B * L, N, D)

        # 构建 key_padding_mask：True 表示该股票是 padding（需忽略）
        key_padding_mask = None
        if mask is not None:
            # mask: [B, N], 1=有效, 0=padding
            key_padding_mask = (mask == 0).unsqueeze(1).expand(-1, L, -1).reshape(B * L, N)

        attn_out, _ = self.attention(x_flat, x_flat, x_flat, key_padding_mask=key_padding_mask)
        x_flat = self.norm(x_flat + self.dropout(attn_out))
        # [B*L, N, D] -> [B, N, L, D]
        x = x_flat.reshape(B, L, N, D).permute(0, 2, 1, 3)
        return x


class MarketGuidedGate(nn.Module):
    """创新3：市场引导门控 (Market-Guided Gating)

    用全市场截面均值作为"市场指数"表示，生成门控信号调制个股特征。
    当市场整体上涨时，放大个股特征；市场下跌时抑制。
    输入: [B, N, L, D] → 门控调制 → 输出: [B, N, L, D]
    """
    def __init__(self, d_model, dropout=0.1):
        super(MarketGuidedGate, self).__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: [B, N, L, D]
        if mask is not None:
            # 仅对有效股票取均值，排除 padding 干扰
            mask_exp = mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, N, 1, 1]
            market = (x * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-12)
        else:
            market = x.mean(dim=1)  # [B, L, D]

        gate = torch.sigmoid(self.gate_proj(market))  # [B, L, D]
        gated = x * gate.unsqueeze(1)  # 广播到所有股票: [B, N, L, D]
        output = self.norm(x + self.dropout(gated))
        return output


class MASTERLayer(nn.Module):
    """MASTER 交替层：日内注意力 → 日间注意力 → 市场引导门控

    每个 MASTERLayer 依次执行三大核心操作，使模型能同时捕捉
    截面关系(日内)、时序模式(日间)和市场环境(门控)。
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(MASTERLayer, self).__init__()
        # 创新1: 日内交互注意力
        self.intra_day_attn = IntraDayAttention(d_model, nhead, dropout)
        # 创新2: 日间时序注意力 (复用 TransformerEncoderLayer)
        self.inter_day_attn = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        # 创新3: 市场引导门控
        self.market_gate = MarketGuidedGate(d_model, dropout)

    def forward(self, x, mask=None):
        # x: [B, N, L, D]
        B, N, L, D = x.shape

        # 1. 日内注意力：每个时间步内股票间交互
        x = self.intra_day_attn(x, mask)

        # 2. 日间注意力：每只股票的时间序列建模
        x_inter = x.reshape(B * N, L, D)
        x_inter = self.inter_day_attn(x_inter)
        x = x_inter.reshape(B, N, L, D)

        # 3. 市场引导门控：市场特征调制个股
        x = self.market_gate(x, mask)

        return x


class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stock_features):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：每只股票都关注其他股票的特征
        attended, _ = self.cross_attention(stock_features, stock_features, stock_features)
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    """特征注意力模块"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=16):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks

        # 是否启用 MASTER 架构
        self.use_master = config.get('enable_master', False)

        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])

        if self.use_master:
            # --- MASTER 架构 (AAAI 2024) ---
            master_layers = config.get('master_num_layers', 1)
            self.master_encoder = nn.ModuleList([
                MASTERLayer(
                    config['d_model'], config['nhead'],
                    config['dim_feedforward'], config['dropout']
                )
                for _ in range(master_layers)
            ])
            # 时间维聚合（MASTER 层输出后，将 L 维聚合为 1）
            self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        else:
            # --- 原始 StockTransformer 架构（向后兼容旧权重） ---
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
            self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
            self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])

        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )

        # 最终排序分数输出
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, mask=None):
        # src: [batch, num_stocks, seq_len, feature_dim]
        # mask: [batch, num_stocks], 1=有效, 0=padding (可选)
        batch_size, num_stocks, seq_len, feature_dim = src.size()

        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)

        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)  # [batch*num_stocks, seq_len, d_model]
        src_proj = self.pos_encoder(src_proj)

        if self.use_master:
            # --- MASTER 架构 ---
            # 重塑回 [B, N, L, D] 供 MASTER 层处理
            x = src_proj.view(batch_size, num_stocks, seq_len, -1)

            # 依次通过 MASTER 交替层
            for layer in self.master_encoder:
                x = layer(x, mask)

            # 时间维聚合: [B, N, L, D] -> [B*N, L, D] -> [B*N, D]
            x_flat = x.view(batch_size * num_stocks, seq_len, -1)
            aggregated_features = self.feature_attention(x_flat)
        else:
            # --- 原始架构 ---
            temporal_features = self.temporal_encoder(src_proj)  # [batch*num_stocks, seq_len, d_model]
            aggregated_features = self.feature_attention(temporal_features)  # [batch*num_stocks, d_model]
            stock_features = aggregated_features.view(batch_size, num_stocks, -1)
            interactive_features = self.cross_stock_attention(stock_features)
            aggregated_features = interactive_features.view(batch_size * num_stocks, -1)

        # 排序特异性变换
        ranking_features = self.ranking_layers(aggregated_features)  # [batch*num_stocks, d_model//2]

        # 生成排序分数
        scores = self.score_head(ranking_features)  # [batch*num_stocks, 1]

        # 重塑为最终输出格式
        output = scores.view(batch_size, num_stocks)  # [batch, num_stocks]

        return output

