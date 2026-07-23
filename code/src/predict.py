import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from industry_map import get_industry
from utils import engineer_features_39, engineer_features_158plus39


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def diversify_by_industry(candidates, cand_scores, config):
	"""行业分散选股：从候选池中取 Top5，同一行业不超过 max_per_sector 只"""
	max_per = config.get('industry_max_per_sector', 2)
	enable = config.get('enable_industry_diversify', False)

	if not enable:
		return [candidates[i] for i in range(min(5, len(candidates)))], np.array(cand_scores[:5])

	selected, selected_scores = [], []
	industry_count = {}

	print(f'\n  行业分散选股 (每行业上限 {max_per} 只):')
	for i, sid in enumerate(candidates):
		ind = get_industry(sid)
		cnt = industry_count.get(ind, 0)
		if cnt < max_per:
			selected.append(sid)
			selected_scores.append(cand_scores[i])
			industry_count[ind] = cnt + 1
			print(f'    #{len(selected)} {sid} [{ind}] 得分={cand_scores[i]:+.4f}')
		if len(selected) >= 5:
			break

	# 不足5只时从被跳过的候选补充
	if len(selected) < 5:
		for i, sid in enumerate(candidates):
			if sid not in selected:
				selected.append(sid)
				selected_scores.append(cand_scores[i])
				print(f'    #補 {sid} [{get_industry(sid)}] (候选池回退补充)')
			if len(selected) >= 5:
				break

	return selected[:5], np.array(selected_scores[:5])


def compute_stock_metrics(raw_df, stock_ids, latest_date, momentum_days, volatility_days):
	"""计算候选股票的近期待征：动量（累计涨跌幅）和波动率（日均振幅）。

	数据来源: train.csv 的 涨跌幅 / 最高 / 最低 / 开盘 列，不依赖外部数据。
	"""
	metrics = {}
	for sid in stock_ids:
		hist = raw_df[
			(raw_df['股票代码'] == sid) & (raw_df['日期'] <= latest_date)
		].sort_values('日期').tail(max(momentum_days, volatility_days))

		if len(hist) < momentum_days:
			momentum = 0.0
		else:
			# 近 N 日累计涨跌幅 = (1+r1)(1+r2)...(1+rN) - 1
			recent = hist.tail(momentum_days)
			momentum = float((1 + recent['涨跌幅'] / 100).prod() - 1)

		if len(hist) < volatility_days:
			volatility = 0.0
		else:
			recent = hist.tail(volatility_days)
			# 日均振幅 = mean((最高 - 最低) / 开盘)
			spread = (recent['最高'] - recent['最低']) / recent['开盘']
			volatility = float(spread.mean())

		metrics[sid] = {'momentum': momentum, 'volatility': volatility}
	return metrics


def filter_candidates(ranked_ids, ranked_scores, metrics, config):
	"""候选池扩大 + 动量筛选 + 波动率风控 → 最终 Top5（单因子硬筛选模式）。

	流程: 取 Top-N 候选 → 动量筛选(剔除下跌) → 波动率筛选(剔除高波动)
	→ 不足5只时从候选池尾部回退补充 → 取 Top5
	"""
	pool_size = config.get('candidate_pool_size', 15)
	enable_mom = config.get('enable_momentum_filter', True)
	mom_days = config.get('momentum_lookback_days', 5)
	enable_vol = config.get('enable_volatility_filter', True)
	vol_days = config.get('volatility_lookback_days', 10)
	vol_threshold = config.get('volatility_max_threshold', 0.15)

	candidates = ranked_ids[:pool_size]
	cand_scores = ranked_scores[:pool_size]

	# 动量筛选
	if enable_mom:
		filtered = [(sid, sc) for sid, sc in zip(candidates, cand_scores)
					if metrics[sid]['momentum'] >= 0]
		print(f'  动量筛选: {len(candidates)} → {len(filtered)} 只 (剔除近{mom_days}日下跌)')
	else:
		filtered = list(zip(candidates, cand_scores))

	# 波动率风控
	if enable_vol:
		before_vol = len(filtered)
		filtered = [(sid, sc) for sid, sc in filtered
					if metrics[sid]['volatility'] <= vol_threshold]
		print(f'  波动率风控: {before_vol} → {len(filtered)} 只 (日均振幅>{vol_threshold:.0%}剔除)')

	# 不足5只时从候选池尾部回退补充
	excluded = [sid for sid in candidates if sid not in {f[0] for f in filtered}]
	for sid in excluded:
		if len(filtered) >= 5:
			break
		# 按 original score 顺序补充
		idx = candidates.index(sid)
		filtered.append((sid, cand_scores[idx]))

	# 按模型分数降序排列
	filtered.sort(key=lambda x: -x[1])
	sorted_ids = [f[0] for f in filtered]
	sorted_scores = [f[1] for f in filtered]

	# 行业分散 + 取 Top5
	top5, top5_scores = diversify_by_industry(sorted_ids, sorted_scores, config)

	# 打印筛选详情
	for i, sid in enumerate(top5):
		m = metrics[sid]
		print(f'  Top{i+1}: {sid} [{get_industry(sid)}]  动量={m["momentum"]:+.2%}  波动={m["volatility"]:.2%}')

	return top5, np.array(top5_scores)


# ============================================================
# 多因子评分引擎（替代硬筛选，更精确的量化排序）
# ============================================================
def compute_factor_scores(raw_df, stock_ids, latest_date):
	"""为候选池每只股票计算 5 个因子原始值。

	返回: {stock_id: {'momentum':, 'reversal':, 'volatility':,
	                   'liquidity':, 'volume_ratio':}}
	数据来源: train.csv，不依赖外部数据。
	"""
	factors = {}
	for sid in stock_ids:
		hist = raw_df[
			(raw_df['股票代码'] == sid) & (raw_df['日期'] <= latest_date)
		].sort_values('日期')

		if len(hist) < 20:
			# 数据不足，填0
			factors[sid] = {'momentum': 0.0, 'reversal': 0.0, 'volatility': 0.0,
							'liquidity': 0.0, 'volume_ratio': 1.0}
			continue

		# 动量因子：近5日累计涨跌幅
		mom_data = hist.tail(5)
		momentum = float((1 + mom_data['涨跌幅'] / 100).prod() - 1)

		# 反转因子：近3日累计涨跌幅（取反，均值回归信号）
		rev_data = hist.tail(3)
		reversal = -float((1 + rev_data['涨跌幅'] / 100).prod() - 1)

		# 波动率因子：近10日均振幅（取反，低波更优）
		vol_data = hist.tail(10)
		vol_spread = (vol_data['最高'] - vol_data['最低']) / vol_data['开盘']
		volatility = -float(vol_spread.mean())

		# 流动性因子：近10日均换手率（中性，极高或极低都不好）
		liq_data = hist.tail(10)
		liquidity = float(liq_data['换手率'].mean())

		# 成交量比：近5日均量 / 近20日均量
		vol5 = hist.tail(5)['成交量'].mean()
		vol20 = hist.tail(min(20, len(hist)))['成交量'].mean()
		volume_ratio = float(vol5 / vol20) if vol20 > 0 else 1.0

		factors[sid] = {
			'momentum': momentum,
			'reversal': reversal,
			'volatility': volatility,
			'liquidity': liquidity,
			'volume_ratio': volume_ratio,
		}
	return factors


def multi_factor_ranking(ranked_ids, ranked_scores, factor_data, config):
	"""多因子评分融合 → 候选池排序 → Top5。

	对每个因子做 z-score 标准化后加权求和（含模型分数），综合分排序取 Top5。
	"""
	pool_size = config.get('candidate_pool_size', 15)
	candidates = ranked_ids[:pool_size]
	cand_scores = ranked_scores[:pool_size]

	# 读取因子权重
	w_model = config.get('multi_factor_model_weight', 0.50)
	w_mom = config.get('multi_factor_momentum_weight', 0.15)
	w_rev = config.get('multi_factor_reversal_weight', 0.05)
	w_vol = config.get('multi_factor_volatility_weight', 0.15)
	w_liq = config.get('multi_factor_liquidity_weight', 0.05)
	w_vr = config.get('multi_factor_volume_weight', 0.10)

	# 提取各因子原始值
	n = len(candidates)
	raw = {
		'model': np.array([cand_scores[i] for i in range(n)]),
		'momentum': np.array([factor_data[candidates[i]]['momentum'] for i in range(n)]),
		'reversal': np.array([factor_data[candidates[i]]['reversal'] for i in range(n)]),
		'volatility': np.array([factor_data[candidates[i]]['volatility'] for i in range(n)]),
		'liquidity': np.array([factor_data[candidates[i]]['liquidity'] for i in range(n)]),
		'volume_ratio': np.array([factor_data[candidates[i]]['volume_ratio'] for i in range(n)]),
	}

	# z-score 标准化（均值为0，标准差为1），带稳定处理
	def zscore(x):
		std = x.std()
		if std < 1e-9:
			return np.zeros_like(x)
		return (x - x.mean()) / std

	z = {k: zscore(v) for k, v in raw.items()}

	# 流动性因子：中间值最优，偏离均值越远扣分越多。用 -abs(z) 转化
	z['liquidity'] = -np.abs(z['liquidity'])
	z['volume_ratio'] = -np.abs(z['volume_ratio'])

	# 加权综合分
	composite = (
		w_model * z['model']
		+ w_mom * z['momentum']
		+ w_rev * z['reversal']
		+ w_vol * z['volatility']
		+ w_liq * z['liquidity']
		+ w_vr * z['volume_ratio']
	)

	# 按综合分降序排列
	idx_order = np.argsort(composite)[::-1]
	sorted_ids = [candidates[i] for i in idx_order]
	sorted_scores = [cand_scores[i] for i in idx_order]

	# 行业分散选股（仅从候选池中调整 Top5，不改变排序基础）
	top5, top5_scores = diversify_by_industry(sorted_ids, sorted_scores, config)

	# 打印详情
	print(f'\n  多因子融合评分 (候选池 {len(candidates)} 只)')
	print(f'  {"股票":<8} {"模型z":>7} {"动量z":>7} {"反转z":>7} {"波动z":>7} {"流动z":>7} {"量比z":>7} {"综合":>7}')
	print(f'  {"-"*60}')
	for i in idx_order[:8]:
		sid = candidates[i]
		print(f'  {sid:<8} {z["model"][i]:>+7.2f} {z["momentum"][i]:>+7.2f} {z["reversal"][i]:>+7.2f} '
			  f'{z["volatility"][i]:>+7.2f} {z["liquidity"][i]:>+7.2f} {z["volume_ratio"][i]:>+7.2f} {composite[i]:>+7.2f}')

	return top5, top5_scores


def main():
	# 赛事方仅挂载 data/stock_data.csv，不提供 train.csv。
	# 优先读 stock_data.csv，本地开发时可仍用 split_train_test.py 生成的 train.csv
	data_file = os.path.join(config['data_path'], 'stock_data.csv')
	if not os.path.exists(data_file):
		data_file = os.path.join(config['data_path'], 'train.csv')
	print(f'读取数据文件: {data_file}')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_path = os.path.join('./output/', 'result.csv')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	scaler = joblib.load(scaler_path)
	processed[features] = scaler.transform(processed[features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]
	ranked_scores = scores[order]

	if len(ranked_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')

	# --- 推理后处理 ---
	pool_size = config.get('candidate_pool_size', 15)
	use_multi_factor = config.get('enable_multi_factor', True)

	if use_multi_factor:
		# 多因子评分模式
		print(f'\n=== 多因子评分 (候选池 Top-{pool_size}) ===')
		factor_data = compute_factor_scores(raw_df, ranked_stock_ids[:pool_size], latest_date)
		top5, top5_scores = multi_factor_ranking(
			ranked_stock_ids, ranked_scores, factor_data, config
		)
	else:
		# 单因子硬筛选模式（向后兼容）
		print(f'\n=== 后处理筛选 (候选池 Top-{pool_size}) ===')
		metrics = compute_stock_metrics(
			raw_df, ranked_stock_ids[:pool_size], latest_date,
			config.get('momentum_lookback_days', 5),
			config.get('volatility_lookback_days', 10),
		)
		top5, top5_scores = filter_candidates(ranked_stock_ids, ranked_scores, metrics, config)

	# softmax 不等权分配（带浮点精度保护）
	temperature = config.get('predict_temperature', 1.0)
	exp_scores = np.exp(top5_scores / max(temperature, 1e-6))
	weights = exp_scores / exp_scores.sum()
	# 四舍五入到6位小数，防止浮点精度导致 weight_sum > 1.0（会触发 -999 评分）
	weights = np.round(weights, 6)
	if weights.sum() > 1.0:
		weights[-1] -= weights.sum() - 1.0

	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': weights,
	})
	output_df.to_csv(output_path, index=False)

	print(f'\n预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'Top5 权重: {dict(zip(top5, [f"{w:.4f}" for w in weights]))}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
