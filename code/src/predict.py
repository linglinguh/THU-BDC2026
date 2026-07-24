import os
import multiprocessing as mp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from strategy import select_portfolio


feature_cloums_map = {
	'39': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
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
	"""Build strictly calendar-aligned sequences ending on latest_date."""
	calendar = np.asarray(
		sorted(data.loc[data['日期'] <= latest_date, '日期'].unique()),
		dtype='datetime64[ns]',
	)
	if len(calendar) < sequence_length:
		raise ValueError(f'公共交易日不足 {sequence_length} 天，当前仅有 {len(calendar)} 天')
	required_dates = pd.Index(calendar[-sequence_length:])
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[data['股票代码'] == stock_id].copy()
		if stock_history['日期'].duplicated().any():
			raise ValueError(f'股票 {stock_id} 存在重复日期记录')
		stock_history = stock_history.set_index('日期').reindex(required_dates)
		values = stock_history[features].to_numpy(dtype=np.float32)
		if np.isfinite(values).all():
			sequences.append(values)
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def normalize_cross_section(scores):
	"""Standardize one model's cross-sectional scores before ensembling."""
	scores = np.asarray(scores, dtype=np.float64)
	if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
		raise ValueError('模型分数必须是一维、非空且全部为有限数值')
	standard_deviation = scores.std()
	if standard_deviation < 1e-12:
		return np.zeros_like(scores)
	return (scores - scores.mean()) / standard_deviation


def aggregate_model_scores(score_by_model):
	"""Average normalized scores on the stocks predicted by every model."""
	if not score_by_model:
		raise ValueError('至少需要一个模型才能生成集成预测')
	common_stock_ids = set(score_by_model[0])
	for model_scores in score_by_model[1:]:
		common_stock_ids.intersection_update(model_scores)
	if len(common_stock_ids) < 5:
		raise ValueError(f'多个模型共同可预测股票不足5只，当前仅有 {len(common_stock_ids)} 只')

	ordered_ids = sorted(common_stock_ids)
	normalized = []
	for model_scores in score_by_model:
		normalized.append(normalize_cross_section([model_scores[stock_id] for stock_id in ordered_ids]))
	return ordered_ids, np.mean(np.stack(normalized, axis=0), axis=0)


def get_model_dirs():
	configured = os.getenv('BDC_ENSEMBLE_DIRS', '').strip()
	if configured:
		model_dirs = [Path(value.strip()) for value in configured.split(',') if value.strip()]
	else:
		model_dirs = [Path(config['output_dir'])]
	if not model_dirs:
		raise ValueError('BDC_ENSEMBLE_DIRS 未包含有效模型目录')
	return model_dirs


def main():
	# 赛事方仅挂载 data/stock_data.csv，不提供 train.csv。
	# 优先读 stock_data.csv，本地开发时可仍用 split_train_test.py 生成的 train.csv
	data_file = os.path.join(config['data_path'], 'stock_data.csv')
	if not os.path.exists(data_file):
		data_file = os.path.join(config['data_path'], 'train.csv')
	print(f'读取数据文件: {data_file}')
	model_dirs = get_model_dirs()
	output_path = os.getenv('BDC_OUTPUT_PATH', os.path.join('./output/', 'result.csv'))
	for model_dir in model_dirs:
		if not (model_dir / 'best_model.pth').exists():
			raise FileNotFoundError(f'未找到模型文件: {model_dir / "best_model.pth"}')
		if not (model_dir / 'scaler.pkl').exists():
			raise FileNotFoundError(f'未找到Scaler文件: {model_dir / "scaler.pkl"}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	score_by_model = []
	for model_dir in model_dirs:
		model_processed = processed.copy()
		scaler = joblib.load(model_dir / 'scaler.pkl')
		model_processed[features] = scaler.transform(model_processed[features])
		sequences_np, sequence_stock_ids = build_inference_sequences(
			model_processed,
			features,
			config['sequence_length'],
			stock_ids,
			latest_date,
		)

		model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
		model.load_state_dict(torch.load(model_dir / 'best_model.pth', map_location=device))
		model.to(device)
		model.eval()
		with torch.no_grad():
			x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
			scores = model(x).squeeze(0).detach().cpu().numpy()
		score_by_model.append(dict(zip(sequence_stock_ids, scores)))
		print(f'模型预测完成: {model_dir}，股票数 {len(sequence_stock_ids)}')
		del model

	sequence_stock_ids, scores = aggregate_model_scores(score_by_model)
	order = np.argsort(scores, kind='stable')[::-1]
	ranked_stock_ids = [sequence_stock_ids[index] for index in order]
	ranked_scores = scores[order]

	if len(ranked_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')

	# Validation imports and executes this exact function too.
	top5, weights = select_portfolio(
		raw_df,
		ranked_stock_ids,
		ranked_scores,
		latest_date,
		config,
		verbose=True,
	)

	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': weights,
	})
	output_dir = os.path.dirname(output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	output_df.to_csv(output_path, index=False)

	print(f'\n预测日期: {latest_date.date()}')
	print(f'集成模型数: {len(model_dirs)}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'Top5 权重: {dict(zip(top5, [f"{w:.4f}" for w in weights]))}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
