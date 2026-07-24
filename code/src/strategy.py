"""Shared portfolio-selection logic used by validation and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industry_map import get_industry

_FACTOR_CACHE = {}


def _zscore(values: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=np.float64)
	std = values.std()
	if not np.isfinite(std) or std < 1e-9:
		return np.zeros_like(values)
	return (values - values.mean()) / std


def compute_factor_scores(raw_df, stock_ids, latest_date):
	"""Compute causal factors using observations available on or before latest_date."""
	factors = {}
	for stock_id in stock_ids:
		cache_key = (id(raw_df), str(stock_id), pd.Timestamp(latest_date).value)
		if cache_key in _FACTOR_CACHE:
			factors[stock_id] = _FACTOR_CACHE[cache_key]
			continue
		history = raw_df[
			(raw_df['股票代码'] == stock_id) & (raw_df['日期'] <= latest_date)
		].sort_values('日期')

		if len(history) < 20:
			factors[stock_id] = {
				'momentum': 0.0,
				'reversal': 0.0,
				'volatility': 0.0,
				'liquidity': 0.0,
				'volume_ratio': 1.0,
			}
			_FACTOR_CACHE[cache_key] = factors[stock_id]
			continue

		momentum = float((1 + history.tail(5)['涨跌幅'] / 100).prod() - 1)
		reversal = -float((1 + history.tail(3)['涨跌幅'] / 100).prod() - 1)
		volatility = -float(
			((history.tail(10)['最高'] - history.tail(10)['最低']) / history.tail(10)['开盘']).mean()
		)
		liquidity = float(history.tail(10)['换手率'].mean())
		volume_5 = history.tail(5)['成交量'].mean()
		volume_20 = history.tail(20)['成交量'].mean()
		volume_ratio = float(volume_5 / volume_20) if volume_20 > 0 else 1.0

		factors[stock_id] = {
			'momentum': momentum,
			'reversal': reversal,
			'volatility': volatility,
			'liquidity': liquidity,
			'volume_ratio': volume_ratio,
		}
		_FACTOR_CACHE[cache_key] = factors[stock_id]
	return factors


def _diversify(candidates, scores, config, verbose=False):
	max_per_sector = config.get('industry_max_per_sector', 2)
	if not config.get('enable_industry_diversify', False):
		return list(candidates[:5]), np.asarray(scores[:5], dtype=np.float64)

	selected_ids = []
	selected_scores = []
	industry_counts = {}
	for stock_id, score in zip(candidates, scores):
		industry = get_industry(stock_id)
		if industry_counts.get(industry, 0) >= max_per_sector:
			continue
		selected_ids.append(stock_id)
		selected_scores.append(score)
		industry_counts[industry] = industry_counts.get(industry, 0) + 1
		if verbose:
			print(f'    #{len(selected_ids)} {stock_id} [{industry}] 得分={score:+.4f}')
		if len(selected_ids) == 5:
			break

	# A restrictive or stale industry map must never make the output invalid.
	if len(selected_ids) < 5:
		for stock_id, score in zip(candidates, scores):
			if stock_id not in selected_ids:
				selected_ids.append(stock_id)
				selected_scores.append(score)
			if len(selected_ids) == 5:
				break

	return selected_ids, np.asarray(selected_scores, dtype=np.float64)


def _multi_factor_select(raw_df, ranked_ids, ranked_scores, latest_date, config, verbose=False):
	pool_size = min(config.get('candidate_pool_size', 15), len(ranked_ids))
	candidates = list(ranked_ids[:pool_size])
	model_scores = np.asarray(ranked_scores[:pool_size], dtype=np.float64)
	factors = compute_factor_scores(raw_df, candidates, latest_date)

	raw = {
		'model': model_scores,
		'momentum': np.asarray([factors[s]['momentum'] for s in candidates]),
		'reversal': np.asarray([factors[s]['reversal'] for s in candidates]),
		'volatility': np.asarray([factors[s]['volatility'] for s in candidates]),
		'liquidity': np.asarray([factors[s]['liquidity'] for s in candidates]),
		'volume_ratio': np.asarray([factors[s]['volume_ratio'] for s in candidates]),
	}
	z = {name: _zscore(values) for name, values in raw.items()}
	z['liquidity'] = -np.abs(z['liquidity'])
	z['volume_ratio'] = -np.abs(z['volume_ratio'])

	composite = (
		config.get('multi_factor_model_weight', 0.50) * z['model']
		+ config.get('multi_factor_momentum_weight', 0.15) * z['momentum']
		+ config.get('multi_factor_reversal_weight', 0.05) * z['reversal']
		+ config.get('multi_factor_volatility_weight', 0.15) * z['volatility']
		+ config.get('multi_factor_liquidity_weight', 0.05) * z['liquidity']
		+ config.get('multi_factor_volume_weight', 0.10) * z['volume_ratio']
	)
	order = np.argsort(composite, kind='stable')[::-1]
	ordered_ids = [candidates[index] for index in order]
	# Allocate weights from the same composite score that selected the portfolio.
	ordered_scores = composite[order]
	if verbose:
		print(f'  多因子融合评分：候选池 {pool_size} 只')
	return _diversify(ordered_ids, ordered_scores, config, verbose=verbose)


def _filtered_select(raw_df, ranked_ids, ranked_scores, latest_date, config, verbose=False):
	pool_size = min(config.get('candidate_pool_size', 15), len(ranked_ids))
	candidates = list(ranked_ids[:pool_size])
	scores = np.asarray(ranked_scores[:pool_size], dtype=np.float64)
	factors = compute_factor_scores(raw_df, candidates, latest_date)

	kept = []
	for stock_id, score in zip(candidates, scores):
		factor = factors[stock_id]
		if config.get('enable_momentum_filter', True) and factor['momentum'] < 0:
			continue
		# compute_factor_scores stores negative volatility so convert it back here.
		if (
			config.get('enable_volatility_filter', True)
			and -factor['volatility'] > config.get('volatility_max_threshold', 0.15)
		):
			continue
		kept.append((stock_id, score))

	for stock_id, score in zip(candidates, scores):
		if len(kept) >= 5:
			break
		if stock_id not in {item[0] for item in kept}:
			kept.append((stock_id, score))
	kept.sort(key=lambda item: -item[1])
	return _diversify(
		[item[0] for item in kept],
		[item[1] for item in kept],
		config,
		verbose=verbose,
	)


def stable_softmax(scores, temperature=1.0):
	scores = np.asarray(scores, dtype=np.float64)
	if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
		raise ValueError('组合评分必须是一维、非空且全部为有限数值')
	temperature = max(float(temperature), 1e-6)
	shifted = (scores - scores.max()) / temperature
	exponentials = np.exp(shifted)
	weights = exponentials / exponentials.sum()
	return weights


def validate_portfolio(stock_ids, weights, max_stocks=5):
	stock_ids = [str(stock_id).zfill(6) for stock_id in stock_ids]
	weights = np.asarray(weights, dtype=np.float64)
	if not 1 <= len(stock_ids) <= max_stocks:
		raise ValueError(f'组合股票数必须为 1 到 {max_stocks}，当前为 {len(stock_ids)}')
	if len(set(stock_ids)) != len(stock_ids):
		raise ValueError('组合中存在重复股票代码')
	if weights.shape != (len(stock_ids),) or not np.isfinite(weights).all():
		raise ValueError('权重数量不匹配或含 NaN/Inf')
	if (weights < 0).any():
		raise ValueError('权重不得为负数')
	if weights.sum() > 1.0 + 1e-9:
		raise ValueError(f'权重和不得超过 1，当前为 {weights.sum()}')
	return stock_ids, weights


def select_portfolio(raw_df, ranked_ids, ranked_scores, latest_date, config, verbose=False):
	"""Run the exact post-processing and weighting used for submission."""
	if len(ranked_ids) < 5:
		raise ValueError(f'可预测股票不足 5 只，当前仅有 {len(ranked_ids)} 只')
	if config.get('enable_multi_factor', True):
		stock_ids, selection_scores = _multi_factor_select(
			raw_df, ranked_ids, ranked_scores, latest_date, config, verbose=verbose
		)
	else:
		stock_ids, selection_scores = _filtered_select(
			raw_df, ranked_ids, ranked_scores, latest_date, config, verbose=verbose
		)
	weights = stable_softmax(selection_scores, config.get('predict_temperature', 1.0))
	# Allocate integer millionths so serialization remains positive for every
	# selected stock and never exceeds the competition's unit cap.
	weight_units = np.floor(weights * 1_000_000).astype(np.int64)
	weight_units[weight_units == 0] = 1
	excess_units = max(int(weight_units.sum()) - 1_000_000, 0)
	if excess_units:
		weight_units[int(np.argmax(weight_units))] -= excess_units
	weights = weight_units.astype(np.float64) / 1_000_000
	return validate_portfolio(stock_ids, weights)
