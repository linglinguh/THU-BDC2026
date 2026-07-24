import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code' / 'src'))
sys.path.insert(0, str(ROOT))

from predict import aggregate_model_scores, build_inference_sequences, normalize_cross_section
from industry_map import STOCK_INDUSTRY, get_industry
from strategy import _diversify, select_portfolio, stable_softmax, validate_portfolio
from train import (
    AlignedRankingDataset,
    _build_label_and_clean,
    changed_strategy_config,
    split_train_val_by_last_month,
)
from utils import engineer_features
import get_stock_data
from run_walk_forward import find_resume_checkpoint


class TradingCalendarTests(unittest.TestCase):
    def test_label_uses_trading_positions_across_weekend(self):
        dates = pd.bdate_range('2026-01-05', periods=8)
        frame = pd.DataFrame({
            '股票代码': ['000001'] * len(dates),
            '日期': dates,
            '开盘': np.arange(10.0, 18.0),
        })
        result = _build_label_and_clean(frame)
        expected = (15.0 - 11.0) / 11.0
        self.assertAlmostEqual(result.loc[0, 'label'], expected)
        self.assertEqual(result['label'].notna().sum(), 3)

    def test_missing_exchange_day_invalidates_affected_label(self):
        dates = pd.bdate_range('2026-01-05', periods=8)
        complete = pd.DataFrame({
            '股票代码': ['000001'] * len(dates),
            '日期': dates,
            '开盘': np.arange(10.0, 18.0),
        })
        missing = complete.copy()
        missing['股票代码'] = '000002'
        missing = missing.drop(index=2)
        result = _build_label_and_clean(pd.concat([complete, missing], ignore_index=True))
        stock_two = result[result['股票代码'] == '000002']
        self.assertTrue(pd.isna(stock_two.iloc[0]['label']))

    def test_aligned_dataset_uses_same_calendar_slots(self):
        dates = pd.bdate_range('2026-01-05', periods=8)
        rows = []
        for stock in range(10):
            for date_index, date in enumerate(dates):
                rows.append({
                    'instrument': stock,
                    '日期': date,
                    'feature': float(date_index),
                    'label': float(stock + date_index),
                })
        frame = pd.DataFrame(rows)
        dataset = AlignedRankingDataset(frame, ['feature'], sequence_length=3)
        sample = dataset[0]
        values = sample['sequences'].numpy()[:, :, 0]
        np.testing.assert_array_equal(values, np.tile([0.0, 1.0, 2.0], (10, 1)))


class InferenceAndStrategyTests(unittest.TestCase):
    def test_ensemble_normalizes_each_model_before_averaging(self):
        first = {'000001': 1.0, '000002': 2.0, '000003': 3.0, '000004': 4.0, '000005': 5.0}
        second = {stock_id: score * 100 + 7 for stock_id, score in first.items()}
        stock_ids, scores = aggregate_model_scores([first, second])
        expected = normalize_cross_section([first[stock_id] for stock_id in stock_ids])
        np.testing.assert_allclose(scores, expected)

    def test_generated_industry_map_covers_current_constituents(self):
        self.assertGreaterEqual(len(STOCK_INDUSTRY), 300)
        for stock_id in ['002463', '301308', '688521', '300408', '688008']:
            self.assertEqual(get_industry(stock_id), '信息技术')

    def test_diversification_limits_same_first_level_industry(self):
        candidates = [
            '002463', '301308', '688521', '300408', '688008',
            '000001', '000002', '000157', '000301', '000333',
        ]
        selected, _ = _diversify(
            candidates,
            np.arange(len(candidates), 0, -1),
            {'enable_industry_diversify': True, 'industry_max_per_sector': 2},
        )
        industries = [get_industry(stock_id) for stock_id in selected]
        self.assertLessEqual(industries.count('信息技术'), 2)

    def test_inference_excludes_stock_missing_latest_day(self):
        dates = pd.bdate_range('2026-01-05', periods=4)
        rows = []
        for stock in ['000001', '000002']:
            for index, date in enumerate(dates):
                if stock == '000002' and date == dates[-1]:
                    continue
                rows.append({'股票代码': stock, '日期': date, 'feature': float(index)})
        sequences, ids = build_inference_sequences(
            pd.DataFrame(rows), ['feature'], 3, ['000001', '000002'], dates[-1]
        )
        self.assertEqual(ids, ['000001'])
        self.assertEqual(sequences.shape, (1, 3, 1))

    def test_stable_softmax_handles_large_scores(self):
        weights = stable_softmax(np.array([10000.0, 9999.0, 9998.0]))
        self.assertTrue(np.isfinite(weights).all())
        self.assertAlmostEqual(weights.sum(), 1.0)

    def test_selection_keeps_rounded_concentrated_weights_non_negative(self):
        dates = pd.bdate_range('2026-01-05', periods=20)
        rows = []
        stock_ids = [f'{index:06d}' for index in range(1, 21)]
        for stock_id in stock_ids:
            for date in dates:
                rows.append({
                    '股票代码': stock_id,
                    '日期': date,
                    '开盘': 10.0,
                    '最高': 10.1,
                    '最低': 9.9,
                    '成交量': 1000.0,
                    '换手率': 1.0,
                    '涨跌幅': 0.0,
                })
        selected, weights = select_portfolio(
            pd.DataFrame(rows),
            stock_ids,
            np.array([100.0] + [0.0] * 19),
            dates[-1],
            {
                'candidate_pool_size': 15,
                'enable_multi_factor': True,
                'enable_industry_diversify': False,
                'multi_factor_model_weight': 1.0,
                'predict_temperature': 0.01,
            },
        )
        self.assertEqual(len(selected), 5)
        self.assertTrue((weights > 0).all())
        self.assertLessEqual(weights.sum(), 1.0)

    def test_portfolio_validation_rejects_duplicates_and_negative_weights(self):
        with self.assertRaises(ValueError):
            validate_portfolio(['000001', '000001'], [0.5, 0.5])
        with self.assertRaises(ValueError):
            validate_portfolio(['000001', '000002'], [1.1, -0.1])

    def test_selection_returns_valid_submission(self):
        dates = pd.bdate_range('2026-01-05', periods=20)
        rows = []
        stock_ids = [f'{index:06d}' for index in range(1, 21)]
        for stock_index, stock_id in enumerate(stock_ids):
            for date_index, date in enumerate(dates):
                rows.append({
                    '股票代码': stock_id,
                    '日期': date,
                    '开盘': 10 + stock_index + date_index * 0.01,
                    '最高': 10.2 + stock_index + date_index * 0.01,
                    '最低': 9.8 + stock_index + date_index * 0.01,
                    '成交量': 1000 + stock_index * 10 + date_index,
                    '换手率': 1 + stock_index * 0.01,
                    '涨跌幅': stock_index * 0.01,
                })
        config = {
            'candidate_pool_size': 15,
            'enable_multi_factor': True,
            'enable_industry_diversify': True,
            'industry_max_per_sector': 2,
            'predict_temperature': 1.0,
        }
        selected, weights = select_portfolio(
            pd.DataFrame(rows), stock_ids, np.arange(20.0)[::-1], dates[-1], config
        )
        self.assertEqual(len(selected), 5)
        self.assertEqual(len(set(selected)), 5)
        self.assertTrue((weights >= 0).all())
        self.assertLessEqual(weights.sum(), 1.0)


class FeatureAndCliTests(unittest.TestCase):
    def test_strategy_change_invalidates_checkpoint_score(self):
        old = {'predict_temperature': 1.0, 'candidate_pool_size': 15}
        new = {'predict_temperature': 0.2, 'candidate_pool_size': 15}
        changes = changed_strategy_config(old, new)
        self.assertEqual(changes['predict_temperature'], (1.0, 0.2))
        self.assertNotIn('candidate_pool_size', changes)

    def test_walk_forward_resumes_only_unfinished_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            checkpoint = run_dir / 'training_checkpoint.pth'
            checkpoint.touch()
            self.assertEqual(find_resume_checkpoint(run_dir), checkpoint)
            self.assertIsNone(find_resume_checkpoint(run_dir, force=True))
            (run_dir / 'training_summary.json').touch()
            self.assertIsNone(find_resume_checkpoint(run_dir))

    def test_validation_window_can_move_to_historical_cutoff(self):
        dates = pd.bdate_range('2025-01-01', '2026-07-23')
        frame = pd.DataFrame({'日期': dates, '股票代码': '000001'})
        train, validation, start = split_train_val_by_last_month(frame, 20, validation_months=2)
        self.assertLess(pd.to_datetime(train['日期']).max(), start)
        self.assertGreaterEqual(pd.to_datetime(validation['日期']).max(), start)

    def test_rsquare_features_are_not_zero_after_warmup(self):
        count = 80
        frame = pd.DataFrame({
            '开盘': np.linspace(10, 20, count),
            '最高': np.linspace(10.5, 20.5, count),
            '最低': np.linspace(9.5, 19.5, count),
            '收盘': np.linspace(10, 20, count) + np.sin(np.arange(count)) * 0.1,
            '成交量': np.linspace(1000, 2000, count),
            '成交额': np.linspace(10000, 40000, count),
        })
        result = engineer_features(frame)
        self.assertGreater(np.count_nonzero(result['RSQR5'].to_numpy()), 50)

    def test_download_defaults_end_today(self):
        with patch.object(sys, 'argv', ['get_stock_data.py']):
            args = get_stock_data.parse_args()
        self.assertEqual(args.end_date, pd.Timestamp.today().strftime('%Y-%m-%d'))


if __name__ == '__main__':
    unittest.main()
