"""
系统成绩对比可视化工具

功能:
1. 用基准逻辑(等权0.2, 无后处理)和当前系统逻辑分别评分
2. 记录历史实验到 tools/score_visualizer/history.csv
3. 绘制对比图表保存到 tools/score_visualizer/comparison.png

用法:
  python tools/score_visualizer/visualize.py                  # 评分当前 output/result.csv 并记录
  python tools/score_visualizer/visualize.py --no-record      # 仅评分不记录历史
  python tools/score_visualizer/visualize.py --view           # 仅查看历史趋势图
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_PATH = os.path.join(PROJECT_ROOT, 'output', 'result.csv')
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
HISTORY_PATH = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'history.csv')
FIG_PATH = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'comparison.png')


def calculate_return(group):
	"""与 score_self.py 完全一致的收益率计算"""
	start = group.iloc[0]
	end = group.iloc[-1]
	return (end['开盘'] - start['开盘']) / start['开盘']


def calc_weighted_score(output_df, test_data):
	"""计算加权收益率得分，逻辑与 score_docker.py 一致 (SDD §0.2)

	output_df: DataFrame, 含 stock_id 和 weight 列
	test_data: DataFrame, 含 股票代码/日期/开盘 列
	返回: float, Final Score
	"""
	output_data = output_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
	output_data['股票代码'] = output_data['股票代码'].astype(str).str.zfill(6)

	# 验证合法性
	if len(output_data) > 5:
		return -999.0
	weight_sum = float(output_data['权重'].sum())
	if not (0 <= weight_sum <= 1.0):
		return -999.0

	# 筛选测试数据
	filtered = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
	filtered = filtered.groupby('股票代码').tail(5)

	if len(filtered) == 0:
		return -999.0

	# 手动计算每只股票收益率，避免 groupby.apply 的 pandas 版本兼容问题
	records = []
	for sid, group in filtered.groupby('股票代码'):
		group = group.sort_values('日期')
		start = group.iloc[0]
		end = group.iloc[-1]
		ret = (end['开盘'] - start['开盘']) / start['开盘']
		records.append({'股票代码': str(sid).zfill(6), '收益率': ret})

	result = pd.DataFrame(records)
	result = result.merge(output_data[['股票代码', '权重']], on='股票代码')

	final_score = float((result['收益率'] * result['权重']).sum())
	return final_score


def calc_baseline_score(test_data):
	"""计算基准程序分数: 等权0.2, 选 test.csv 中随机5只(这里用前5只模拟)

	注意: 真正的 baseline 分数需要用 baseline 的 predict.py 跑 output/result.csv。
	此函数用于在没有 baseline result 时的估算参考。
	如果 test/baseline_result.csv 存在则直接用它。
	"""
	baseline_result_path = os.path.join(PROJECT_ROOT, 'test', 'baseline_result.csv')
	if os.path.exists(baseline_result_path):
		baseline_df = pd.read_csv(baseline_result_path)
		return calc_weighted_score(baseline_df, test_data)

	# 无 baseline_result.csv 时, 用 test.csv 前5只股票等权0.2 做估算
	all_stocks = sorted(test_data['股票代码'].unique())
	if len(all_stocks) < 5:
		return 0.0
	top5 = all_stocks[:5]
	baseline_df = pd.DataFrame({
		'stock_id': top5,
		'weight': [0.2] * 5
	})
	return calc_weighted_score(baseline_df, test_data)


def calc_per_stock_detail(output_df, test_data):
	"""计算每只股票的收益率明细，用于图表展示"""
	output_data = output_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
	output_data['股票代码'] = output_data['股票代码'].astype(str).str.zfill(6)
	filtered = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
	filtered = filtered.groupby('股票代码').tail(5)

	# 手动计算每只股票的收益率，避免 groupby.apply 的 pandas 版本兼容问题
	records = []
	for sid, group in filtered.groupby('股票代码'):
		group = group.sort_values('日期')
		start = group.iloc[0]
		end = group.iloc[-1]
		ret = (end['开盘'] - start['开盘']) / start['开盘']
		records.append({'股票代码': str(sid).zfill(6), '收益率': ret})

	result = pd.DataFrame(records)
	result = result.merge(output_data[['股票代码', '权重']], on='股票代码')
	result['贡献分'] = result['收益率'] * result['权重']
	return result.sort_values('收益率', ascending=False).reset_index(drop=True)


def record_history(score, label='current'):
	"""记录本次评分到历史文件"""
	row = {
		'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
		'label': label,
		'score': score,
	}
	if os.path.exists(HISTORY_PATH):
		df = pd.read_csv(HISTORY_PATH)
		df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
	else:
		df = pd.DataFrame([row])
	df.to_csv(HISTORY_PATH, index=False, encoding='utf-8-sig')
	print(f'已记录到历史: {row["timestamp"]} | {label} | {score:.6f}')


def draw_comparison(current_score, baseline_score, per_stock_df):
	"""绘制对比图表"""
	import matplotlib
	matplotlib.use('Agg')  # 无界面环境也能保存图片
	import matplotlib.pyplot as plt
	import matplotlib.font_manager as fm

	# 设置中文字体（Windows 优先使用微软雅黑）
	for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']:
		try:
			fm.findfont(font_name, fallback_to_default=False)
			plt.rcParams['font.sans-serif'] = [font_name]
			plt.rcParams['axes.unicode_minus'] = False
			break
		except Exception:
			continue

	fig, axes = plt.subplots(1, 2, figsize=(14, 6))

	# === 左图: 总分对比柱状图 ===
	ax1 = axes[0]
	labels = ['基准程序\n(等权0.2)', '当前系统']
	scores = [baseline_score, current_score]
	colors = ['#888888', '#2196F3']
	bars = ax1.bar(labels, scores, color=colors, width=0.5)
	ax1.set_ylabel('Final Score (加权收益率)', fontsize=12)
	ax1.set_title('成绩对比', fontsize=14, fontweight='bold')
	ax1.axhline(y=baseline_score, color='gray', linestyle='--', alpha=0.5, label=f'基准线 {baseline_score:.4f}')
	for bar, score in zip(bars, scores):
		ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
				 f'{score:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
	# 标注是否超越
	if current_score > baseline_score:
		ax1.text(1, max(scores) * 0.5, '已超越基准', ha='center', fontsize=11,
				color='green', fontweight='bold',
				bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
	else:
		ax1.text(1, max(scores) * 0.5, '未达基准', ha='center', fontsize=11,
				color='red', fontweight='bold',
				bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
	ax1.legend(loc='upper right')
	ax1.grid(axis='y', alpha=0.3)

	# === 右图: 每只股票收益率明细 ===
	ax2 = axes[1]
	if per_stock_df is not None and len(per_stock_df) > 0:
		x = range(len(per_stock_df))
		returns = per_stock_df['收益率'].values
		weights = per_stock_df['权重'].values
		contributions = per_stock_df['贡献分'].values
		width = 0.35
		ax2.bar([i - width/2 for i in x], returns, width, label='收益率', color='#4CAF50', alpha=0.8)
		ax2.bar([i + width/2 for i in x], contributions, width, label='贡献分(收益×权重)', color='#FF9800', alpha=0.8)
		ax2.set_xticks(list(x))
		ax2.set_xticklabels([str(s) for s in per_stock_df['股票代码']], rotation=45, ha='right')
		ax2.set_ylabel('收益率 / 贡献分', fontsize=12)
		ax2.set_title('Top5 股票收益明细', fontsize=14, fontweight='bold')
		ax2.axhline(y=0, color='black', linewidth=0.8)
		ax2.legend(loc='upper right')
		ax2.grid(axis='y', alpha=0.3)

	plt.tight_layout()
	plt.savefig(FIG_PATH, dpi=150, bbox_inches='tight')
	plt.close()
	print(f'对比图已保存: {FIG_PATH}')


def draw_history():
	"""绘制历史趋势图"""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.font_manager as fm

	for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Arial Unicode MS']:
		try:
			fm.findfont(font_name, fallback_to_default=False)
			plt.rcParams['font.sans-serif'] = [font_name]
			plt.rcParams['axes.unicode_minus'] = False
			break
		except Exception:
			continue

	if not os.path.exists(HISTORY_PATH):
		print('历史记录文件不存在，请先运行评分')
		return

	df = pd.read_csv(HISTORY_PATH)
	if len(df) == 0:
		print('历史记录为空')
		return

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(range(len(df)), df['score'].values, marker='o', linewidth=2, color='#2196F3', label='系统得分')

	# 标注每个点
	for i, row in df.iterrows():
		ax.annotate(f"{row['score']:.4f}",
					(i, row['score']), textcoords='offset points',
					xytext=(0, 10), ha='center', fontsize=8)

	ax.set_xlabel('实验序号', fontsize=12)
	ax.set_ylabel('Final Score', fontsize=12)
	ax.set_title('系统优化历史趋势', fontsize=14, fontweight='bold')
	ax.set_xticks(range(len(df)))
	ax.set_xticklabels([f"#{i+1}\n{row['label']}" for i, row in df.iterrows()], fontsize=8)
	ax.grid(axis='y', alpha=0.3)
	ax.legend()

	plt.tight_layout()
	plt.savefig(FIG_PATH.replace('comparison', 'history'), dpi=150, bbox_inches='tight')
	plt.close()
	print(f'历史趋势图已保存: {FIG_PATH.replace("comparison", "history")}')


def main():
	parser = argparse.ArgumentParser(description='系统成绩对比可视化工具')
	parser.add_argument('--no-record', action='store_true', help='仅评分不记录历史')
	parser.add_argument('--view', action='store_true', help='仅查看历史趋势图')
	parser.add_argument('--label', type=str, default='current', help='本次实验标签(如 amp_softmax)')
	args = parser.parse_args()

	if args.view:
		draw_history()
		return

	# 1. 读取数据
	if not os.path.exists(RESULT_PATH):
		print(f'错误: 找不到预测结果文件 {RESULT_PATH}')
		print('请先运行: python code/src/predict.py')
		sys.exit(1)
	if not os.path.exists(TEST_DATA_PATH):
		print(f'错误: 找不到测试数据 {TEST_DATA_PATH}')
		print('请先运行: python data/split_train_test.py')
		sys.exit(1)

	output_df = pd.read_csv(RESULT_PATH)
	test_data = pd.read_csv(TEST_DATA_PATH, dtype={'股票代码': str})
	test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)

	# 2. 计算当前系统得分
	current_score = calc_weighted_score(output_df, test_data)

	# 3. 计算基准程序得分
	baseline_score = calc_baseline_score(test_data)

	# 4. 计算每只股票明细
	per_stock_df = calc_per_stock_detail(output_df, test_data)

	# 5. 打印结果
	print('\n' + '=' * 60)
	print('              成 绩 对 比 报 告')
	print('=' * 60)
	print(f'  基准程序得分:  {baseline_score:>12.6f}')
	print(f'  当前系统得分:  {current_score:>12.6f}')
	diff = current_score - baseline_score
	print(f'  差值:          {diff:>+12.6f}')
	if current_score > baseline_score:
		print(f'  状态:          已超越基准')
		improve_pct = (diff / abs(baseline_score) * 100) if baseline_score != 0 else float('inf')
		print(f'  提升幅度:      {improve_pct:>+11.2f}%')
	else:
		print(f'  状态:          未达基准，需继续优化')
	print('=' * 60)

	# 6. 打印股票明细
	if per_stock_df is not None and len(per_stock_df) > 0:
		print('\nTop5 股票收益明细:')
		print('-' * 60)
		print(f'{"股票代码":<10} {"收益率":>10} {"权重":>8} {"贡献分":>10}')
		print('-' * 60)
		for _, row in per_stock_df.iterrows():
			print(f'{row["股票代码"]:<10} {row["收益率"]:>10.4%} {row["权重"]:>8.4f} {row["贡献分"]:>10.6f}')
		print('-' * 60)

	# 7. 记录历史
	if not args.no_record:
		record_history(current_score, args.label)

	# 8. 绘制对比图
	try:
		draw_comparison(current_score, baseline_score, per_stock_df)
	except ImportError:
		print('\n提示: 未安装 matplotlib，跳过图表生成。安装方式: pip install matplotlib')
	except Exception as e:
		print(f'\n图表生成失败: {e}')

	# 9. 如果有历史记录，绘制趋势图
	if os.path.exists(HISTORY_PATH):
		try:
			draw_history()
		except Exception as e:
			print(f'历史趋势图生成失败: {e}')


if __name__ == '__main__':
	main()
