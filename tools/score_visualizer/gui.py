#!/usr/bin/env python3
r"""
成绩对比可视化桌面应用 v2.0

用法:
  python test/score_gui.py
"""
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')


# ============================================================
# 评分引擎（进程内计算，不依赖 subprocess 输出解析）
# ============================================================
def _stock_return(group):
	start, end = group.iloc[0], group.iloc[-1]
	return (end['开盘'] - start['开盘']) / start['开盘']


def compute_scores():
	"""直接从文件计算当前分/基准分/明细，返回 dict"""
	import pandas as pd

	result_path = os.path.join(PROJECT_ROOT, 'output', 'result.csv')
	test_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
	baseline_result_path = os.path.join(PROJECT_ROOT, 'test', 'baseline_result.csv')

	if not os.path.exists(result_path):
		return None
	if not os.path.exists(test_path):
		return None

	output_df = pd.read_csv(result_path)
	test_data = pd.read_csv(test_path, dtype={'股票代码': str})
	test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)

	# 当前系统分
	output_data = output_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
	output_data['股票代码'] = output_data['股票代码'].astype(str).str.zfill(6)

	current_score = -999.0
	if len(output_data) <= 5 and 0 <= float(output_data['权重'].sum()) <= 1.0:
		filtered = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
		if len(filtered) > 0:
			filtered = filtered.groupby('股票代码').tail(5)
			records = []
			for sid, grp in filtered.groupby('股票代码'):
				grp = grp.sort_values('日期')
				ret = _stock_return(grp)
				records.append({'股票代码': str(sid).zfill(6), '收益率': ret})
			ret_df = pd.DataFrame(records)
			ret_df = ret_df.merge(output_data[['股票代码', '权重']], on='股票代码')
			current_score = float((ret_df['收益率'] * ret_df['权重']).sum())

	# 基准分
	if os.path.exists(baseline_result_path):
		bl_df = pd.read_csv(baseline_result_path)
		bl_data = bl_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
		bl_data['股票代码'] = bl_data['股票代码'].astype(str).str.zfill(6)
		filtered = test_data[test_data['股票代码'].isin(bl_data['股票代码'])]
		filtered = filtered.groupby('股票代码').tail(5)
		records = []
		for sid, grp in filtered.groupby('股票代码'):
			grp = grp.sort_values('日期')
			ret = _stock_return(grp)
			records.append({'股票代码': str(sid).zfill(6), '收益率': ret})
		ret_df = pd.DataFrame(records)
		ret_df = ret_df.merge(bl_data[['股票代码', '权重']], on='股票代码')
		baseline_score = float((ret_df['收益率'] * ret_df['权重']).sum())
	else:
		# fallback: 前5只等权
		all_stocks = sorted(test_data['股票代码'].unique())
		if len(all_stocks) < 5:
			baseline_score = 0.0
		else:
			top5 = all_stocks[:5]
			bl_data = pd.DataFrame({'股票代码': top5, '权重': [0.2]*5})
			filtered = test_data[test_data['股票代码'].isin(bl_data['股票代码'])]
			filtered = filtered.groupby('股票代码').tail(5)
			records = []
			for sid, grp in filtered.groupby('股票代码'):
				grp = grp.sort_values('日期')
				ret = _stock_return(grp)
				records.append({'股票代码': str(sid).zfill(6), '收益率': ret})
			ret_df = pd.DataFrame(records)
			ret_df = ret_df.merge(bl_data, on='股票代码')
			baseline_score = float((ret_df['收益率'] * ret_df['权重']).sum())

	return {
		'current': current_score,
		'baseline': baseline_score,
		'diff': current_score - baseline_score,
		'improve_pct': (current_score - baseline_score) / abs(baseline_score) * 100 if baseline_score != 0 else 0.0,
		'exceed': current_score > baseline_score,
	}


def load_history():
	import pandas as pd
	hist_csv = os.path.join(PROJECT_ROOT, 'test', 'score_history.csv')
	if not os.path.exists(hist_csv):
		return None
	df = pd.read_csv(hist_csv)
	return df if len(df) > 0 else None


def run_subprocess(script_path):
	"""运行子脚本并返回 (输出文本, 是否成功)"""
	cmd = [VENV_PYTHON, script_path]
	try:
		result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
								encoding='utf-8', errors='replace', timeout=600)
		return result.stdout + '\n' + result.stderr, result.returncode == 0
	except subprocess.TimeoutExpired:
		return '执行超时', False
	except Exception as e:
		return str(e), False


# ============================================================
# 步骤进度条组件（替代 tkinter 的 indeterminate 滑块）
# ============================================================
class StepProgress(tk.Frame):
	"""自定义步骤进度条：显示圆形节点 + 连线 + 步骤名 + 状态"""
	CIRCLE_R = 12    # 圆半径
	GAP = 80         # 节点间距

	def __init__(self, parent, steps, colors):
		super().__init__(parent, bg=colors['card'])
		self.colors = colors
		self.steps = steps   # list of str
		self.statuses = ['pending'] * len(steps)  # pending | active | done
		self.canvas = tk.Canvas(self, bg=colors['card'], highlightthickness=0,
								height=70, width=len(steps) * self.GAP + 40)
		self.canvas.pack()
		self._draw()

	def set_step(self, index, status):
		"""设置第 index 步的状态: 'pending' | 'active' | 'done'"""
		if 0 <= index < len(self.statuses):
			self.statuses[index] = status
		self._draw()

	def reset(self):
		self.statuses = ['pending'] * len(self.steps)
		self._draw()

	def _draw(self):
		c = self.canvas
		c.delete('all')
		n = len(self.steps)
		x0 = 30
		y = 30

		for i in range(n):
			x = x0 + i * self.GAP
			# 连线（到下一个节点）
			if i < n - 1:
				nx = x0 + (i + 1) * self.GAP
				line_color = self.colors['success'] if self.statuses[i] == 'done' else self.colors['border']
				c.create_line(x + self.CIRCLE_R, y, nx - self.CIRCLE_R, y,
							fill=line_color, width=3)

			# 圆形节点
			st = self.statuses[i]
			if st == 'done':
				fill, outline, text = self.colors['success'], self.colors['success'], '✓'
			elif st == 'active':
				fill, outline, text = self.colors['primary'], self.colors['primary'], str(i + 1)
			else:
				fill, outline, text = self.colors['card'], self.colors['border'], str(i + 1)

			c.create_oval(x, y - self.CIRCLE_R, x + self.CIRCLE_R * 2, y + self.CIRCLE_R,
						fill=fill, outline=outline, width=3)
			c.create_text(x + self.CIRCLE_R, y, text=text,
						fill='white' if st != 'pending' else self.colors['text3'],
						font=('Microsoft YaHei UI', 10, 'bold'))

			# 步骤名
			lbl_color = self.colors['text'] if st != 'pending' else self.colors['text3']
			c.create_text(x + self.CIRCLE_R, y + 28, text=self.steps[i],
						fill=lbl_color, font=('Microsoft YaHei UI', 8),
						anchor='center')


# ============================================================
# UI 主窗口
# ============================================================
class ScoreApp:
	def __init__(self, root):
		self.root = root
		self.root.title('BDC2026 成绩对比工具')
		self.root.geometry('1050x800')
		self.root.minsize(950, 680)

		self.C = {
			'bg': '#0f172a',
			'surface': '#1e293b',
			'card': '#1e293b',
			'primary': '#6366f1',
			'primary_hover': '#4f46e5',
			'success': '#10b981',
			'danger': '#ef4444',
			'warning': '#f59e0b',
			'info': '#3b82f6',
			'text': '#f1f5f9',
			'text2': '#94a3b8',
			'text3': '#64748b',
			'border': '#334155',
			'log_bg': '#0f172a',
			'log_fg': '#86efac',
			'purple': '#8b5cf6',
			'pink': '#ec4899',
		}

		self.root.configure(bg=self.C['bg'])
		self._build()
		self._refresh_scores()

	def _build(self):
		# 标题栏
		hdr = tk.Frame(self.root, bg=self.C['bg'], height=50)
		hdr.pack(fill='x', padx=20, pady=(12, 8))
		hdr.pack_propagate(False)
		title_left = tk.Frame(hdr, bg=self.C['bg'])
		title_left.pack(side='left')
		tk.Label(title_left, text='📈', font=('Segoe UI Emoji', 22), bg=self.C['bg']).pack(side='left', padx=(0, 8))
		tk.Label(title_left, text='BDC2026 股票预测成绩对比', font=('Microsoft YaHei UI', 16, 'bold'),
				fg=self.C['text'], bg=self.C['bg']).pack(side='left')

		title_right = tk.Frame(hdr, bg=self.C['bg'])
		title_right.pack(side='right')
		self.version_label = tk.Label(title_right, text='v2.1', font=('Consolas', 9),
									fg=self.C['text3'], bg=self.C['bg'])
		self.version_label.pack(side='right', padx=5)

		# 主体三栏
		body = tk.Frame(self.root, bg=self.C['bg'])
		body.pack(fill='both', expand=True, padx=20, pady=(0, 8))

		# 左栏: 分数
		left = tk.Frame(body, bg=self.C['bg'], width=340)
		left.pack(side='left', fill='y', padx=(0, 8))
		left.pack_propagate(False)
		self._build_score_panel(left)

		# 中栏: 操作 & 状态
		mid = tk.Frame(body, bg=self.C['bg'], width=260)
		mid.pack(side='left', fill='y', padx=(0, 8))
		mid.pack_propagate(False)
		self._build_control_panel(mid)

		# 右栏: 日志
		right = tk.Frame(body, bg=self.C['bg'])
		right.pack(side='right', fill='both', expand=True)
		self._build_log_panel(right)

		# 底部状态栏
		bar = tk.Frame(self.root, bg=self.C['border'], height=26)
		bar.pack(fill='x', side='bottom')
		bar.pack_propagate(False)
		self.footer_var = tk.StringVar(value='就绪')
		tk.Label(bar, textvariable=self.footer_var, font=('Microsoft YaHei UI', 8),
				fg=self.C['text2'], bg=self.C['border']).pack(side='left', padx=15)
		self.footer_time = tk.Label(bar, font=('Consolas', 8), fg=self.C['text3'], bg=self.C['border'])
		self.footer_time.pack(side='right', padx=15)

	def _make_card(self, parent, **kw):
		f = tk.Frame(parent, bg=self.C['card'], padx=16, pady=14,
					highlightbackground=self.C['border'], highlightthickness=1)
		f.pack(fill='x', pady=(0, 10), **kw)
		return f

	def _make_btn(self, parent, text, command, color=None, size=10):
		c = color or self.C['primary']
		bg_hover = self._darken(c)
		btn = tk.Button(parent, text=text, font=('Microsoft YaHei UI', size, 'bold'),
						bg=c, fg='white', relief='flat', padx=14, pady=10,
						command=command, cursor='hand2', bd=0,
						activebackground=bg_hover, activeforeground='white')
		def on_enter(e): btn.configure(bg=bg_hover)
		def on_leave(e): btn.configure(bg=c)
		btn.bind('<Enter>', on_enter)
		btn.bind('<Leave>', on_leave)
		return btn

	@staticmethod
	def _darken(hex_color, factor=0.78):
		h = hex_color.lstrip('#')
		r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
		return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'

	# ======================== 左侧: 分数面板 ========================
	def _build_score_panel(self, parent):
		# 当前分大卡片
		card = self._make_card(parent)
		tk.Label(card, text='当前系统得分', font=('Microsoft YaHei UI', 11),
				fg=self.C['text2'], bg=self.C['card']).pack(anchor='w')
		row1 = tk.Frame(card, bg=self.C['card'])
		row1.pack(fill='x', pady=(6, 0))
		self.current_var = tk.StringVar(value='---')
		tk.Label(row1, textvariable=self.current_var, font=('Consolas', 38, 'bold'),
				fg=self.C['success'], bg=self.C['card']).pack(side='left')
		# 状态徽章
		self.badge = tk.Frame(row1, bg=self.C['card'])
		self.badge.pack(side='right')
		self.badge_label = tk.Label(self.badge, text='等待评分', font=('Microsoft YaHei UI', 10, 'bold'),
									padx=8, pady=4, bg=self.C['text3'], fg='white')
		self.badge_label.pack()

		# 分隔线
		tk.Frame(card, bg=self.C['border'], height=1).pack(fill='x', pady=(12, 10))

		# 基准分
		row2 = tk.Frame(card, bg=self.C['card'])
		row2.pack(fill='x')
		tk.Label(row2, text='基准程序得分', font=('Microsoft YaHei UI', 10),
				fg=self.C['text3'], bg=self.C['card']).pack(anchor='w')
		self.baseline_var = tk.StringVar(value='---')
		tk.Label(row2, textvariable=self.baseline_var, font=('Consolas', 28, 'bold'),
				fg=self.C['text3'], bg=self.C['card']).pack(anchor='w', pady=(2, 0))

		# 差值 & 提升
		diff_row = tk.Frame(card, bg=self.C['card'])
		diff_row.pack(fill='x', pady=(12, 0))
		self.diff_var = tk.StringVar(value='')
		tk.Label(diff_row, textvariable=self.diff_var, font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.C['primary'], bg=self.C['card']).pack(anchor='w')

		# 历史摘要卡片
		hcard = self._make_card(parent)
		tk.Label(hcard, text='实验历史', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 6))
		self.hist_var = tk.StringVar(value='暂无记录')
		tk.Label(hcard, textvariable=self.hist_var, font=('Microsoft YaHei UI', 9),
				fg=self.C['text2'], bg=self.C['card'], wraplength=280, justify='left').pack(anchor='w')

	# ======================== 中间: 操作面板 ========================
	def _build_control_panel(self, parent):
		card = self._make_card(parent)
		tk.Label(card, text='操作', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 10))

		self.btn_full = self._make_btn(card, '🚀  一键预测 + 评分', self._on_full_pipeline, self.C['success'], 11)
		self.btn_full.pack(fill='x', pady=4)

		self.btn_predict = self._make_btn(card, '📊  仅运行预测', self._on_predict, self.C['purple'])
		self.btn_predict.pack(fill='x', pady=4)

		self.btn_score = self._make_btn(card, '📋  仅评分', self._on_refresh_score, self.C['info'])
		self.btn_score.pack(fill='x', pady=4)

		tk.Frame(card, bg=self.C['border'], height=1).pack(fill='x', pady=(10, 8))

		self.btn_chart = self._make_btn(card, '📈  打开对比图表', self._on_show_chart, self.C['warning'], 9)
		self.btn_chart.pack(fill='x', pady=3)

		self.btn_history = self._make_btn(card, '📉  打开历史趋势', self._on_show_history, self.C['pink'], 9)
		self.btn_history.pack(fill='x', pady=3)

		# 步骤进度条
		pcard = self._make_card(parent)
		self.progress_var = tk.StringVar(value='空闲')
		tk.Label(pcard, textvariable=self.progress_var, font=('Microsoft YaHei UI', 9, 'bold'),
				fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 6))
		self.stepper = StepProgress(pcard, steps=['预测', '评分', '图表'], colors=self.C)
		self.stepper.pack()

		# 快捷提示
		tcard = self._make_card(parent)
		tk.Label(tcard, text='💡 提示', font=('Microsoft YaHei UI', 10, 'bold'),
				fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 4))
		tips = (
			'• 一键按钮 = 预测 + 评分\n'
			'• 每次改进后点一键即可\n'
			'• 图表需先运行评分生成\n'
			'• baseline_result.csv 可自定义基准'
		)
		tk.Label(tcard, text=tips, font=('Microsoft YaHei UI', 8),
				fg=self.C['text3'], bg=self.C['card'], justify='left', wraplength=220).pack(anchor='w')

	# ======================== 右侧: 日志 ========================
	def _build_log_panel(self, parent):
		card = tk.Frame(parent, bg=self.C['card'], padx=16, pady=14,
					   highlightbackground=self.C['border'], highlightthickness=1)
		card.pack(fill='both', expand=True)

		hh = tk.Frame(card, bg=self.C['card'])
		hh.pack(fill='x', pady=(0, 8))
		tk.Label(hh, text='运行日志', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.C['text'], bg=self.C['card']).pack(side='left')
		tk.Button(hh, text='清空', font=('Microsoft YaHei UI', 9), bg=self.C['primary'], fg='white',
				relief='flat', command=self._clear_log, cursor='hand2', padx=10,
				activebackground=self.C['primary_hover']).pack(side='right')

		self.log = scrolledtext.ScrolledText(
			card, bg=self.C['log_bg'], fg=self.C['log_fg'], insertbackground='white',
			font=('Consolas', 9), wrap='word', relief='flat', borderwidth=0, height=8,
		)
		self.log.pack(fill='both', expand=True)
		self._log_raw('应用已启动，点击操作按钮开始。')

	# ======================== 动作方法 ========================
	def _log_raw(self, text):
		self.log.insert('end', f'[{datetime.now().strftime("%H:%M:%S")}] {text}\n')
		self.log.see('end')
		self.root.update_idletasks()

	def _log(self, text):
		self._log_raw(text)

	def _clear_log(self):
		self.log.delete('1.0', 'end')

	def _busy(self, on=True, step=None, msg=''):
		"""设置忙碌状态。step 为当前激活步骤索引 (0/1/2)，msg 为状态文字"""
		for b in [self.btn_full, self.btn_predict, self.btn_score, self.btn_chart, self.btn_history]:
			b.configure(state='disabled' if on else 'normal')
		if on:
			self.stepper.reset()
			self.progress_var.set(msg)
			if step is not None:
				self.stepper.set_step(step, 'active')
		else:
			self.stepper.reset()
			self.progress_var.set('空闲')
		self.root.update_idletasks()

	def _step_done(self, index):
		"""标记某步骤完成"""
		self.stepper.set_step(index, 'done')

	def _step_active(self, index):
		"""标记某步骤进行中"""
		self.stepper.set_step(index, 'active')

	# ---- 评分刷新（不跑子进程） ----
	def _refresh_scores(self):
		res = compute_scores()
		if res is None:
			self.current_var.set('---')
			self.baseline_var.set('---')
			self.diff_var.set('请先运行预测生成 result.csv')
			self.badge_label.configure(bg=self.C['text3'], text='等待评分')
			return

		self.current_var.set(f'{res["current"]:.6f}')
		self.baseline_var.set(f'{res["baseline"]:.6f}')
		self.diff_var.set(f'差值: {res["diff"]:+.6f}  提升: {res["improve_pct"]:+.1f}%')

		if res['exceed']:
			self.badge_label.configure(bg=self.C['success'], text='✓ 已超越基准')
		else:
			self.badge_label.configure(bg=self.C['danger'], text='✗ 未达基准')

		self._refresh_history()

	def _on_refresh_score(self):
		self._busy(True, step=1, msg='正在计算评分 ...')
		self._log('📋 刷新评分...')
		self._refresh_scores()
		self._step_done(1)
		self._log('✅ 评分刷新完成')
		self.root.after(500, lambda: self._busy(False))

	def _refresh_history(self):
		df = load_history()
		if df is None:
			self.hist_var.set('暂无实验记录')
			return
		best = df.loc[df['score'].idxmax()]
		latest = df.iloc[-1]
		self.hist_var.set(
			f'实验次数: {len(df)}\n'
			f'最佳: {best["score"]:.6f} ({best["label"]})\n'
			f'最新: {latest["score"]:.6f} ({latest["label"]})'
		)
		self.footer_var.set(f'共 {len(df)} 次实验')

	# ---- 预测 ----
	def _on_predict(self):
		self._busy(True, step=0, msg='模型预测中 ...')
		self._log('▶ 开始运行 predict.py ...')

		def cb(output, ok):
			for line in output.strip().split('\n')[-6:]:
				if line.strip():
					self._log(line)
			if ok:
				self._step_done(0)
				self._log('✅ 预测完成')
				self._refresh_scores()
			else:
				self._log('❌ 预测失败')
			self.root.after(500, lambda: self._busy(False))

		script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
		threading.Thread(target=lambda: cb(*run_subprocess(script)), daemon=True).start()

	# ---- 一键流水线 ----
	def _on_full_pipeline(self):
		self._busy(True, step=0, msg='① 模型预测中 ...')
		self._log('🚀 一键流水线启动')

		def step2(output, ok):
			for line in output.strip().split('\n')[-6:]:
				if line.strip():
					self._log(line)
			if ok:
				self._step_done(0)
				self._busy(True, step=1, msg='② 生成图表和评分 ...')
				self._log('✅ 预测完成，生成图表和评分 ...')
				chart_script = os.path.join(PROJECT_ROOT, 'test', 'visualize_score.py')

				def chart_cb(cout, cok):
					self._step_done(1)
					self._busy(True, step=2, msg='③ 刷新成绩 ...')
					self._log('📊 图表已生成' if cok else '⚠ 图表生成失败')
					# 标记第三步完成
					self._step_done(2)
					# 短暂显示完成状态
					self.root.after(600, lambda: self._busy(False))
					self._refresh_scores()
					self._log('🎉 流水线完成')
					self._refresh_history()

				threading.Thread(target=lambda: chart_cb(*run_subprocess(chart_script)), daemon=True).start()
			else:
				self._log('❌ 预测失败，流水线中断')
				self._busy(False)

		predict_script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
		threading.Thread(target=lambda: step2(*run_subprocess(predict_script)), daemon=True).start()

	# ---- 图表 ----
	def _on_show_chart(self):
		fp = os.path.join(PROJECT_ROOT, 'test', 'score_comparison.png')
		if os.path.exists(fp):
			os.startfile(fp)
			self._log('📈 已打开对比图表')
		else:
			messagebox.showinfo('提示', '图表尚未生成，请先运行评分。')

	def _on_show_history(self):
		fp = os.path.join(PROJECT_ROOT, 'test', 'score_history.png')
		if os.path.exists(fp):
			os.startfile(fp)
			self._log('📉 已打开历史趋势图')
		else:
			messagebox.showinfo('提示', '历史趋势图尚未生成，请先运行评分。')


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
	root = tk.Tk()
	root.update_idletasks()
	w, h = 1050, 800
	sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
	root.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')
	ScoreApp(root)
	root.mainloop()
