#!/usr/bin/env python3
r"""
成绩对比可视化桌面应用

功能:
- 一键运行预测 → 评分 → 对比展示
- 成绩历史追踪
- 图表可视化
- 无需命令行，全 GUI 操作

用法:
  激活虚拟环境后: python test/score_gui.py
"""
import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from datetime import datetime

# ---------- 路径配置 ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')


def run_script(script_path, callback=None):
	"""在子进程中运行脚本，完成后回调"""
	cmd = [VENV_PYTHON, script_path]
	try:
		result = subprocess.run(
			cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
			encoding='utf-8', errors='replace', timeout=600,
		)
		output = result.stdout + '\n' + result.stderr
		success = result.returncode == 0
	except subprocess.TimeoutExpired:
		output = '⚠ 执行超时 (10分钟)'
		success = False
	except Exception as e:
		output = f'⚠ 执行异常: {e}'
		success = False

	if callback:
		callback(output, success)
	return output, success


# ---------- 主窗口 ----------
class ScoreApp:
	def __init__(self, root):
		self.root = root
		self.root.title('BDC2026 成绩对比工具')
		self.root.geometry('1000x720')
		self.root.minsize(900, 600)

		# 配色方案
		self.colors = {
			'bg': '#1e1e2e',
			'card': '#2a2a3e',
			'accent': '#3b82f6',
			'success': '#22c55e',
			'danger': '#ef4444',
			'warning': '#f59e0b',
			'text': '#e4e4e7',
			'text_secondary': '#a1a1aa',
			'border': '#3f3f5c',
			'button': '#3b82f6',
			'button_hover': '#2563eb',
			'button_text': '#ffffff',
			'log_bg': '#18181b',
		}

		self.root.configure(bg=self.colors['bg'])
		self._setup_ui()
		self._load_history()

	# ---------- UI 布局 ----------
	def _setup_ui(self):
		style = ttk.Style()
		style.theme_use('clam')
		style.configure('TFrame', background=self.colors['bg'])
		style.configure('Card.TFrame', background=self.colors['card'], relief='flat', borderwidth=1)
		style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['text'], font=('Microsoft YaHei UI', 10))
		style.configure('CardTitle.TLabel', background=self.colors['card'], foreground=self.colors['accent'], font=('Microsoft YaHei UI', 12, 'bold'))
		style.configure('Score.TLabel', background=self.colors['card'], foreground=self.colors['success'], font=('Consolas', 20, 'bold'))
		style.configure('TButton', background=self.colors['button'], foreground=self.colors['button_text'], font=('Microsoft YaHei UI', 10), padding=8)
		style.map('TButton', background=[('active', self.colors['button_hover'])])

		# 顶部标题栏
		header = tk.Frame(self.root, bg=self.colors['bg'], pady=15)
		header.pack(fill='x', padx=20)
		tk.Label(header, text='🏆 BDC2026 股票预测成绩对比', font=('Microsoft YaHei UI', 18, 'bold'),
				fg=self.colors['accent'], bg=self.colors['bg']).pack(side='left')

		# 主内容区（左右两列）
		main = tk.Frame(self.root, bg=self.colors['bg'])
		main.pack(fill='both', expand=True, padx=20, pady=(0, 10))

		# === 左列: 操作面板 + 成绩卡片 ===
		left = tk.Frame(main, bg=self.colors['bg'])
		left.pack(side='left', fill='both', expand=True, padx=(0, 10))

		self._build_control_panel(left)
		self._build_score_cards(left)

		# === 右列: 日志 + 图表 ===
		right = tk.Frame(main, bg=self.colors['bg'])
		right.pack(side='right', fill='both', expand=True, padx=(10, 0))

		self._build_log_panel(right)
		self._build_chart_panel(right)

		# 底部状态栏
		self._build_statusbar()

	def _build_control_panel(self, parent):
		card = tk.Frame(parent, bg=self.colors['card'], padx=20, pady=15, highlightbackground=self.colors['border'], highlightthickness=1)
		card.pack(fill='x', pady=(0, 10))

		tk.Label(card, text='操作面板', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.colors['accent'], bg=self.colors['card']).pack(anchor='w', pady=(0, 10))

		# 按钮行
		btn_frame = tk.Frame(card, bg=self.colors['card'])
		btn_frame.pack(fill='x')

		self.btn_predict = self._make_button(btn_frame, '▶  运行预测 (predict.py)', self._on_predict, '#7c3aed')
		self.btn_predict.pack(fill='x', pady=3)

		self.btn_score = self._make_button(btn_frame, '📊 评分 & 对比', self._on_score, self.colors['accent'])
		self.btn_score.pack(fill='x', pady=3)

		self.btn_full = self._make_button(btn_frame, '🚀 一键预测+评分', self._on_full_pipeline, self.colors['success'])
		self.btn_full.pack(fill='x', pady=3)

		self.btn_chart = self._make_button(btn_frame, '📈 显示对比图表', self._on_show_chart, self.colors['warning'])
		self.btn_chart.pack(fill='x', pady=3)

		self.btn_history = self._make_button(btn_frame, '📋 显示历史趋势', self._on_show_history, '#ec4899')
		self.btn_history.pack(fill='x', pady=3)

		# 进度条
		self.progress = ttk.Progressbar(card, mode='indeterminate', length=200)
		self.progress.pack(fill='x', pady=(10, 0))

		# 状态标签
		self.status_label = tk.Label(card, text='就绪', font=('Microsoft YaHei UI', 9),
									fg=self.colors['text_secondary'], bg=self.colors['card'])
		self.status_label.pack(anchor='w', pady=(5, 0))

	def _build_score_cards(self, parent):
		"""成绩对比卡片"""
		card = tk.Frame(parent, bg=self.colors['card'], padx=20, pady=15, highlightbackground=self.colors['border'], highlightthickness=1)
		card.pack(fill='x', pady=(0, 10))

		tk.Label(card, text='成绩对比', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.colors['accent'], bg=self.colors['card']).pack(anchor='w', pady=(0, 10))

		# 两列分数
		score_row = tk.Frame(card, bg=self.colors['card'])
		score_row.pack(fill='x')

		# 基准程序
		base_col = tk.Frame(score_row, bg=self.colors['card'])
		base_col.pack(side='left', expand=True, fill='both', padx=(0, 5))
		tk.Label(base_col, text='基准程序', font=('Microsoft YaHei UI', 10),
				fg=self.colors['text_secondary'], bg=self.colors['card']).pack(anchor='center')
		self.baseline_score_var = tk.StringVar(value='---')
		bl = tk.Label(base_col, textvariable=self.baseline_score_var, font=('Consolas', 24, 'bold'),
				fg='#888888', bg=self.colors['card'])
		bl.pack(anchor='center')

		# 分隔线
		tk.Frame(score_row, bg=self.colors['border'], width=1).pack(side='left', fill='y', padx=8)

		# 当前系统
		curr_col = tk.Frame(score_row, bg=self.colors['card'])
		curr_col.pack(side='left', expand=True, fill='both', padx=(5, 0))
		tk.Label(curr_col, text='当前系统', font=('Microsoft YaHei UI', 10),
				fg=self.colors['text_secondary'], bg=self.colors['card']).pack(anchor='center')
		self.current_score_var = tk.StringVar(value='---')
		cl = tk.Label(curr_col, textvariable=self.current_score_var, font=('Consolas', 24, 'bold'),
				fg=self.colors['success'], bg=self.colors['card'])
		cl.pack(anchor='center')

		# 差值 + 状态
		diff_frame = tk.Frame(card, bg=self.colors['card'], pady=8)
		diff_frame.pack(fill='x')
		self.diff_var = tk.StringVar(value='等待评分...')
		tk.Label(diff_frame, textvariable=self.diff_var, font=('Microsoft YaHei UI', 11, 'bold'),
				fg=self.colors['accent'], bg=self.colors['card']).pack(anchor='center')

		# 历史摘要
		self.history_summary_var = tk.StringVar(value='')
		hs = tk.Label(card, textvariable=self.history_summary_var, font=('Microsoft YaHei UI', 9),
					fg=self.colors['text_secondary'], bg=self.colors['card'], wraplength=400, justify='center')
		hs.pack(anchor='center', pady=(5, 0))

	def _build_log_panel(self, parent):
		card = tk.Frame(parent, bg=self.colors['card'], padx=15, pady=12, highlightbackground=self.colors['border'], highlightthickness=1)
		card.pack(fill='both', expand=True, pady=(0, 10))

		header = tk.Frame(card, bg=self.colors['card'])
		header.pack(fill='x', pady=(0, 8))
		tk.Label(header, text='运行日志', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.colors['accent'], bg=self.colors['card']).pack(side='left')
		tk.Button(header, text='清空', font=('Microsoft YaHei UI', 9),
				bg=self.colors['button'], fg=self.colors['button_text'], relief='flat',
				command=self._clear_log,
				activebackground=self.colors['button_hover'], activeforeground='white',
				cursor='hand2', padx=10).pack(side='right')

		self.log = scrolledtext.ScrolledText(
			card, bg=self.colors['log_bg'], fg='#a0ffa0', insertbackground='white',
			font=('Consolas', 9), wrap='word', relief='flat', borderwidth=0,
			height=10,
		)
		self.log.pack(fill='both', expand=True)
		self.log.insert('end', '应用已启动。点击操作面板按钮开始。\n')
		self.log.see('end')

	def _build_chart_panel(self, parent):
		card = tk.Frame(parent, bg=self.colors['card'], padx=15, pady=12, highlightbackground=self.colors['border'], highlightthickness=1)
		card.pack(fill='x')

		tk.Label(card, text='图表预览', font=('Microsoft YaHei UI', 12, 'bold'),
				fg=self.colors['accent'], bg=self.colors['card']).pack(anchor='w', pady=(0, 8))

		# 图表路径显示
		self.chart_path_var = tk.StringVar(value='')
		self.chart_label = tk.Label(card, textvariable=self.chart_path_var, font=('Microsoft YaHei UI', 9),
									fg=self.colors['text_secondary'], bg=self.colors['card'], wraplength=380, justify='left')
		self.chart_label.pack(anchor='w')

		# 使用 PIL 显示图表缩略图（如果 PIL 可用）
		self.chart_thumb = tk.Label(card, bg=self.colors['card'], text='点击下方按钮生成图表')
		self.chart_thumb.pack(pady=5)

	def _build_statusbar(self):
		bar = tk.Frame(self.root, bg=self.colors['border'], height=28)
		bar.pack(fill='x', side='bottom')
		self.footer_var = tk.StringVar(value='数据: data/train.csv | 测试: data/test.csv')
		tk.Label(bar, textvariable=self.footer_var, font=('Microsoft YaHei UI', 8),
				fg=self.colors['text_secondary'], bg=self.colors['border']).pack(side='left', padx=15)

	def _make_button(self, parent, text, command, color):
		"""自定义着色按钮"""
		btn = tk.Button(parent, text=text, font=('Microsoft YaHei UI', 10, 'bold'),
						bg=color, fg='white', relief='flat', padx=12, pady=8,
						command=command, cursor='hand2',
						activebackground=color, activeforeground='white',
						bd=0)
		# hover 效果
		def on_enter(e):
			e.widget.configure(bg=self._darken(color))
		def on_leave(e):
			e.widget.configure(bg=color)
		btn.bind('<Enter>', on_enter)
		btn.bind('<Leave>', on_leave)
		return btn

	@staticmethod
	def _darken(hex_color, factor=0.8):
		"""加深颜色"""
		hex_color = hex_color.lstrip('#')
		r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:], 16)
		r, g, b = int(r*factor), int(g*factor), int(b*factor)
		return f'#{r:02x}{g:02x}{b:02x}'

	# ---------- 动作方法 ----------
	def _log(self, text, tag=None):
		self.log.insert('end', f'[{datetime.now().strftime("%H:%M:%S")}] {text}\n')
		self.log.see('end')
		self.root.update_idletasks()

	def _set_busy(self, busy=True):
		for btn in [self.btn_predict, self.btn_score, self.btn_full, self.btn_chart, self.btn_history]:
			btn.configure(state='disabled' if busy else 'normal')
		if busy:
			self.progress.start(10)
		else:
			self.progress.stop()
		self.root.update_idletasks()

	def _on_predict(self):
		self._set_busy(True)
		self._log('▶ 开始运行 predict.py ...')
		self.status_label.configure(text='正在预测...')

		def callback(output, success):
			self._log(output.split('\n')[-5:], 'result')
			if success:
				self._log('✓ 预测完成', 'success')
				self.status_label.configure(text='预测完成')
			else:
				self._log('✗ 预测失败', 'error')
				self.status_label.configure(text='预测失败')
			self._set_busy(False)

		script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
		threading.Thread(target=run_script, args=(script, callback), daemon=True).start()

	def _on_score(self):
		self._set_busy(True)
		self._log('📊 开始评分...')
		self.status_label.configure(text='正在评分...')

		script = os.path.join(PROJECT_ROOT, 'test', 'visualize_score.py')

		def callback(output, success):
			self._log(output)
			if success:
				self._parse_score_output(output)
				self.status_label.configure(text='评分完成')
			else:
				self.status_label.configure(text='评分失败')
			self._set_busy(False)

		threading.Thread(target=run_script, args=(script, callback), daemon=True).start()

	def _on_full_pipeline(self):
		self._set_busy(True)
		self._log('🚀 开始一键流水线...')
		self.status_label.configure(text='Step 1/2: 预测中...')

		def predict_callback(output, success):
			self._log(output.split('\n')[-10:])
			if success:
				self._log('✓ 预测完成，开始评分...')
				self.status_label.configure(text='Step 2/2: 评分中...')
				score_script = os.path.join(PROJECT_ROOT, 'test', 'visualize_score.py')
				run_script(score_script, score_callback)
			else:
				self._log('✗ 预测失败，流水线中断', 'error')
				self.status_label.configure(text='流水线失败')
				self._set_busy(False)

		def score_callback(output, success):
			self._log(output)
			if success:
				self._parse_score_output(output)
				self.status_label.configure(text='流水线完成')
			else:
				self.status_label.configure(text='评分失败')
			self._set_busy(False)

		predict_script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
		threading.Thread(target=run_script, args=(predict_script, predict_callback), daemon=True).start()

	def _on_show_chart(self):
		chart_path = os.path.join(PROJECT_ROOT, 'test', 'score_comparison.png')
		if os.path.exists(chart_path):
			os.startfile(chart_path)
			self._log(f'📈 已打开对比图: {chart_path}')
			self.chart_path_var.set(f'已生成: {chart_path}')
		else:
			self._log('⚠ 图表尚未生成，请先运行评分')
			messagebox.showinfo('提示', '请先点击"评分 & 对比"生成图表。')

	def _on_show_history(self):
		hist_path = os.path.join(PROJECT_ROOT, 'test', 'score_history.png')
		hist_csv = os.path.join(PROJECT_ROOT, 'test', 'score_history.csv')
		if os.path.exists(hist_path):
			os.startfile(hist_path)
			self._log(f'📋 已打开历史趋势图')
		elif os.path.exists(hist_csv):
			# 先生成趋势图再打开
			script = os.path.join(PROJECT_ROOT, 'test', 'visualize_score.py')
			self._set_busy(True)
			self._log('正在生成历史趋势图...')
			def callback(_, __):
				if os.path.exists(hist_path):
					os.startfile(hist_path)
				self._set_busy(False)
			run_script(script, ['--view'])
			if os.path.exists(hist_path):
				os.startfile(hist_path)
			self._set_busy(False)
		else:
			self._log('⚠ 历史记录为空，请先运行评分生成记录')

	def _clear_log(self):
		self.log.delete('1.0', 'end')

	# ---------- 数据解析 ----------
	def _parse_score_output(self, output):
		"""从 visualize_score.py 输出中提取分数"""
		lines = output.split('\n')
		baseline, current = None, None
		for line in lines:
			if '基准程序得分' in line or '基準程序得分' in line:
				try:
					baseline = float(line.split()[-1])
				except ValueError:
					pass
			if '当前系统得分' in line or '當前系統得分' in line:
				try:
					current = float(line.split()[-1])
				except ValueError:
					pass
			if '已超越基准' in line or '已超越基準' in line:
				self.diff_var.set('👍 已超越基准')
			if '未达基准' in line or '未達基準' in line:
				self.diff_var.set('⚠ 未达基准，需继续优化')

		if baseline is not None:
			self.baseline_score_var.set(f'{baseline:.6f}')
		if current is not None:
			self.current_score_var.set(f'{current:.6f}')
		if baseline is not None and current is not None:
			diff = current - baseline
			improve = (diff / abs(baseline) * 100) if baseline != 0 else float('inf')
			self.diff_var.set(f'差值: {diff:+.6f} | 提升: {improve:+.2f}%')

		self._load_history()
		self.chart_path_var.set('test/score_comparison.png')

	def _load_history(self):
		"""加载历史记录摘要"""
		hist_csv = os.path.join(PROJECT_ROOT, 'test', 'score_history.csv')
		if not os.path.exists(hist_csv):
			self.history_summary_var.set('暂无历史记录')
			return

		try:
			import pandas as pd
			df = pd.read_csv(hist_csv)
			if len(df) == 0:
				self.history_summary_var.set('暂无历史记录')
				return

			best_idx = df['score'].idxmax()
			best_score = df.loc[best_idx, 'score']
			best_label = df.loc[best_idx, 'label']
			latest_score = df.iloc[-1]['score']

			summary = (
				f'📊 共 {len(df)} 次实验 | '
				f'最佳: {best_score:.6f} ({best_label}) | '
				f'最新: {latest_score:.6f}'
			)
			self.history_summary_var.set(summary)
			self.footer_var.set(
				f'数据: data/train.csv | 测试: data/test.csv | 实验次数: {len(df)}'
			)
		except Exception:
			self.history_summary_var.set('历史记录加载失败')


# ---------- 入口 ----------
if __name__ == '__main__':
	root = tk.Tk()

	# 居中显示
	root.update_idletasks()
	w, h = 1000, 720
	sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
	x, y = (sw - w) // 2, (sh - h) // 2
	root.geometry(f'{w}x{h}+{x}+{y}')

	app = ScoreApp(root)
	root.mainloop()
