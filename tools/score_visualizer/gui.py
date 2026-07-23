#!/usr/bin/env python3
r"""成绩对比可视化桌面应用 v3.0

用法: python tools/score_visualizer/gui.py
"""
import os, sys, subprocess, threading, time, re, json
import tkinter as tk
from tkinter import messagebox, scrolledtext
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PYTHON = os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe')


# ============================================================
# 评分引擎
# ============================================================
def _stock_return(group):
    start, end = group.iloc[0], group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']


def compute_scores():
    import pandas as pd
    result_path = os.path.join(PROJECT_ROOT, 'output', 'result.csv')
    test_path = os.path.join(PROJECT_ROOT, 'data', 'test.csv')
    baseline_result_path = os.path.join(PROJECT_ROOT, 'test', 'baseline_result.csv')
    if not os.path.exists(result_path) or not os.path.exists(test_path):
        return None

    output_df = pd.read_csv(result_path)
    test_data = pd.read_csv(test_path, dtype={'股票代码': str})
    test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)
    output_data = output_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
    output_data['股票代码'] = output_data['股票代码'].astype(str).str.zfill(6)

    def calc_score(out_df):
        if len(out_df) > 5 or not (0 <= float(out_df['权重'].sum()) <= 1.0):
            return (-999.0, None)
        f = test_data[test_data['股票代码'].isin(out_df['股票代码'])]
        f = f.groupby('股票代码').tail(5)
        if len(f) == 0:
            return (-999.0, None)
        records = []
        for sid, grp in f.groupby('股票代码'):
            grp = grp.sort_values('日期')
            ret = _stock_return(grp)
            records.append({'股票代码': str(sid).zfill(6), '收益率': ret})
        ret_df = pd.DataFrame(records)
        ret_df = ret_df.merge(out_df[['股票代码', '权重']], on='股票代码')
        score = float((ret_df['收益率'] * ret_df['权重']).sum())
        ret_df['贡献分'] = ret_df['收益率'] * ret_df['权重']
        return (score, ret_df)

    current_score, stock_detail = calc_score(output_data)

    if os.path.exists(baseline_result_path):
        bl_df = pd.read_csv(baseline_result_path)
        bl_data = bl_df.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
        bl_data['股票代码'] = bl_data['股票代码'].astype(str).str.zfill(6)
        baseline_score, _ = calc_score(bl_data)
    else:
        all_stocks = sorted(test_data['股票代码'].unique())
        if len(all_stocks) < 5:
            baseline_score = 0.0
        else:
            bl_data = pd.DataFrame({'股票代码': all_stocks[:5], '权重': [0.2]*5})
            baseline_score, _ = calc_score(bl_data)

    return {
        'current': current_score, 'baseline': baseline_score,
        'diff': current_score - baseline_score,
        'improve_pct': (current_score - baseline_score) / abs(baseline_score) * 100 if baseline_score != 0 else 0.0,
        'exceed': current_score > baseline_score,
        'stock_detail': stock_detail,
    }


def load_history():
    import pandas as pd
    hist_csv = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'history.csv')
    if not os.path.exists(hist_csv):
        return None
    df = pd.read_csv(hist_csv)
    return df if len(df) > 0 else None


def compute_stock_detail():
    import pandas as pd
    result_path = os.path.join(PROJECT_ROOT, 'output', 'result.csv')
    if not os.path.exists(result_path):
        return None
    output_df = pd.read_csv(result_path)
    return output_df


def get_industry_for_stock(code):
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code', 'src'))
        from industry_map import get_industry
        return get_industry(code)
    except Exception:
        return '--'


def clean_log_output(text):
    cleaned_lines = []
    for line in text.split('\n'):
        if '\r' in line:
            line = line.split('\r')[-1]
        if re.search(r'\d+%\|\S*\|', line) or re.search(r'\d+\.\d+it/s', line):
            continue
        if 'FutureWarning' in line or 'DeprecationWarning' in line:
            continue
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)
        line = line.strip()
        if not line:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def run_subprocess(script_path, hide_progress=True):
    cmd = [VENV_PYTHON, script_path]
    env = os.environ.copy()
    if hide_progress:
        env['TQDM_DISABLE'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=600, env=env)
        cleaned = clean_log_output(result.stdout + '\n' + result.stderr)
        return (cleaned if cleaned else '(无输出)', result.returncode == 0)
    except subprocess.TimeoutExpired:
        return ('超时 (>10分钟)', False)
    except Exception as e:
        return (str(e), False)


# ============================================================
# 迷你趋势图组件
# ============================================================
class Sparkline(tk.Canvas):
    def __init__(self, parent, colors, **kw):
        kw.setdefault('height', 50)
        kw.setdefault('bg', colors['card'])
        kw.setdefault('highlightthickness', 0)
        super().__init__(parent, **kw)
        self.colors = colors

    def set_data(self, scores):
        self.delete('all')
        if not scores or len(scores) < 2:
            return
        w, h = self.winfo_width() or 200, 50
        n = len(scores)
        mn, mx = min(scores), max(scores)
        rng = max(mx - mn, 1e-9)
        pts = []
        for i, v in enumerate(scores):
            x = 5 + i * (w - 10) / (n - 1)
            y = 5 + (mx - v) / rng * (h - 10)
            pts.extend([x, y])
        if len(pts) >= 4:
            self.create_line(pts, fill=self.colors['primary'], width=2, smooth=True)
            self.create_oval(pts[-2] - 3, pts[-1] - 3, pts[-2] + 3, pts[-1] + 3,
                             fill=self.colors['success'], outline='')
        # 基线
        base_y = 5 + (mx - 0) / rng * (h - 10) if mx > 0 > mn else h - 5
        self.create_line(5, base_y, w - 5, base_y, fill=self.colors['text3'], dash=(4, 4))


# ============================================================
# 步骤进度条
# ============================================================
class StepProgress(tk.Frame):
    CIRCLE_R = 12
    GAP = 80

    def __init__(self, parent, steps, colors):
        super().__init__(parent, bg=colors['card'])
        self.colors = colors
        self.steps = steps
        self.statuses = ['pending'] * len(steps)
        self.canvas = tk.Canvas(self, bg=colors['card'], highlightthickness=0,
                                height=70, width=len(steps) * self.GAP + 40)
        self.canvas.pack()
        self._draw()

    def set_step(self, index, status):
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
        x0, y = 30, 30
        for i in range(n):
            x = x0 + i * self.GAP
            if i < n - 1:
                nx = x0 + (i + 1) * self.GAP
                lc = self.colors['success'] if self.statuses[i] == 'done' else self.colors['border']
                c.create_line(x + self.CIRCLE_R, y, nx - self.CIRCLE_R, y, fill=lc, width=3)
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
            lbl_color = self.colors['text'] if st != 'pending' else self.colors['text3']
            c.create_text(x + self.CIRCLE_R, y + 28, text=self.steps[i],
                          fill=lbl_color, font=('Microsoft YaHei UI', 8), anchor='center')


# ============================================================
# 主窗口
# ============================================================
class ScoreApp:
    def __init__(self, root):
        self.root = root
        self.root.title('BDC2026 成绩对比工具')
        self.root.geometry('1110x830')
        self.root.minsize(1000, 700)

        self.C = {
            'bg': '#0b1120',
            'card': '#131c31',
            'primary': '#6366f1', 'primary_hover': '#4f46e5',
            'success': '#10b981', 'danger': '#ef4444',
            'warning': '#f59e0b', 'info': '#3b82f6',
            'text': '#f1f5f9', 'text2': '#94a3b8', 'text3': '#64748b',
            'border': '#1e2d4a', 'log_bg': '#0a0f1c', 'log_fg': '#86efac',
            'purple': '#8b5cf6', 'pink': '#ec4899',
        }
        self.op_start = 0  # 操作计时

        self.root.configure(bg=self.C['bg'])
        self._build()
        self._refresh_scores()
        self._update_clock()

    # ============================ 构建 ============================
    def _build(self):
        hdr = tk.Frame(self.root, bg=self.C['bg'], height=50)
        hdr.pack(fill='x', padx=20, pady=(12, 6))
        hdr.pack_propagate(False)
        tk.Label(hdr, text='📈', font=('Segoe UI Emoji', 22), bg=self.C['bg']).pack(side='left', padx=(0, 8))
        tk.Label(hdr, text='BDC2026 成绩对比', font=('Microsoft YaHei UI', 16, 'bold'),
                 fg=self.C['text'], bg=self.C['bg']).pack(side='left')
        self.version_label = tk.Label(hdr, text='v3.0', font=('Consolas', 9), fg=self.C['text3'], bg=self.C['bg'])
        self.version_label.pack(side='right', padx=5)

        body = tk.Frame(self.root, bg=self.C['bg'])
        body.pack(fill='both', expand=True, padx=20, pady=(0, 6))

        # 左: 分数 + 股票明细
        left = tk.Frame(body, bg=self.C['bg'], width=370)
        left.pack(side='left', fill='y', padx=(0, 8))
        left.pack_propagate(False)
        self._build_score_panel(left)

        # 中: 操作 + 配置
        mid = tk.Frame(body, bg=self.C['bg'], width=250)
        mid.pack(side='left', fill='y', padx=(0, 8))
        mid.pack_propagate(False)
        self._build_control_panel(mid)

        # 右: 日志
        right = tk.Frame(body, bg=self.C['bg'])
        right.pack(side='right', fill='both', expand=True)
        self._build_log_panel(right)

        # 底部栏
        bar = tk.Frame(self.root, bg=self.C['border'], height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        self.footer_var = tk.StringVar(value='就绪')
        tk.Label(bar, textvariable=self.footer_var, font=('Microsoft YaHei UI', 8),
                 fg=self.C['text2'], bg=self.C['border']).pack(side='left', padx=15)
        self.clock_var = tk.StringVar()
        tk.Label(bar, textvariable=self.clock_var, font=('Consolas', 8),
                 fg=self.C['text3'], bg=self.C['border']).pack(side='right', padx=15)

    def _card(self, p, **kw):
        f = tk.Frame(p, bg=self.C['card'], padx=14, pady=12,
                     highlightbackground=self.C['border'], highlightthickness=1)
        f.pack(fill='x', pady=(0, 8), **kw)
        return f

    def _btn(self, parent, text, command, color=None, size=10):
        c = color or self.C['primary']
        h = self._darken(c)
        btn = tk.Button(parent, text=text, font=('Microsoft YaHei UI', size, 'bold'),
                        bg=c, fg='white', relief='flat', padx=12, pady=9,
                        command=command, cursor='hand2', bd=0, activebackground=h, activeforeground='white')
        btn.bind('<Enter>', lambda e: btn.configure(bg=h))
        btn.bind('<Leave>', lambda e: btn.configure(bg=c))
        return btn

    @staticmethod
    def _darken(hex_color, factor=0.78):
        h = hex_color.lstrip('#')
        r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
        return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'

    # ==================== 左栏: 分数 + 股票 ====================
    def _build_score_panel(self, parent):
        # 当前分
        c = self._card(parent)
        tk.Label(c, text='当前系统得分', font=('Microsoft YaHei UI', 10), fg=self.C['text2'], bg=self.C['card']).pack(anchor='w')
        r1 = tk.Frame(c, bg=self.C['card'])
        r1.pack(fill='x', pady=(4, 0))
        self.current_var = tk.StringVar(value='---')
        tk.Label(r1, textvariable=self.current_var, font=('Consolas', 34, 'bold'),
                 fg=self.C['success'], bg=self.C['card']).pack(side='left')
        self.badge = tk.Label(c, text='等待评分', font=('Microsoft YaHei UI', 9, 'bold'),
                              padx=8, pady=2, bg=self.C['text3'], fg='white')
        self.badge.pack(side='right', anchor='e', pady=(0, 6))

        # 迷你趋势
        self.spark = Sparkline(c, self.C, width=330, height=46, bg=self.C['card'])
        self.spark.pack(fill='x', pady=(6, 0))

        tk.Frame(c, bg=self.C['border'], height=1).pack(fill='x', pady=(8, 6))

        # 基准分 + 差值
        tk.Label(c, text='基准程序得分', font=('Microsoft YaHei UI', 9), fg=self.C['text3'], bg=self.C['card']).pack(anchor='w')
        self.baseline_var = tk.StringVar(value='---')
        tk.Label(c, textvariable=self.baseline_var, font=('Consolas', 24, 'bold'),
                 fg=self.C['text3'], bg=self.C['card']).pack(anchor='w')
        self.diff_var = tk.StringVar(value='')
        tk.Label(c, textvariable=self.diff_var, font=('Microsoft YaHei UI', 11, 'bold'),
                 fg=self.C['primary'], bg=self.C['card']).pack(anchor='w', pady=(6, 0))

        # 历史摘要
        hc = self._card(parent)
        tk.Label(hc, text='实验历史', font=('Microsoft YaHei UI', 11, 'bold'), fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 4))
        self.hist_var = tk.StringVar(value='暂无记录')
        tk.Label(hc, textvariable=self.hist_var, font=('Microsoft YaHei UI', 9),
                 fg=self.C['text2'], bg=self.C['card'], wraplength=320, justify='left').pack(anchor='w')

        # 预测股票明细
        sc = self._card(parent)
        tk.Label(sc, text='最新预测股票', font=('Microsoft YaHei UI', 11, 'bold'), fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 4))
        self.stock_table = tk.Text(sc, bg=self.C['log_bg'], fg=self.C['log_fg'],
                                   font=('Consolas', 9), height=6, relief='flat', borderwidth=0,
                                   wrap='none', cursor='arrow')
        self.stock_table.pack(fill='x')
        self.stock_table.insert('1.0', '  (运行预测后显示)')
        self.stock_table.configure(state='disabled')

    # ==================== 中栏 ====================
    def _build_control_panel(self, parent):
        # 操作按钮
        c = self._card(parent)
        tk.Label(c, text='操作', font=('Microsoft YaHei UI', 11, 'bold'), fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 8))
        self.btn_full = self._btn(c, '🚀 一键预测 + 评分', self._on_full_pipeline, self.C['success'], 11)
        self.btn_full.pack(fill='x', pady=3)
        self.btn_predict = self._btn(c, '📊 仅运行预测', self._on_predict, self.C['purple'])
        self.btn_predict.pack(fill='x', pady=3)
        self.btn_score = self._btn(c, '📋 仅评分', self._on_refresh_score, self.C['info'])
        self.btn_score.pack(fill='x', pady=3)
        tk.Frame(c, bg=self.C['border'], height=1).pack(fill='x', pady=(6, 4))
        self.btn_chart = self._btn(c, '📈 对比图表', self._on_show_chart, self.C['warning'], 9)
        self.btn_chart.pack(fill='x', pady=2)
        self.btn_history = self._btn(c, '📉 历史趋势', self._on_show_history, self.C['pink'], 9)
        self.btn_history.pack(fill='x', pady=2)

        # 进度
        pc = self._card(parent)
        self.stepper = StepProgress(pc, ['预测', '评分', '图表'], self.C)
        self.stepper.pack()
        self.progress_var = tk.StringVar(value='空闲')
        tk.Label(pc, textvariable=self.progress_var, font=('Microsoft YaHei UI', 9, 'bold'),
                 fg=self.C['primary'], bg=self.C['card']).pack(anchor='center', pady=(4, 0))
        self.timer_var = tk.StringVar(value='')
        tk.Label(pc, textvariable=self.timer_var, font=('Consolas', 9),
                 fg=self.C['text3'], bg=self.C['card']).pack(anchor='center')

        # 配置开关
        cc = self._card(parent)
        tk.Label(cc, text='推理后处理', font=('Microsoft YaHei UI', 11, 'bold'), fg=self.C['text'], bg=self.C['card']).pack(anchor='w', pady=(0, 6))

        self.cfg_vars = {}
        for key, label in [('enable_multi_factor', '多因子评分'), ('enable_industry_diversify', '行业分散')]:
            v = tk.BooleanVar(value=self._read_config(key))
            self.cfg_vars[key] = v
            cb = tk.Checkbutton(cc, text=label, variable=v, font=('Microsoft YaHei UI', 9),
                                bg=self.C['card'], fg=self.C['text2'],
                                selectcolor=self.C['card'], activebackground=self.C['card'],
                                activeforeground=self.C['text'], command=self._on_config_save)
            cb.pack(anchor='w')

    # ==================== 右栏 ====================
    def _build_log_panel(self, parent):
        c = tk.Frame(parent, bg=self.C['card'], padx=14, pady=12,
                     highlightbackground=self.C['border'], highlightthickness=1)
        c.pack(fill='both', expand=True)
        hh = tk.Frame(c, bg=self.C['card'])
        hh.pack(fill='x', pady=(0, 6))
        tk.Label(hh, text='运行日志', font=('Microsoft YaHei UI', 11, 'bold'), fg=self.C['text'], bg=self.C['card']).pack(side='left')
        tk.Button(hh, text='清空', font=('Microsoft YaHei UI', 8), bg=self.C['primary'], fg='white',
                  relief='flat', command=self._clear_log, cursor='hand2', padx=8,
                  activebackground=self.C['primary_hover']).pack(side='right')

        self.log = scrolledtext.ScrolledText(
            c, bg=self.C['log_bg'], fg=self.C['log_fg'], insertbackground='white',
            font=('Consolas', 9), wrap='word', relief='flat', borderwidth=0, height=6)
        self.log.pack(fill='both', expand=True)
        self._log_raw('应用 v3.0 已启动 ✓')

    # ==================== 配置读写 ====================
    def _read_config(self, key):
        try:
            config_path = os.path.join(PROJECT_ROOT, 'code', 'src', 'config.py')
            txt = open(config_path, encoding='utf-8').read()
            m = re.search(r"'{}'\s*:\s*(True|False)".format(key), txt)
            return m.group(1) == 'True' if m else False
        except Exception:
            return False

    def _on_config_save(self):
        config_path = os.path.join(PROJECT_ROOT, 'code', 'src', 'config.py')
        txt = open(config_path, encoding='utf-8').read()
        for key, var in self.cfg_vars.items():
            txt = re.sub(r"('{}'\s*:\s*)(True|False)".format(key),
                         r"\1{}".format(str(var.get())), txt)
        open(config_path, 'w', encoding='utf-8').write(txt)
        self._log(f'⚙ 配置已更新: {", ".join(k+"="+str(v.get()) for k,v in self.cfg_vars.items())}')

    # ==================== 时钟和计时 ====================
    def _update_clock(self):
        self.clock_var.set(datetime.now().strftime('%H:%M:%S'))
        if self.op_start > 0:
            elapsed = int(time.time() - self.op_start)
            self.timer_var.set(f'⏱ {elapsed//60}:{elapsed%60:02d}')
        self.root.after(1000, self._update_clock)

    def _start_timer(self):
        self.op_start = time.time()

    def _stop_timer(self):
        self.op_start = 0
        self.timer_var.set('')

    # ==================== 日志 ====================
    def _log_raw(self, text):
        self.log.insert('end', f'[{datetime.now().strftime("%H:%M:%S")}] {text}\n')
        self.log.see('end')
        self.root.update_idletasks()

    def _log(self, text):
        self._log_raw(text)

    def _clear_log(self):
        self.log.delete('1.0', 'end')

    def _busy(self, on=True, step=None, msg=''):
        btns = [self.btn_full, self.btn_predict, self.btn_score, self.btn_chart, self.btn_history]
        for b in btns:
            b.configure(state='disabled' if on else 'normal')
        if on:
            self.stepper.reset()
            self.progress_var.set(msg)
            self._start_timer()
            if step is not None:
                self.stepper.set_step(step, 'active')
        else:
            self.stepper.reset()
            self.progress_var.set('空闲')
            self._stop_timer()
        self.root.update_idletasks()

    def _step_done(self, index):
        self.stepper.set_step(index, 'done')

    # ==================== 刷新 ====================
    def _refresh_scores(self):
        res = compute_scores()
        if res is None:
            self.current_var.set('---')
            self.baseline_var.set('---')
            self.diff_var.set('请先运行预测生成 result.csv')
            self.badge.configure(bg=self.C['text3'], text='等待评分')
            self._update_stock_table(None)
            return

        self.current_var.set(f'{res["current"]:.6f}')
        self.baseline_var.set(f'{res["baseline"]:.6f}')
        self.diff_var.set(f'差值: {res["diff"]:+.6f}  提升: {res["improve_pct"]:+.1f}%')
        self.badge.configure(bg=self.C['success'] if res['exceed'] else self.C['danger'],
                             text='✓ 超越基准' if res['exceed'] else '✗ 未达基准')

        self._update_stock_table(res.get('stock_detail'))
        self._refresh_history()
        self._refresh_sparkline()

    def _update_stock_table(self, detail_df):
        self.stock_table.configure(state='normal')
        self.stock_table.delete('1.0', 'end')
        if detail_df is None or detail_df.empty:
            output_df = compute_stock_detail()
            if output_df is not None and len(output_df) > 0:
                lines = [f'  {"股票":<8} {"行业":<8} {"权重":>8}']
                lines.append(f'  {"-"*30}')
                for _, row in output_df.iterrows():
                    code = str(row['stock_id']).zfill(6)
                    ind = get_industry_for_stock(code)
                    lines.append(f'  {code:<8} {ind:<8} {row["weight"]:>8.4f}')
                self.stock_table.insert('1.0', '\n'.join(lines))
            else:
                self.stock_table.insert('1.0', '  (暂无预测数据)')
        else:
            lines = [f'  {"股票":<8} {"行业":<8} {"收益":>8} {"权重":>8}']
            lines.append(f'  {"-"*38}')
            for _, row in detail_df.iterrows():
                code = str(row['股票代码']).zfill(6)
                ind = get_industry_for_stock(code)
                lines.append(f'  {code:<8} {ind:<8} {row["收益率"]:>7.2%} {row["权重"]:>8.4f}')
            self.stock_table.insert('1.0', '\n'.join(lines))
        self.stock_table.configure(state='disabled')

    def _refresh_sparkline(self):
        df = load_history()
        if df is not None and len(df) >= 2:
            self.spark.set_data(df['score'].tolist())

    def _refresh_history(self):
        df = load_history()
        if df is None:
            self.hist_var.set('暂无实验记录')
            self.footer_var.set('就绪')
            return
        best = df.loc[df['score'].idxmax()]
        latest = df.iloc[-1]
        self.hist_var.set(f'实验 {len(df)} 次 | 最佳: {best["score"]:.6f} | 最新: {latest["score"]:.6f}')
        self.footer_var.set(f'共 {len(df)} 次实验 | 最佳: {best["score"]:.6f}')

    def _on_refresh_score(self):
        self._busy(True, step=1, msg='计算评分 ...')
        self._log('📋 刷新评分')
        self._refresh_scores()
        self._step_done(1)
        self._root_after(500, lambda: self._busy(False))
        self._log('✅ 刷新完成')

    def _root_after(self, ms, fn):
        self.root.after(ms, fn)

    # ==================== 预测 ====================
    def _on_predict(self):
        self._busy(True, step=0, msg='预测中 ...')
        self._log('▶ predict.py ...')

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
            self._root_after(500, lambda: self._busy(False))

        script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
        threading.Thread(target=lambda: cb(*run_subprocess(script)), daemon=True).start()

    # ==================== 一键流水线 ====================
    def _on_full_pipeline(self):
        self._busy(True, step=0, msg='① 模型预测 ...')
        self._log('🚀 一键流水线启动')

        def step2(output, ok):
            for line in output.strip().split('\n')[-6:]:
                if line.strip():
                    self._log(line)
            if ok:
                self._step_done(0)
                self._busy(True, step=1, msg='② 图表和评分 ...')
                self._log('✅ 预测完成 → 生成图表')
                chart_script = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'visualize.py')

                def chart_cb(cout, cok):
                    self._step_done(1)
                    self._busy(True, step=2, msg='③ 刷新成绩 ...')
                    if cok:
                        self._log('📊 图表已生成')
                    else:
                        self._log('⚠ 图表生成失败:')
                        for line in cout.split('\n')[:6]:
                            if line.strip():
                                self._log(f'  {line}')
                    self._step_done(2)
                    self._root_after(600, lambda: self._busy(False))
                    self._refresh_scores()
                    self._log('🎉 流水线完成')
                    self._refresh_history()
                    self._refresh_sparkline()

                threading.Thread(target=lambda: chart_cb(*run_subprocess(chart_script)), daemon=True).start()
            else:
                self._log('❌ 预测失败，中断')
                self._busy(False)

        predict_script = os.path.join(PROJECT_ROOT, 'code', 'src', 'predict.py')
        threading.Thread(target=lambda: step2(*run_subprocess(predict_script)), daemon=True).start()

    # ==================== 图表 ====================
    def _on_show_chart(self):
        fp = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'comparison.png')
        if os.path.exists(fp):
            os.startfile(fp)
            self._log('📈 打开对比图表')
        else:
            messagebox.showinfo('提示', '图表尚未生成，请先运行评分。')

    def _on_show_history(self):
        fp = os.path.join(PROJECT_ROOT, 'tools', 'score_visualizer', 'history.png')
        if os.path.exists(fp):
            os.startfile(fp)
            self._log('📉 打开历史趋势')
        else:
            messagebox.showinfo('提示', '历史趋势图尚未生成，请先运行评分。')


if __name__ == '__main__':
    root = tk.Tk()
    w, h = 1110, 830
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f'{w}x{h}+{(sw-w)//2}+{(sh-h)//2}')
    ScoreApp(root)
    root.mainloop()
