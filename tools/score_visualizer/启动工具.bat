@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 正在启动 BDC2026 成绩对比工具 v2.0...
start "" ".venv\Scripts\pythonw.exe" "test\score_gui.py"
