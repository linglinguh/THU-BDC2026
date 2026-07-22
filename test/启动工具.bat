@echo off
chcp 65001 >nul
cd /d "%~dp0.."
start "BDC2026 成绩对比工具" ".venv\Scripts\pythonw.exe" test\score_gui.py
