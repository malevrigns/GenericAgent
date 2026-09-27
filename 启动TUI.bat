@echo off
cd /d %~dp0
rem 自部署模型走内网/平台映射地址，必须绕过本地代理（Clash 7897 到不了会 502）
set "NO_PROXY=localhost,127.0.0.1,::1,172.25.14.14,172.25.13.242,172.30.60.216,172.25.0.0/16,172.30.0.0/16,10.100.0.0/16"
set "no_proxy=%NO_PROXY%"
.venv\Scripts\python.exe -X utf8 frontends\tuiapp_v2.py %*
rem TUI abnormal exit leaves mouse-reporting / alt-screen modes on, which floods
rem the window with escape-sequence garbage like ^[[<35;82;10M. Reset on exit.
.venv\Scripts\python.exe -c "import sys; sys.stdout.write('\033[?1006l\033[?1003l\033[?1002l\033[?1000l\033[?1049l\033[?25h\033[0m'); sys.stdout.flush()"
