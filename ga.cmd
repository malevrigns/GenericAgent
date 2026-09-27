@echo off
cd /d "%~dp0"
rem 自部署模型走内网/平台映射地址，必须绕过本地代理（Clash 7897 到不了会 502）
set "NO_PROXY=localhost,127.0.0.1,::1,172.25.14.14,172.25.13.242,172.30.60.216,172.25.0.0/16,172.30.0.0/16,10.100.0.0/16"
set "no_proxy=%NO_PROXY%"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m ga_cli %*
