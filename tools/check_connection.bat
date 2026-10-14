@echo off
rem Double-click to run the connection check. The window stays open afterwards.
rem Optional: pass the vLLM URL, e.g.  check_connection.bat http://vllm-host:8000
if "%~1"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0check_connection.ps1" -NoPause
) else (
  powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0check_connection.ps1" -NoPause -VllmUrl "%~1"
)
