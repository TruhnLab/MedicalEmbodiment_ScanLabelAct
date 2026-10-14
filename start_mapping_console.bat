@echo off
rem Mapping console: drive, SLAM, exploration, recording, 3D reconstruction and labels.
rem Starts the local helper if needed (tools\start_console.ps1) and opens http://127.0.0.1:8765/.
rem First time only: conda env create -f environment.yml  &&  conda run -n landerpi pip install -e .
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_console.ps1"
