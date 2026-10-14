@echo off
rem Task console (projects/navigation): put the robot into a mapped room, it finds its position;
rem later: click an object and it drives there. Same helper as the mapping console.
rem Starts the local helper if needed (tools\start_console.ps1) and opens http://127.0.0.1:8765/tasks.html.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_console.ps1" -Page tasks.html
