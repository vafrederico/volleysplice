@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0benchmark.ps1" %*
