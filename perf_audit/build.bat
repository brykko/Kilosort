@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -O3 -arch=sm_120 -shared -o fused_tm.dll fused_tm.cu
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin\nvcc.exe" -O3 -arch=sm_120 -shared -o fused_match.dll fused_match.cu
