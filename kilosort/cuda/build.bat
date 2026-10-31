@echo off
rem Build kilosort_kernels.dll next to this script.
rem Needs the CUDA toolkit (nvcc) and the MSVC build tools. Set CUDA_PATH and VCVARS if
rem they're not at the default locations.
setlocal
if "%VCVARS%"=="" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if "%CUDA_PATH%"=="" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
call "%VCVARS%" >nul || exit /b 1
cd /d "%~dp0"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -shared -o kilosort_kernels.dll kilosort_kernels.cu ^
    -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 ^
    -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 ^
    -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 ^
    -gencode arch=compute_120,code=compute_120
