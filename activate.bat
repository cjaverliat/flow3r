@echo off
set "PATH=%CONDA_PREFIX%\Library\bin;%PATH%"
set "CUDA_HOME=%CONDA_PREFIX%\Library"
set "CUDA_PATH=%CONDA_PREFIX%\Library"
set "PYTHONNOUSERSITE=1"