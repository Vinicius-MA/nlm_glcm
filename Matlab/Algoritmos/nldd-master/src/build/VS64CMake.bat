@echo off
cls
rem Set paths to the header source files
set ZLIB_PATH=C:\Libs\zlib-1.2.8
set PNG_PATH=C:\Libs\lpng1512
set FFTW_PATH=C:\Libs\FFTW-3.3.4

rem Full paths to the static libraries
set ZLIB_LIBRARY=%ZLIB_PATH%\build\Release\zlib.lib
set PNG_LIBRARY=%PNG_PATH%\build\Release\libpng15.lib

rem Call cmake
rem Remember that the three libraries have to be compiled using the same target machine (either all in 64 bits either all in 32 bits) 
rem and this target machine has to be the same as the used compiler (in this example VS2010 64 bits)
cmake -G "Visual Studio 10 Win64" -D ZLIB_INCLUDE_DIR=%ZLIB_PATH% -D PNG_PNG_INCLUDE_DIR=%PNG_PATH% -D ZLIB_LIBRARY=%ZLIB_LIBRARY% -D PNG_LIBRARY=%PNG_LIBRARY%  -D FFTW3_ROOT_DIR=%FFTW_PATH% ..