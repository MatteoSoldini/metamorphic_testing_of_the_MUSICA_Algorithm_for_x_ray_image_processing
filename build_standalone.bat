@echo off
cd shaders

echo Building shaders
call compile.bat
cd ..

if not exist out mkdir out
cd out

cmake ../
cmake --build . --target maverick-standalone --config Release

xcopy shaders Release\shaders /e /v /d /y