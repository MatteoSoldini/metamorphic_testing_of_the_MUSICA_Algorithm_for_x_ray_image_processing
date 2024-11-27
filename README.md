# Metamorphic Testing of the MUSICA Algorithm for X-Ray Image Processing

## How to run metamorphic testing

1. Initialize submodules: `git submodule update --init`
1. Compile shaders: `shaders/compile.bat`
2. Compile `maverick-standalone.exe` using VS
3. Run MT script: `test/metamorphic_test/run.bat`