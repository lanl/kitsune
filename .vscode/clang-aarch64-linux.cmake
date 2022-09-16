set(TARGET_TRIPLE "aarch64-linux-gnu")

set(CMAKE_C_COMPILER "/home/omair/work/tools/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang")

set(CMAKE_CXX_COMPILER "/home/omair/work/tools/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++")

set(CMAKE_LIBRARY_ARCHITECTURE "${TARGET_TRIPLE}")

execute_process(
    COMMAND "gcc" -dumpversion
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE GCC_V3
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GCC_INC "/usr/${TARGET_TRIPLE}/include")

set(CMAKE_C_FLAGS "-target ${TARGET_TRIPLE} -I/${GCC_INC} -I/${GCC_INC}/c++/${GCC_V3}/${TARGET_TRIPLE}")

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS}")

execute_process(
    COMMAND "gcc" -dumpmachine
    RESULT_VARIABLE HAD_ERROR
    OUTPUT_VARIABLE BUILD_ENV_TRIPLE
    OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_IGNORE_PATH "/usr/lib/${BUILD_ENV_TRIPLE}")