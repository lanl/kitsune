# This file provides a basic configuration for developer's working
# with the Kitsune+Tapir toolchain -- it is designed for active
# breadth, testing, and development vs. a general install.
#
# A ninja-based build is encouraged:
#
#      $ cmake -G Ninja [...] -C this-cache-file.cmake ../llvm
#
message(DEBUG "KITSUNE - loading kitsune-devs' desktop cache file...")
cmake_policy(SET CMP0057 NEW)

# Pick a path for the install location -- note you can use the
# build directory for in-tree testing but this is a nice way to
# also check the install does the right things and can be tested
# within the tree.
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/kitsune CACHE STRING "")

# DEBUG/RELEASE BUILDS
#
# Choose carefully when selecting a full debug build -- builds can
# be very slow and linking can chew up tons of system memory.
#
# Combine this setting with the parallel build parameters. This is
# one reason we encourage ninja use above as you can separately set
# the number of compilation and linking threads.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_PARALLEL_COMPILE_JOBS 16 CACHE STRING "")
set(LLVM_PARALLEL_LINK_JOBS 5 CACHE STRING "")

# note: hip requires lld...
set(LLVM_ENABLE_PROJECTS
  clang;clang-tools-extra;lld;lldb;mlir
  CACHE STRING "")
# note: hip requires compiler-rt
set(LLVM_ENABLE_RUNTIMES
    cheetah;compiler-rt;kitrt;openmp
    CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD
    X86;NVPTX;AMDGPU
    CACHE STRING "")

set(COMPILER_RT_BUILD_MEMPROF OFF CACHE STRING "")
set(COMPILER_RT_BUILD_XRAY OFF CACHE STRING "")
set(COMPILER_RT_BUILD_LIBFUZZER OFF CACHE STRING "")
set(COMPILER_RT_DEFAULT_TARGET_ONLY ON CACHE STRING "")
set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 80 CACHE STRING "")
set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_80 CACHE STRING "")
set(CUDA_HOST_COMPILER "$ENV{CUDA_PATH}/bin/gcc" CACHE STRING "")

# Various helpful LLVM-level settings for development/debugging.
set(LLVM_ENABLE_WARNINGS ON CACHE BOOL "")
set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO ON CACHE BOOL "")
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
set(LLVM_INCLUDE_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")
set(LLVM_BUILD_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_DYLIB_COMPONENTS "all" CACHE STRING "")
set(LLVM_LINK_LLVM_DYLIB ON CACHE BOOL "")
set(LLVM_CCACHE_BUILD ON CACHE BOOL "")

# Various helpful Clang-level settings for development/debugging.
set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_TOOL_CLANG_CHECK_BUILD ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")
set(CLANG_VENDOR "Kitsune" CACHE STRING "")
set(CLANG_VENDOR_UTI "gov.lanl.kitsune" CACHE STRING "")

# Enable specific tapir targets.
set(KITSUNE_ENABLE_TAPIR_TARGETS "cuda;hip;opencilk")

# Enable tailored Kokkos compilation.
set(KITSUNE_KOKKOS_ENABLED ON CACHE BOOL
  "Enable custom recognition and compilation of Kokkos")

set(KITSUNE_BUILD_EXAMPLES OFF CACHE BOOL "")
set(KITSUNE_EXPERIMENTS OFF CACHE BOOL "")
if (LLVM_INCLUDE_TESTS)
  set(KITSUNE_INCLUDE_TESTS ON CACHE BOOL "")
endif()

#set(KITRT_ENABLE_NVTX OFF CACHE BOOL "")
set(KITRT_ENABLE_NVTX ON CACHE BOOL "")

message(DEBUG "  --> KITSUNE-DEV: fixing gcc prefix path.")
execute_process(COMMAND /bin/bash --norc --noprofile -c "dirname \$(dirname `which gcc`)"
  OUTPUT_VARIABLE _gcc_prefix
  ECHO_OUTPUT_VARIABLE)
string(LENGTH ${_gcc_prefix} _gcc_prefix_len)
  if (_gcc_prefix_len GREATER 0)
    string(REGEX REPLACE "\n$" "" _gcc_prefix "${_gcc_prefix}")
    set(GCC_INSTALL_PREFIX ${_gcc_prefix} CACHE STRING "")
  endif()
message(DEBUG "  --> KITSUNE-DEV - gcc prefix: ${GCC_INSTALL_PREFIX}")

message(DEBUG "kitsune-dev: DONE.")
#
#####
