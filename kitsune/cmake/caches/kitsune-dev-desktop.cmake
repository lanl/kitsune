# This file provides a basic configuration for developer's working
# with the Kitsune+Tapir toolchain -- it is more designed for
# active testing and development vs. general use.
#
# We often use a ninja-based build with a command line of the
# form:
#
#      $ cmake -G Ninja [...] -C this-cache-file.cmake ../llvm
#
# For development the cmake log-level can be helpful.  For example,
#
#      $ cmake ...  --log-level=DEBUG ...
#

message(DEBUG "KITSUNE-DEV - loading developer's cache file...")
cmake_policy(SET CMP0057 NEW)

# Pick a path for the install location -- note you can use the
# build directory for in-tree testing but this is a nice way to
# also check the install does the right things and can be tested
# within the tree.
set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/local CACHE STRING "")

# Careful with a full debug build -- linking can be painfully slow
# and you'll want to tweak the parallel 'width' of the compile and
# link stages below to make sure all is within the capabilities of
# the system you are developing on (see compile+link thread count
# settings below).
#set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# You'll want to tweak this to reduce build times based on what
# you are working on.  By default we provide the full suite of
# clang+tools, openmp, lld, and a debugger via lldb.
set(LLVM_ENABLE_PROJECTS
  clang;clang-tools-extra;openmp;
  CACHE STRING "")

set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 80 CACHE STRING "")
set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_80 CACHE STRING "")

# CUDA/NVCC is really only happy with older host compilers.  You
# may need to tweak this to make things happy but we typically
# need to fall back to older compilers than what we use to build
# LLVM/Clang/etc.  gcc 8.x and 9.x are typically safe here...
 set(CUDA_HOST_COMPILER "/usr/bin/gcc" CACHE STRING "")

set(_runtimes_list "cheetah;cilktools;kitsune")

# Various helpful LLVM-level settings for development/debugging.
set(CLANG_ROUND_TRIP_CC1_ARGS OFF CACHE BOOL "")
set(LLVM_ENABLE_WARNINGS OFF CACHE BOOL "")    # sometimes errors get lost in all the warnings...
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

# You should carefully look at the parallel workload parameters as
# LLVM builds can easily swamp systems if the level of parallelism
# exceeds system resources -- especially memory during the linking
# stages.  Also use care when building on network mounted file
# systems as parallel I/O operations can also bog down the
# scalability of the parallel build.
#
# desktop:
set(LLVM_PARALLEL_COMPILE_JOBS 12 CACHE STRING "")
set(LLVM_PARALLEL_LINK_JOBS 4 CACHE STRING "")

# Various helpful Clang-level settings for development/debugging.
set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_TOOL_CLANG_CHECK_BUILD ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")
set(CLANG_VENDOR "kitsune+tapir" CACHE STRING "")
set(CLANG_VENDOR_UTI "gov.lanl.kitsune" CACHE STRING "")

#set(LLVM_TARGETS_TO_BUILD X86;AArch64;AMDGPU;NVPTX;RISCV CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD X86;NVPTX CACHE STRING "")

# Enable Kitsune mode within the toolchain.
set(CLANG_ENABLE_KITSUNE ON CACHE BOOL
  "Enable Kitsune features in Clang.")
set(KITSUNE_ENABLE_GPU_ABI_TARGET ON CACHE BOOL "")
set(KITSUNE_ENABLE_CUDA_ABI_TARGET ON CACHE BOOL "")
set(KITSUNE_ENABLE_OPENMP_ABI_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_QTHREADS_ABI_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_OPENCL_ABI_TARGET OFF CACHE BOOL "")

# Enable tailored Kokkos compilation.
set(KITSUNE_ENABLE_KOKKOS_SUPPORT ON CACHE BOOL
  "Enable custom recognition and compilation of Kokkos.")

# Enable the Kitsune runtime.
set(KITSUNE_ENABLE_KITRT ON CACHE BOOL "Enable the kitsune runtime.")
set(LLVM_ENABLE_RUNTIMES ${_runtimes_list} CACHE STRING "")
message(DEBUG "  --> KITSUNE-DEV - enabled LLVM runtimes: ${LLVM_ENABLE_RUNTIMES}")

set(KITSUNE_BUILD_EXAMPLES OFF CACHE BOOL "")
set(KITSUNE_EXPERIMENTS ON CACHE BOOL "")
if (LLVM_INCLUDE_TESTS)
  set(KITSUNE_INCLUDE_TESTS ON CACHE BOOL "")
endif()

message(DEBUG "  --> KITSUNE-DEV: fixing gcc prefix path.")
execute_process(
  COMMAND /bin/bash --norc --noprofile -c "dirname \$(dirname `which gcc`)"
  OUTPUT_VARIABLE _gcc_prefix
  ECHO_OUTPUT_VARIABLE
  )
string(LENGTH ${_gcc_prefix} _gcc_prefix_len)
if (_gcc_prefix_len GREATER 0)
  string(REGEX REPLACE "\n$" "" _gcc_prefix "${_gcc_prefix}")
  set(GCC_INSTALL_PREFIX ${_gcc_prefix} CACHE STRING "")
endif()
message(DEBUG "  --> KITSUNE-DEV - gcc prefix: ${GCC_INSTALL_PREFIX}")

unset(_gcc_prefix)
unset(_gcc_prefix_len)
unset(_runtimes_list)

message(DEBUG "kitsune-dev: DONE.")
#
#####
