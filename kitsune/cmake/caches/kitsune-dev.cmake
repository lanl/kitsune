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

message(DEBUG "KITSUNE-DEV - loading example cache file...")
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
set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "")

# You'll want to tweak this to reduce build times based on what
# you are working on.  By default we provide the full suite of
# clang+tools, openmp, lld, and a debugger via lldb.
set(LLVM_ENABLE_PROJECTS
  clang;openmp;
  CACHE STRING "")

message(DEBUG "  --> KITSUNE-DEV - enabled LLVM projects: ${LLVM_ENABLE_PROJECTS}")

# Keep the in-tree paths sound (i.e., no need for a full install to use these).
set(CLANG_CONFIG_SYSTEM_DIR "${CMAKE_BINARY_DIR}/bin" CACHE STRING "")
set(CLANG_CONFIG_FILE_KITSUNE_DIR "${CMAKE_BINARY_DIR}/share/kitsune" CACHE STRING "")
set(CLANG_CONFIG_FILE_USER_DIR "$ENV(HOME)/.kitsune" CACHE STRING "")

#if ("openmp" IN_LIST LLVM_ENABLE_PROJECTS)
  # Disable this for now -- openmp backend needs to be udpated. 
  #set(KITSUNE_ENABLE_OPENMP_TARGET ON CACHE BOOL "")

  # The default nvidia architecture versions within the openmp project
  # are a bit crufty and can be problematic -- we just blindly set 
  # them to something a bit more modern so the build at least has a 
  # chance.  You may have to tweak this depending upon what you are 
  # doing. 
  set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES 70 CACHE STRING "")
  set(CLANG_OPENMP_NVPTX_DEFAULT_ARCH sm_70 CACHE STRING "")

  # CUDA/NVCC is really only happy with older host compilers.  You 
  # may need to tweak this to make things happy but we typically 
  # need to fall back to older compilers than what we use to build 
  # LLVM/Clang/etc.  gcc 8.x and 9.x are typically safe here... 
 set(CUDA_HOST_COMPILER "/usr/bin/gcc-8" CACHE STRING "")
#endif()

set(_runtimes_list "cheetah")

# Various helpful LLVM-level settings for development/debugging.
set(LLVM_ENABLE_BACKTRACES ON CACHE BOOL "")
set(LLVM_ENABLE_DUMP ON CACHE BOOL "")
set(LLVM_ENABLE_TERMINFO ON CACHE BOOL "")
set(LLVM_ENABLE_THREADS ON CACHE BOOL "")
set(LLVM_INCLUDE_TESTS ON CACHE BOOL "")
set(LLVM_INCLUDE_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
set(LLVM_INSTALL_BINUTILS_SYMLINKS ON CACHE BOOL "")

# Various helpful Clang-level settings for development/debugging.
set(CLANG_BUILD_TOOLS ON CACHE BOOL "")
set(CLANG_INCLUDE_TESTS ON CACHE BOOL "")
set(CLANG_TOOL_CLANG_CHECK_BUILD ON CACHE BOOL "")
set(CLANG_PLUGIN_SUPPORT ON CACHE BOOL "")
set(CLANG_VENDOR "kitsune+tapir" CACHE STRING "")
set(CLANG_VENDOR_UTI "gov.lanl.kitsune" CACHE STRING "")

# Build a minimal set of targets under the assumption the
# build host is the appropriate platform.
set(LLVM_TARGETS_TO_BUILD host;NVPTX;AMDGPU CACHE STRING "")
message(DEBUG "  --> kitsune-dev: enabled LLVM targets: ${LLVM_TARGETS_TO_BUILD}")


# You should carefully look at the parallel workload parameters as
# LLVM builds can easily swamp systems if the amount of parallelism
# exceeds system resources.
set(LLVM_PARALLEL_COMPILE_JOBS 24 CACHE STRING "")
set(LLVM_PARALLEL_LINK_JOBS 4 CACHE STRING "")

# Enable Kitsune mode within the toolchain.
set(CLANG_ENABLE_KITSUNE ON CACHE BOOL
  "Enable Kitsune features in Clang.")

# Enable tailored Kokkos compilation.
set(KITSUNE_ENABLE_KOKKOS_SUPPORT ON CACHE BOOL
  "Enable custom recognition and compilation of Kokkos.")

# NOTE: The OpenCilk runtime is always enabled within our builds
# so it is no longer necessary to explicitly list it (the runtime
# will be downloaded and built as part of the full llvm build 
# process).


set(KITSUNE_ENABLE_QTHREADS_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_REALM_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_CUDATK_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_HIP_TARGET OFF CACHE BOOL "")
set(KITSUNE_ENABLE_OPENCL_TARGET OFF CACHE BOOL "")


if (KITSUNE_ENABLE_CUDATK_TARGET OR
    KITSUNE_ENABLE_HIP_TARGET OR 
    KITSUNE_ENABLE_EXAMPLES OR 
    KITSUNE_ENABLE_KOKKOS_SUPPORT)
  list(APPEND _runtimes_list "kitsune")
endif()

set(LLVM_ENABLE_RUNTIMES ${_runtimes_list} CACHE STRING "")
message(DEBUG "  --> KITSUNE-DEV - enabled LLVM runtimes: ${LLVM_ENABLE_RUNTIMES}")

set(KITSUNE_BUILD_EXAMPLES ON CACHE BOOL "")

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
