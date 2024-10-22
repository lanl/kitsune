# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
cmake_minimum_required(VERSION 3.20.0)

project(kitrt C CXX)

include(FetchContent)

set(KITRT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITRT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(KITRT_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})

# There is support for two different levels of debugging.  The general
# debug mode (KITRT_ENABLE_DEBUG) will enable runtime assertion checks
# that will cause the runtime to both check and abort on any error
# conditions from the underlying runtimes (e.g., CUDA, HIP, etc.).  The
# second option (KITRT_ENABLE_VERBOSE) will report runtime details and
# give a general feel for what is going on (memory allocations, kernel
# launches, etc.).
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose execution mode for debugging." OFF)

option(KITCUDA_ENABLE_NVTX "Enable NVTX profiling within the runtime" OFF)

# Common headers and source files.
set(KITRT_HDRS
  kitrt.h
  debug.h
  memory.h
  memory_map.h)

set(KITRT_SRCS
  kitrt.cpp
  debug.cpp
  memory.cpp
  memory_map.cpp)

set(KITRT kitrt)

add_library(${KITRT} SHARED ${KITRT_SRCS})

if (KITRT_ENABLE_DEBUG)
  target_compile_definitions(${KITRT} -D_KITRT_DEBUG_)
endif()

if (KITRT_ENABLE_VERBOSE)
  target_compile_definitions(${KITRT} -D_KITRT_VERBOSE_)
endif()

target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(${KITRT} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CLANG_RESOURCE_INTDIR}/lib)

target_include_directories(${KITRT}
  BEFORE PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_INCLUDE_DIR}
  ${KITSUNE_SOURCE_DIR}/include
  ${KITSUNE_RUNTIME_INCLUDE_DIR})

find_library(LIB_LLVM
  NAMES LLVM
  LLVM-${LLVM_VERSION_MAJOR}
  LLVM-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}
  REQUIRED
  PATHS ${LLVM_LIBRARY_OUTPUT_INTDIR}
  NO_DEFAULT_PATH)

# FIXME: We probably don't need to explicitly link pthread and rt. These are
# only necessary for older versions of glibc, but we might get some antique
# software stacks on HPC machines.
target_link_libraries(${KITRT}
  PUBLIC
  ${LIB_LLVM}
  pthread
  rt
  dl)

# The KITSUNE_CUDA_* variables are defined in ../CMakeLists.txt (i.e. in
# kitsune/CMakeLists.txt). They are passed in via CMake because the Kitsune
# runtime is treated as an "ExternalProject".
if (KITSUNE_CUDA_ENABLE)
  list(APPEND KITRT_HDRS
    cuda/kitcuda.h
    cuda/kitcuda_dylib.h)

  target_sources(${KITRT} PUBLIC
    cuda/kitcuda.cpp
    cuda/dylib_support.cpp
    cuda/launching.cpp
    cuda/memory.cpp
    cuda/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_CUDA_ENABLED)
  target_include_directories(${KITRT} SYSTEM PUBLIC ${KITSUNE_CUDA_INCLUDE_DIR})
  target_include_directories(${KITRT} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  target_link_directories(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIBRARY_DIR})

  target_link_libraries(${KITRT} PUBLIC
    ${KITSUNE_CUDA_LIB_NVPTX_STATIC}
    ${KITSUNE_CUDA_LIB_CUDART}
    ${KITSUNE_CUDA_LIB_CUDA})

  if (KITCUDA_ENABLE_NVTX)
    target_compile_definitions(${KITRT} PUBLIC KITCUDA_ENABLE_NVTX)

    find_library(LIB_NVTOOLS_EXT nvToolsExt REQUIRED
      PATHS ${KITSUNE_CUDA_LIBRARY_DIR}
      NO_DEFAULT_PATH)

    target_link_libraries(${KITRT} PUBLIC ${LIB_NVTOOLS_EXT})
  endif()
endif()

# The KITSUNE_HIP_* variables are defined in ../CMakeLists.txt (i.e. in
# kitsune/CMakeLists.txt). They are passed in via CMake because the Kitsune
# runtime is treated as an "ExternalProject".
#
# Note on AMD HIP CMake details...  When is HIP not HIP?  When it is HIP...
# It is easy to confuse requirements for HIP as the HIP "language" vs. the
# HIP runtime.   The language has tangled dependencies within a LLVM context
# as they build around LLVM and Clang too.  Using some of the 'hip::host'
# targets are actually intended for the language and not runtime libraries.
# Unfortunately, these details seem to be a moving target with ecah release
# of rocm/hip...
if (KITSUNE_HIP_ENABLE)
  list(APPEND KITRT_HDRS
    hip/kithip.h
    hip/kithip_dylib.h)

  target_sources(${KITRT} PUBLIC
    hip/kithip.cpp
    hip/dylib_support.cpp
    hip/launching.cpp
    hip/memory.cpp
    hip/streams.cpp)

  target_compile_definitions(${KITRT} PUBLIC KITRT_HIP_ENABLED)
  target_compile_definitions(${KITRT} PUBLIC __HIP_PLATFORM_AMD__)
  target_include_directories(${KITRT} PUBLIC ${KITSUNE_HIP_INCLUDE_DIR})
  target_link_directories(${KITRT} PUBLIC ${KITSUNE_HIP_LIBRARY_DIR})
  target_link_libraries(${KITRT} PUBLIC ${KITSUNE_HIP_LIB_AMDHIP64})
endif()

# ${KITRT_INSTALL_PREFIX} will be set to ${CMAKE_INSTALL_PREFIX} for the broader
# LLVM project that is being built. On the other hand, ${CMAKE_INSTALL_PREFIX}
# will be set to some directory within the build directory into which this is
# "installed" at build time (this is because libkitrt is treated as an
# "external project". There is probably a better way of handling all this).
# This
set_property(TARGET ${KITRT} APPEND PROPERTY
  INSTALL_RPATH "$ORIGIN/../../../")
set_property(TARGET ${KITRT} APPEND PROPERTY
  BUILD_RPATH "$ORIGIN/../../../")

install(TARGETS ${KITRT}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/lib)

# KITSUNE FIXME: Do we really need to install the runtime headers? Do we intend
# this to be used by third-parties?
install(FILES ${KITRT_HDRS}
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/include/kitrt)

# KITSUNE FIXME: Do we really need to install the headers for the targets?
install(DIRECTORY cuda hip realm
  DESTINATION ${CLANG_RESOURCE_SUBDIR}/include/kitrt
  FILES_MATCHING PATTERN "*.h")
