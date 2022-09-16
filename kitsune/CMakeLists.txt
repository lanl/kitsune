#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune+tapir project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.19)
project(Kitsune)
include(FetchContent)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(KitsuneUtils)

set(KITSUNE_SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KITSUNE_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR})

if (KITSUNE_ENABLE_KOKKOS_SUPPORT)
  set(KITSUNE_KOKKOS_SRC_DIR  ${LLVM_BINARY_DIR}/kokkos)
  set(KITSUNE_KOKKOS_BUILD_ARGS
      -DKokkos_ENABLE_SERIAL=ON -DBUILD_TESTING=OFF
      -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release)
  #
  # TODO: Note that we have to patch kokkos to disable some sections of code
  # that are using exceptions.  We do this alongside the 'fetch content'
  # mechanisms in cmake. Some aspect of this continually wants to run the
  # patch when it has already been previosuly run...  So...  The patch
  # command has been hacked to always return success to avoid a failed
  # application of the patch from halting the build.
  #
  # TODO: Look for newer Kokkos updates.
  #
  FetchContent_Declare(kokkos
    GIT_REPOSITORY      https://github.com/kokkos/kokkos.git
    GIT_TAG             3.5.00
    SOURCE_DIR          ${KITSUNE_KOKKOS_SRC_DIR}
    UPDATE_DISCONNECTED TRUE
    CMAKE_ARGS          ${KITSUNE_KOKKOS_BUILD_ARGS}
    PATCH_COMMAND patch -p0 --verbose --input=${KITSUNE_SOURCE_DIR}/patches/Kokkos_MemorySpace.patch)

  FetchContent_MakeAvailable(kokkos)

endif()


option(KITSUNE_ENABLE_KITRT "Build the Kitsune runtime library (kitrt)." ON)
if (KITSUNE_ENABLE_KITRT)
  set(KITSUNE_ALL_KITRT_TARGETS "cuda;hip;llvm;realm;spirv" CACHE STRING "List of all Kitsune runtime targets.")
  set(KITSUNE_ENABLE_KITRT_TARGETS "" CACHE STRING
    "Semicolon-separated list of runtime targets to build (or \"all\".)")

  # When building the full suite (LLVM + runtime) we use the
  # compiler-side flags to also enable runtime target support.

  message(STATUS "kitsune: checking compiler build settings to enable runtime components...")

  if (KITSUNE_ENABLE_KITRT_TARGETS STREQUAL "all")
    set(KITSUNE_ENABLE_KITRT_TARGETS ${KITSUNE_ALL_KITRT_TARGETS})
  else()

    if (KITSUNE_ENABLE_CUDA_ABI_TARGET)
      message(STATUS "kitsune: cuda compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "cuda")
    endif()

    if (KITSUNE_ENABLE_HIP_ABI_TARGET)
      message(STATUS "kitsune: hip compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "hip")
    endif()

    if (KITSUNE_ENABLE_LLVM_GPU_ABI_TARGET)
      message(STATUS "kitsune: llvm/jit compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "llvm")
    endif()

    if (KITSUNE_ENABLE_REALM_ABI_TARGET)
      message(STATUS "kitsune: realm compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "realm")
    endif()

    if (KITSUNE_ENABLE_SPIRV_ABI_TARGET)
      message(STATUS "kitsune: spirv compiler target enabled -- enabling runtime support.")
      list(APPEND KITSUNE_ENABLE_KITRT_TARGETS "spirv")
    endif()

  endif()

  if (NOT KITSUNE_ENABLE_KITRT_TARGETS STREQUAL "'")
    foreach(target ${KITSUNE_ALL_KITRT_TARGETS})
      string(TOUPPER "${target}" upper_target)
      message(STATUS "Kitsune runtime: ${target} --> ${upper_target}")
      if (${target} IN_LIST KITSUNE_ENABLE_KITRT_TARGETS)
        message(STATUS "Kitsune runtime target '${target}' ENABLED.")
        message(STATUS "    KITSUNE_RT_ENABLE_${upper_target} ON")
        set(KITSUNE_RT_ENABLE_${upper_target} ON CACHE
            BOOL "Build kitsune runtime support for ${upper_lib} target.")
      else()
        message(STATUS "Kitsune runtime target ${target} DISABLED.")
        message(STATUS "    KITSUNE_RT_ENABLE_${upper_target} OFF")
        set(KITSUNE_RT_ENABLE_${upper_target} OFF CACHE
            BOOL "Build kitsune runtime support for ${upper_lib} target.")
      endif()
    endforeach()
  endif()
endif()

add_subdirectory(runtime)
add_subdirectory(include)

#option(KITSUNE_BUILD_EXAMPLES "Build the Kitsune example programs." OFF)
#if (KITSUNE_BUILD_EXAMPLES)
#  add_subdirectory(examples)
#endif()


