/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifdef KITSUNE_CONFIG_H
#error clang/Config/kitsune.h should only be included once.
#else
#define KITSUNE_CONFIG_H

/* This shouldn't happen but could help us catch config issues. */
#ifndef CLANG_ENABLE_KITSUNE
#error kitsune config included when CLANG_ENABLE_KITSUNE undefined.
#endif

#cmakedefine01 KITSUNE_ENABLE_KOKKOS_SUPPORT

/* kitsune-centric cfg file search location */
#cmakedefine CLANG_CONFIG_FILE_KITSUNE_DIR "${CLANG_CONFIG_FILE_KITSUNE_DIR}"

/* kitsune+tapir ABI/runtime codegen targets */
/* Note: The OpenCilk target is (now) always enabled */
#cmakedefine01 KITSUNE_ENABLE_OPENMP_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_CUDA_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_GPU_ABI_TARGET 
#cmakedefine01 KITSUNE_ENABLE_REALM_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_OPENCL_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_HIP_ABI_TARGET

/* kitsune default kokkos configure (.cfg) filename */
#cmakedefine KITSUNE_KOKKOS_CFG_FILENAME "${KITSUNE_KOKKOS_CFG_FILENAME}"
#cmakedefine KITSUNE_KOKKOS_EXTRA_LINK_FLAGS "${KITSUNE_KOKKOS_EXTRA_LINK_FLAGS}"
#cmakedefine KITSUNE_KOKKOS_EXTRA_LINK_LIBS "${KITSUNE_KOKKOS_EXTRA_LINK_LIBS}"

/* tapir default target-specific configure (.cfg) filenames */
#cmakedefine TAPIR_NONE_TARGET_CFG_FILENAME "${TAPIR_SERIAL_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_SERIAL_ABI_TARGET_CFG_FILENAME "${TAPIR_SERIAL_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENCILK_ABI_TARGET_CFG_FILENAME "${TAPIR_OPENCILK_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_CUDA_ABI_TARGET_CFG_FILENAME "${TAPIR_CUDA_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_REALM_ABI_TARGET_CFG_FILENAME "${TAPIR_REALM_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENMP_ABI_TARGET_CFG_FILENAME "${TAPIR_OPENMP_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_QTHREADS_ABI_TARGET_CFG_FILENAME "${TAPIR_QTHREADS_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_HIP_ABI_TARGET_CFG_FILENAME "${TAPIR_HIP_ABI_TARGET_CFG_FILENAME}"
#cmakedefine TAPIR_OPENCL_ABI_TARGET_CFG_FILENAME "${TAPIR_OPENCL_ABI_TARGET_CFG_FILENAME}"

#endif
