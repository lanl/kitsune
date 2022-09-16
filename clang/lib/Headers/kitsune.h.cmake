
/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */
#ifndef __CLANG_KITSUNE_H__
#define __CLANG_KITSUNE_H__

#include <stdint.h>

#cmakedefine01 KITSUNE_ENABLE_OPENMP_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_QTHREADS_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_CUDA_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_GPU_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_REALM_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_OPENCL_ABI_TARGET
#cmakedefine01 KITSUNE_ENABLE_HIP_ABI_TARGET

#include "kitsune_rt.h"

#if defined(KITSUNE_ENABLE_OPENCL_ABI_TARGET)
#define ocl_mmap(a, n) __kitsune_opencl_mmap_marker((void*)a, n)
#ifdef __cplusplus
extern "C" {
#endif
  void __kitsune_opencl_mmap_marker(void* ptr, uint64_t n);
#ifdef __cplusplus
}
#endif
#endif

#if defined(spawn)
#warning encountered multiple definitions of spawn!
#else
#define spawn _kitsune_spawn
#endif

#if defined(sync)
#warning encountered multiple definitions of sync!
#else
#define sync _kitsune_sync
#endif

#if defined(forall)
#warning encountered multiple definitions of forall!
#else
#define forall _kitsune_forall
#endif

#endif
