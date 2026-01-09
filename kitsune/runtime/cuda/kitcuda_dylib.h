/*
 *===---- Kitsune CUDA dynamic library helpers ----------------------------===
 *
 * TODO: Need to update LANL/Triad Copyright notice.
 *
 * Copyright (c) 2021, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2021. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 *
 *===----------------------------------------------------------------------===
 */
#ifndef __KITRT_CUDA_DL_H__
#define __KITRT_CUDA_DL_H__

#include <cuda.h>

#include "dlutils.h"

/**
 * Load the cuda dynamic library symbols we need for the kitsune
 * runtime.  This requires the cuda library (libcuda.so) to be
 * present in your dynamic library search path.
 */
extern bool __kitcuda_load_symbols();

/*
 * NOTE: It is worthwhile to take a look at each CUDA release to check
 * on these entry points.
 */

/* Initialization related entry points */
DECLARE_DLSYM(cuInit);
DECLARE_DLSYM(cuDeviceGetCount);
DECLARE_DLSYM(cuDeviceGet);
DECLARE_DLSYM(cuDeviceGetAttribute);
DECLARE_DLSYM(cuDriverGetVersion);
DECLARE_DLSYM(cuFuncSetCacheConfig);
DECLARE_DLSYM(cuFuncGetName);
DECLARE_DLSYM(cuFuncGetAttribute);

/* Context management */
DECLARE_DLSYM(cuCtxCreate_v4);
DECLARE_DLSYM(cuDevicePrimaryCtxRetain);
DECLARE_DLSYM(cuCtxGetCurrent);
DECLARE_DLSYM(cuCtxSetCurrent);
DECLARE_DLSYM(cuCtxPushCurrent_v2);
DECLARE_DLSYM(cuCtxPopCurrent_v2);
DECLARE_DLSYM(cuDevicePrimaryCtxRelease_v2);
DECLARE_DLSYM(cuDevicePrimaryCtxReset_v2);
DECLARE_DLSYM(cuCtxDestroy_v2);
DECLARE_DLSYM(cuCtxSynchronize);

/* Stream management */
DECLARE_DLSYM(cuStreamCreate);
DECLARE_DLSYM(cuStreamDestroy_v2);
DECLARE_DLSYM(cuStreamSynchronize);
DECLARE_DLSYM(cuStreamAttachMemAsync);

/* Kernel launching, fat binary, module related */
DECLARE_DLSYM(cuLaunchKernel);
DECLARE_DLSYM(cuModuleLoadDataEx);
DECLARE_DLSYM(cuModuleLoadData);
DECLARE_DLSYM(cuModuleLoadFatBinary);
DECLARE_DLSYM(cuModuleGetFunction);
DECLARE_DLSYM(cuModuleUnload);
DECLARE_DLSYM(cuOccupancyMaxPotentialBlockSize);
DECLARE_DLSYM(cuOccupancyMaxPotentialBlockSizeWithFlags);
DECLARE_DLSYM(cuModuleGetGlobal_v2);

/* Memory management and movement */
DECLARE_DLSYM(cuMemAllocManaged);
DECLARE_DLSYM(cuMemAllocHost);
DECLARE_DLSYM(cuMemHostAlloc);
DECLARE_DLSYM(cuMemsetD8Async);
DECLARE_DLSYM(cuMemFree_v2);
DECLARE_DLSYM(cuMemPrefetchAsync);
DECLARE_DLSYM(cuMemAdvise);
DECLARE_DLSYM(cuPointerGetAttribute);
DECLARE_DLSYM(cuPointerSetAttribute);
DECLARE_DLSYM(cuMemcpy);
DECLARE_DLSYM(cuMemcpyHtoD);

/* Error handling */
DECLARE_DLSYM(cuGetErrorName);
DECLARE_DLSYM(cuGetErrorString);

#endif
