//===- dylib_support.cpp - Kitsune runtime CUDA dynamic library support ---===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
// To support separate compiliation units within the library the "DL" header can
// either provide us with external or local declarations.  This file is the only
// spot where the external declaration mode should be disabled as the entry
// points will all be declared at global scope here and used elsewhere via
// "extern" access.

#include <cstdio>

// TODO: We should determine why we need to disable "extern" on decl's in
// certain cases. There may be a better way of doing this.
#define __KITRT_DISABLE_EXTERN_DECLS__
#include "kitcuda.h"
#include "kitcuda_dylib.h"
#undef __KITRT_DISABLE_EXTERN_DECLS__

// TODO: Any reason to provide this via an environment variable?
static const char *CUDA_DSO_LIBNAME = "libcuda.so";

bool __kitcuda_load_symbols() {
  KIT_NVTX_PUSH("kitcuda:load_symbols", KIT_NVTX_INIT);

  // NOTE: The handle variable below is named to support use of macros for each
  // load call below -- changing the name will break things... TODO: we should
  // probably fix this...
  static void *kitrt_dl_handle = nullptr;
  if (kitrt_dl_handle) {
    fprintf(stderr, "kitcuda: warning - avoiding reloading of symbols...\n");
    return true;
  }

  // TODO: Should we make this a bit more flexible or simply assume adequate
  // setup via environment variables?  For now we've taken the path of a hard
  // error if we can't open the CUDA library.
  kitrt_dl_handle = dlopen(CUDA_DSO_LIBNAME, RTLD_LAZY);
  if (kitrt_dl_handle == NULL) {
    fprintf(stderr, "kitcuda: unable to open '%s'!\n", CUDA_DSO_LIBNAME);
    fprintf(stderr, "  -- Make sure it can be found in your "
                    "shared library path.\n");
    return false; // this will force an abort() during runtime intialization
  }

  if (__kitrt_verbose_mode()) {
    char dl_path[1025];
    fprintf(stderr, "\tkitcuda: successfully opened '%s'\n", CUDA_DSO_LIBNAME);
    if (dlinfo(kitrt_dl_handle, RTLD_DI_ORIGIN, dl_path) != -1) {
      fprintf(stderr, "\t\tfull path: %s/%s\n", dl_path, CUDA_DSO_LIBNAME);
    }
  }

  // NOTE: Try to keep the ordering and grouping here sync'ed with
  // kitcuda_dylib.h.  It makes life a bit easier when adding/removing entry
  // points.

  /* Initialization and query related entry points */
  DLSYM_LOAD(cuInit);
  DLSYM_LOAD(cuDeviceGetCount);
  DLSYM_LOAD(cuDeviceGet);
  DLSYM_LOAD(cuDeviceGetAttribute);
  DLSYM_LOAD(cuDeviceGetAttribute);
  DLSYM_LOAD(cuDriverGetVersion);
  DLSYM_LOAD(cuFuncSetCacheConfig);
  DLSYM_LOAD(cuFuncGetName);
  DLSYM_LOAD(cuFuncGetAttribute);

  /* Context management */
  DLSYM_LOAD(cuCtxCreate_v3);
  DLSYM_LOAD(cuDevicePrimaryCtxRetain);
  DLSYM_LOAD(cuCtxGetCurrent);
  DLSYM_LOAD(cuCtxSetCurrent);
  DLSYM_LOAD(cuCtxPushCurrent_v2);
  DLSYM_LOAD(cuCtxPopCurrent_v2);
  DLSYM_LOAD(cuDevicePrimaryCtxRelease_v2);
  DLSYM_LOAD(cuDevicePrimaryCtxReset_v2);
  DLSYM_LOAD(cuCtxDestroy_v2);
  DLSYM_LOAD(cuCtxSynchronize);

  /* Stream management */
  DLSYM_LOAD(cuStreamCreate);
  DLSYM_LOAD(cuStreamDestroy_v2);
  DLSYM_LOAD(cuStreamSynchronize);
  DLSYM_LOAD(cuStreamAttachMemAsync);

  /* Kernel launching, fat binary, module related */
  DLSYM_LOAD(cuLaunchKernel);
  DLSYM_LOAD(cuModuleLoadDataEx);
  DLSYM_LOAD(cuModuleLoadData);
  DLSYM_LOAD(cuModuleLoadFatBinary);
  DLSYM_LOAD(cuModuleGetFunction);
  DLSYM_LOAD(cuModuleUnload);
  DLSYM_LOAD(cuOccupancyMaxPotentialBlockSize);
  DLSYM_LOAD(cuOccupancyMaxPotentialBlockSizeWithFlags);
  DLSYM_LOAD(cuModuleGetGlobal_v2);

  /* Memory management and movement */
  DLSYM_LOAD(cuMemAllocManaged);
  DLSYM_LOAD(cuMemAllocHost);
  DLSYM_LOAD(cuMemHostAlloc);
  DLSYM_LOAD(cuMemsetD8Async);
  DLSYM_LOAD(cuMemFree_v2);
  DLSYM_LOAD(cuMemPrefetchAsync);
  DLSYM_LOAD(cuMemAdvise);
  DLSYM_LOAD(cuPointerGetAttribute);
  DLSYM_LOAD(cuPointerSetAttribute);
  DLSYM_LOAD(cuMemcpy);
  DLSYM_LOAD(cuMemcpyHtoD_v2);

  /* Error handling */
  DLSYM_LOAD(cuGetErrorName);
  DLSYM_LOAD(cuGetErrorString);

  KIT_NVTX_POP();

  return true;
}
