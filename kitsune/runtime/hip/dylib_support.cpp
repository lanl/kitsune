/*
 *===- dylib_support.cpp - HIP dynamic library support --------------------===
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 * Copyright 2021, 2023. Los Alamos National Security, LLC. This
 * software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *   with or without modification, are permitted provided that the
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
 *===----------------------------------------------------------------------===
 */
#include <cstdio>

// TODO: We should determine why we need to disable "extern" on decl's in
// certain cases. There may be a better way of doing this.
#define __KITRT_DISABLE_EXTERN_DECLS__
#include "kithip.h"
#include "kithip_dylib.h"
#undef __KITRT_DISABLE_EXTERN_DECLS__

// TODO: Any reason to provide this via an environment variable?
static const char *HIP_DSO_LIBNAME = "libamdhip64.so";

bool __kithip_load_symbols() {
  // NOTE: The handle variable below is named to support use of  macros for each
  // load call below -- changing the name will break things...  TODO: we should
  // probably fix this...
  static void *kitrt_dl_handle = nullptr;
  if (kitrt_dl_handle) {
    fprintf(stderr, "kithip: warning - avoiding reloading of symbols...\n");
    return true;
  }

  // TODO: Should we make this a bit more flexible or simply assume adequate
  // setup via environment variables?  For now we've taken the path of a hard
  // error if we can't open the HIP library.
  kitrt_dl_handle = dlopen(HIP_DSO_LIBNAME, RTLD_LAZY);
  if (kitrt_dl_handle == NULL) {
    fprintf(stderr, "kithip: unable to open '%s'!\n", HIP_DSO_LIBNAME);
    fprintf(stderr, "  -- Make sure it can be found in your "
                    "shared library path.\n");
    return false; // this will force an abort() during runtime intialization
  }

  // NOTE: Try to keep the ordering and grouping here sync'ed with
  // kithip_dylib.h.  It makes life a bit easier when adding/removing entry
  // points.

  /* Initialization and query related entry points */
  DLSYM_LOAD(hipInit);
  DLSYM_LOAD(hipGetDeviceCount);
  DLSYM_LOAD(hipGetDevice);
  DLSYM_LOAD(hipSetDevice);
  DLSYM_LOAD(hipDeviceGetAttribute);
  DLSYM_LOAD(hipGetDeviceProperties);
  DLSYM_LOAD(hipDeviceReset);
  DLSYM_LOAD(hipDeviceSynchronize);

  /* Context management */

  /* Stream management */
  DLSYM_LOAD(hipStreamCreate);
  DLSYM_LOAD(hipStreamCreateWithFlags);
  DLSYM_LOAD(hipStreamDestroy);
  DLSYM_LOAD(hipStreamSynchronize);

  /* Kernel launching, fat binary, module related */
  DLSYM_LOAD(hipModuleLoadData);
  DLSYM_LOAD(hipModuleGetGlobal);
  DLSYM_LOAD(hipModuleGetFunction);
  DLSYM_LOAD(hipLaunchKernel);
  DLSYM_LOAD(hipModuleLaunchKernel);
  DLSYM_LOAD(hipModuleOccupancyMaxPotentialBlockSize);

  /* Memory management and movement */
  DLSYM_LOAD(hipMallocManaged);
  DLSYM_LOAD(hipFree);
  DLSYM_LOAD(hipMemAdvise);
  DLSYM_LOAD(hipMemRangeGetAttribute);
  DLSYM_LOAD(hipPointerGetAttribute);
  DLSYM_LOAD(hipPointerGetAttributes);
  DLSYM_LOAD(hipMemcpy);
  DLSYM_LOAD(hipMemcpyHtoD);
  DLSYM_LOAD(hipMemsetD8Async);
  DLSYM_LOAD(hipMemPrefetchAsync);

  /* Error handling */
  DLSYM_LOAD(hipGetErrorName);
  DLSYM_LOAD(hipGetErrorString);

  return true;
}
