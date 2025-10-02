//===- cuda.cpp - Kitsune runtime CUDA support    ------------------------===//
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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <stdbool.h>
#include <sys/syscall.h>

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include "kitrt.h"
#include "memory_map.h"

// Global state -- see accessors in kitcuda.h...
bool _kitcuda_initialized = false;
int _kitcuda_device_id = -1;
CUdevice _kitcuda_device = -1;
CUmemLocation _kitcuda_mem_location; 
CUcontext _kitcuda_context;

// TODO: We currently don't use these values within the runtime but
// need to do so!
static int _kitcuda_driver_version;
static int _kitcuda_max_threads_per_blk;
static int _kitcuda_warp_size;
static int _kitcuda_supports_gpu_overlap;
static int _kitcuda_supports_concurrent_kerns;
static int _kitcuda_max_regs_per_blk;
static int _kitcuda_major_compute_capability;
static int _kitcuda_minor_compute_capability;

#ifdef KITCUDA_ENABLE_NVTX
const int KIT_NVTX_INIT = 0;
const int KIT_NVTX_MEM = 1;
const int KIT_NVTX_STREAM = 2;
const int KIT_NVTX_LAUNCH = 3;
const int KIT_NVTX_CLEANUP = 4;

#endif

extern "C" {

bool __kitcuda_initialize() {
  KIT_NVTX_PUSH("kitcuda: initialize", KIT_NVTX_INIT);
  if (_kitcuda_initialized) {
    if (__kitrt_verbose_mode())
      fprintf(stderr, "kitcuda: warning, multiple initialization calls!\n");
    return true;
  }

  // Initialize the shared components of the higher-level runtime.
  __kitrt_initialize();

  if (not __kitcuda_load_symbols()) {
    // TODO: This error block is repetative in the runtime...  Probably best
    // to collapse them down to a call so that we can get consistent messages
    // and mechanisms across the runtime...
    fprintf(stderr, "kitrt: FATAL ERROR - "
                    "unable to resolve dynamic symbols for CUDA!\n");
    fprintf(stderr, "kitrt: aborting.\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // Standard CUDA initialization steps follow...
  int device_count = 0;
  CU_SAFE_CALL(cuInit_p(0));
  CU_SAFE_CALL(cuDeviceGetCount_p(&device_count));
  if (device_count == 0) {
    fprintf(stderr, "kitcuda: FATAL ERROR - "
                    "no suitable CUDA devices found!\n");
    fprintf(stderr, "kitcuda: aborting.\n");
    __kitrt_print_stack_trace();
    abort();
  }

  // Note that instead of sharing a common device id across runtime
  // components we instead isolate them within each sub-component;
  // this allows us to think crazy (future) thoughts like running
  // code on both NVIDIA and AMD GPUs.

  // On systems with multiple devices we can select one via the
  // environment.  This can be helpful when chasing issues related
  // to GPU location within a node (e.g. NUMA-ness).
  if (!__kitrt_get_env_value("KITCUDA_DEVICE_ID", _kitcuda_device_id))
    _kitcuda_device_id = 0;

  _kitcuda_mem_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  _kitcuda_mem_location.id = _kitcuda_device_id; 

  assert(_kitcuda_device_id < device_count &&
         "kitcuda: KITCUDA_DEVICE_ID value exceeds available number"
         " of devices.");

  CU_SAFE_CALL(cuDeviceGet_p(&_kitcuda_device, _kitcuda_device_id));
  CU_SAFE_CALL(cuDevicePrimaryCtxRetain_p(&_kitcuda_context, _kitcuda_device));
  CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));
  _kitcuda_initialized = true;

  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_max_threads_per_blk,
                                      CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_supports_gpu_overlap,
                                      CU_DEVICE_ATTRIBUTE_GPU_OVERLAP,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(&_kitcuda_supports_concurrent_kerns,
                                      CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                                      _kitcuda_device));
  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_max_regs_per_blk, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
      _kitcuda_device));

  CU_SAFE_CALL(cuDriverGetVersion_p(&_kitcuda_driver_version));

  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_major_compute_capability,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _kitcuda_device));

  CU_SAFE_CALL(cuDeviceGetAttribute_p(
      &_kitcuda_minor_compute_capability,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _kitcuda_device));

  if (__kitrt_verbose_mode()) {
    fprintf(stderr, "    kitcuda: found %d devices.\n", device_count);
    fprintf(stderr, "             using device:     %d\n", _kitcuda_device_id);
    fprintf(stderr, "             driver version:   %d\n",
            _kitcuda_driver_version);
    fprintf(stderr, "             compute capability: %d.%d (sm_%d)\n",
            _kitcuda_major_compute_capability,
            _kitcuda_minor_compute_capability,
            _kitcuda_major_compute_capability * 10 +
            _kitcuda_minor_compute_capability);
    fprintf(stderr, "             warp size:        %d\n", _kitcuda_warp_size);
    fprintf(stderr, "             max threads/blk:  %d\n",
            _kitcuda_max_threads_per_blk);
    fprintf(stderr, "             max regs/blk:     %d\n",
            _kitcuda_max_regs_per_blk);
    fprintf(stderr, "             concurrent kerns: %d\n",
            _kitcuda_supports_concurrent_kerns);
    fprintf(stderr, "             gpu overlap:      %d\n",
            _kitcuda_supports_gpu_overlap);
  }

  // At this point we're ready to go as far as CUDA initialization
  // goes.  The remainder of the initialization checks to see if any
  // environment variables are set that tweak the runtime behavior.

  int threads_per_block = 256;
  if (__kitrt_get_env_value("KITCUDA_THREADS_PER_BLOCK", threads_per_block)) {
    __kitcuda_set_default_threads_per_blk(threads_per_block);
    if (__kitrt_verbose_mode())
      fprintf(stderr, "  kitcuda: threads/block: %d\n", threads_per_block);
  }

  bool disable_refined_launches;
  __kitrt_get_env_value("KITCUDA_DISABLE_LAUNCH_REFINEMENT",
                        disable_refined_launches);
  if (disable_refined_launches)
    __kitcuda_enable_launch_refinement(false);
  
  KIT_NVTX_POP();
  return _kitcuda_initialized;
}

void __kitcuda_destroy() {
  if (not _kitcuda_initialized)
    return;

  KIT_NVTX_PUSH("kitcuda:destroy", KIT_NVTX_CLEANUP);
  __kitcuda_destroy_thread_streams();
  __kitrt_destroy_memory_map(__kitcuda_mem_destroy);
  // Note that all resources associated with the context will be destroyed.
  CU_SAFE_CALL(cuDevicePrimaryCtxReset_v2_p(_kitcuda_device));
  _kitcuda_initialized = false;
  KIT_NVTX_POP();
}

} // extern "C"
