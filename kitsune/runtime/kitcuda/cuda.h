//
//===- kitrt-debug.h - Kitsune ABI runtime debug support    --------------===//
//
// TODO: Need to update LANL/Triad Copyright notice.
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
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

#ifndef __KITRT_CUDA_H__
#define __KITRT_CUDA_H__

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif


  /// Initialize the runtime.  This call may be made mulitple
  /// times -- only the intial call will initialize CUDA and
  /// subsequent calls are essentially no-ops.
  bool __kitrt_cuInit();

  /// Clean up and destroy runtime components.
  void __kitrt_cuDestroy();

  /// Provide a set of launch parameters for the next kernel
  /// to be launched after this call.  If these values are not
  /// provided, the runtime will use a simple (commonly used)
  /// calculation for determining the launch parameters.
  void __kitrt_cuSetCustomLaunchParameters(unsigned BlockPerGrid,
                                           unsigned ThreadsPerBlock);

  /// Provide the number of threads per block to use as part of the
  /// parameters for the next kernel launch.
  void  __kitrt_cuSetDefaultThreadsPerBlock(unsigned tpb);
  void  __kitrt_cuEnableEventTiming(unsigned report);
  void  __kitrt_cuDisableEventTiming();
  void  __kitrt_cuToggleEventTiming();
  double __kitrt_cuGetLastEventTime();
  void *__kitrt_cuCreateEvent();
  void  __kitrt_cuRecordEvent(void*);
  void  __kitrt_cuSynchronizeEvent(void*);

  void  __kitrt_cuDestroyEvent(void*);
  float __kitrt_cuElapsedEventTime(void *start, void *stop);


  // Location of the most recently updated UVM allocated memory.
  // This can be either host or device, or host AND device meaning
  // there is valid data in both locations (e.g., computed on host
  // and then prefetched and used as a read-only variable on the
  // GPU).
  enum KitRTMemoryAffinity {
    _KITRT_Host = 0x01,
    _KITRT_Device = 0x02,
    _KITRT_HostAndDevice = 0x04
  };

  bool  __kitrt_cuIsMemManaged(void *vp);
  void  __kitrt_cuEnablePrefetch();
  void  __kitrt_cuDisablePrefetch();
  void  __kitrt_cuMemPrefetchIfManaged(void *vp, size_t size);
  void  __kitrt_cuMemPrefetchAsync(void *vp, size_t size);
  void  __kitrt_cuMemPrefetch(void *vp);
  void  __kitrt_cuMemNeedsPrefetch(void *vp);
  __attribute__((malloc))
  void *__kitrt_cuMemAllocManaged(size_t size);
  void  __kitrt_cuMemFree(void *vp);
  void  __kitrt_cuAdviseRead(void *vp, size_t size);

  bool  __kitrt_cuMemHasHostAffinity(const void *vp);
  bool  __kitrt_cuMemHasDeviceAffinity(const void *vp);
  void  __kitrt_cuMemSetAffinity(void *vp, enum KitRTMemoryAffinity affinity);
  void  __kitrt_cuMemSetHostAffinity(void *vp);
  void  __kitrt_cuMemSetDeviceAffinity(void *vp);
  void  __kitrt_cuMemHintReadOnly(void *vp);
  void  __kitrt_cuMemHintWriteOnly(void *vp);
  void  __kitrt_cuMemHintReadWrite(void *vp);

  void  __kitrt_cuMemcpySymbolToDevice(void *hostSym,
                                       uint64_t devSym,
                                       size_t size);
  void *__kitrt_cuLaunchFBKernel(const void *fatBin,
                                 const char *kernelName,
                                 void **fatBinArgs,
                                 uint64_t numElements);
  void *__kitrt_cuStreamLaunchFBKernel(const void *fatBin,
                                       const char *kernelName,
                                       void **fatBinArgs,
                                       uint64_t numElements);
  void *__kitrt_cuLaunchELFKernel(const void *elf, void **args,
                                  size_t numElements);
  void __kitrt_cuStreamSynchronize(void *vs);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // __KITRT_CUDA_H__

