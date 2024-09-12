//===- memory.cpp - Kitsune runtime CUDA memory support  ------------------===//
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021, 2023. Los Alamos National Security, LLC. This software
//  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
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

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include "memory_map.h"
#include <mutex>

static std::mutex _kitcuda_mem_alloc_mutex;

extern "C" {

__attribute__((malloc)) void *__kitcuda_mem_alloc_managed(size_t size) {
  KIT_NVTX_PUSH("kitcuda:mem_alloc_managed",KIT_NVTX_MEM);

  extern bool _kitcuda_initialized;
  if (not _kitcuda_initialized)
    __kitcuda_initialize();

  CUcontext curctx;
  CU_SAFE_CALL(cuCtxGetCurrent_p(&curctx));
  if (curctx == NULL)
    CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));

  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_GLOBAL));

  // Flag the allocation with some CUDA specific flags.  At present these
  // have little impact given the use of a single device and the default
  // stream.  Recall that the current practice is for the actual allocation
  // to occur on first touch -- thus our 'prefetch' status here is a bit
  // misleading (technically we are not prefetched to either host nor device).
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_ACCESSED_BY,
                             _kitcuda_device));
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                             _kitcuda_device));

  int enable = 1;
  CU_SAFE_CALL(
      cuPointerSetAttribute_p(&enable, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, devp));

  // Register this allocation so the runtime can help track the
  // locality (and affinity) of data.
  _kitcuda_mem_alloc_mutex.lock();
  __kitrt_register_mem_alloc((void *)devp, size);
  _kitcuda_mem_alloc_mutex.unlock();

  // NOTE: We can no longer do this in a thread-safe manner... 
  //CU_SAFE_CALL(cuMemPrefetchAsync_p(devp, size, _kitcuda_device,
  //                                  __kitcuda_get_thread_stream()));
  KIT_NVTX_POP();
  return (void *)devp;
}

__attribute__((malloc)) void *
__kitcuda_mem_calloc_managed(size_t count, size_t element_size) {
  assert(count != 0 && "zero-valued item count!");
  assert(element_size != 0 && "zero-valued element size!");

  KIT_NVTX_PUSH("kitcuda:calloc_managed", KIT_NVTX_MEM);

  size_t nbytes = count * element_size;
  CUdeviceptr memp = (CUdeviceptr)__kitcuda_mem_alloc_managed(nbytes);

  // TODO: Is there a risk of a race here?  From the driver API docs:
  //
  //   The cudaMemset functions are asynchronous with respect to the host
  //   except when the target memory is pinned host memory. The Async
  //   versions are always asynchronous with respect to the host.
  //
  // Given our use of UVM we might also be able to use a straight memset()
  // call... which would, of course, place all pages on the host...
  //
  // TODO: We're not set to run on anything but the default stream...
  CU_SAFE_CALL(cuMemsetD8Async_p(memp, 0, nbytes, NULL));
  KIT_NVTX_POP();
  return (void *)memp;
}

__attribute__((malloc)) void *__kitcuda_mem_realloc_managed(void *ptr,
                                                             size_t size) {
  assert(size != 0 && "zero-valued size!");

  KIT_NVTX_PUSH("kitcuda:realloc_managed", KIT_NVTX_MEM);
  void *memptr = nullptr;
  if (ptr == nullptr)
    memptr = __kitcuda_mem_alloc_managed(size);
  else {
    // Check to make sure this is a pointer we're actually managing.
    bool read_only, write_only;
    size_t nbytes = __kitrt_get_mem_alloc_size(ptr, &read_only, &write_only);
    if (nbytes == 0) {
      fprintf(stderr, "kitcuda: warning, realloc() on untracked allocation!\n");
      KIT_NVTX_POP();
      return nullptr;
    }

    if (size > nbytes) {
      // Requested size is larger than currently tracked allocation.
      // Replace it.
      memptr = __kitcuda_mem_alloc_managed(size);
      cuMemcpy_p(/* dest */ (CUdeviceptr)memptr,
                 /* source */ (CUdeviceptr)ptr, nbytes);

      // NOTE: realloc does not guarantee initialized memory outside
      // of existing data...
      __kitcuda_mem_free(ptr);
    } else if (size < nbytes) {
      memptr = __kitcuda_mem_alloc_managed(size);
      cuMemcpy_p(/* dest */ (CUdeviceptr)memptr,
                 /* source */ (CUdeviceptr)ptr, size);
      __kitcuda_mem_free(ptr);
    } else
      memptr = ptr; // same size, just return it...
  }
  KIT_NVTX_POP();
  return memptr;
}

void __kitcuda_mem_free(void *vp) {
  assert(vp && "unexpected null pointer!");

  KIT_NVTX_PUSH("kitcuda:mem_free", KIT_NVTX_MEM);
  // We first remove the allocation from the runtime's
  // map, and then actually release it via CUDA...
  // Note that the versioned free calls are important
  // here -- a non-v2 version will actually result in
  // crashes...
  _kitcuda_mem_alloc_mutex.lock();
  __kitrt_unregister_mem_alloc(vp);
  _kitcuda_mem_alloc_mutex.unlock();
  CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
  KIT_NVTX_POP();
}

void __kitcuda_mem_destroy(void *vp) {
  // This entry point is used to clean up only the
  // CUDA portions of an allocation -- it is used
  // by the runtime at program exit.
  KIT_NVTX_PUSH("kitcuda: mem_destroy", KIT_NVTX_MEM);
  CU_SAFE_CALL(cuMemFree_v2_p((CUdeviceptr)vp));
  KIT_NVTX_POP();
}

bool __kitcuda_is_mem_managed(void *vp) {
  assert(vp && "unexpected null pointer!");
  assert(__kitcuda_is_initialized() && "kitrt: runtime not initialized!");
  KIT_NVTX_PUSH("kitcuda:is_mem_managed", KIT_NVTX_MEM);

  CUdeviceptr devp = (CUdeviceptr)vp;
  unsigned int is_managed;
  // NOTE: We don't wrap in a CUDA-safe call here as we could be
  // passing a bogus pointer -- if we get a CUDA error we will
  // assume the pointer is unmanaged and return false accordingly.
  CUresult r = cuPointerGetAttribute_p(&is_managed,
                                       CU_POINTER_ATTRIBUTE_IS_MANAGED, devp);
  KIT_NVTX_POP();
  return (r == CUDA_SUCCESS) && is_managed;
}

// NOTE: See within the code below for notes about the prefetching
// semantics.
void* __kitcuda_mem_gpu_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "unexpected null pointer!");

  KIT_NVTX_PUSH("kitcuda:mem_gpu_prefetch", KIT_NVTX_MEM);

  size_t size = 0;
  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis
  // and code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior yet.  If we have
  // ever issued a prefetch on a data region it will show here as
  // prefetched and we avoid reissuing a prefetch.  There are
  // obviously cases where this is helpful and others where it will
  // lead to page faults and evictions of pages...  At present this
  // has lead to the best general performance and reduced complexity,
  // while also maintaining correctness.
  if (not __kitrt_is_mem_prefetched(vp, &size)) {
    if (size > 0) {
      CUcontext cu_context;
      CU_SAFE_CALL(cuCtxGetCurrent_p(&cu_context));
      if (cu_context == NULL)
        CU_SAFE_CALL(cuCtxSetCurrent_p(_kitcuda_context));

      // TODO: More work and experimentation needs to be done with
      // managed memory and the advice settings...

      // The runtime (and compiler) semantics assume that a prefetch
      // request implies an inbound kernel launch.  Setting the
      // preferred location does not cause data to migrate to the GPU
      // immediately. Instead, it guides the page migration policy
      // when faults are encountered.
      //
      // If the data is already in device memory and the faulting
      // processor can establish a mapping then migration will be
      // avoided. Otherwise, if a direct mapping cannot be
      // established, then data will be migrated to the processor
      // accessing it.
      //
      // Setting the preferred location does not prevent data
      // prefetching done using cuMemPrefetchAsync() it can also
      // override the page thrash detection and resolution logic in
      // the UM driver. Normally, if a page is constantly thrashing
      // between host and device memory, the driver may eventually pin
      // the page to host memory. If the preferred location is a
      // device memory, then the page will continue to thrash
      // indefinitely.
      //
      // If CU_MEM_ADVISE_SET_READ_MOSTLY is also set on this memory,
      // or any subset of it, then the policies associated with that
      // advice will override the policies of this advice, unless read
      // accesses from the device will not result in a read-only copy
      // being created on that device. See the CUDA docs on the
      // CU_MEM_ADVISE_SET_READ_MOSTLY advice flag.
      CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                                 _kitcuda_device));



      // Issue a prefetch request on the provided stream.  If the given 
      // stream is null, create a new stream and return it. Once issued 
      // go ahead and mark the memory as having been prefetched.  This 
      // "mark" does not guarantee prefetching is complete it simply 
      // flags that the "instruction" has been issued by the runtime.
      CUstream cu_stream;
      if (opaque_stream) 
        cu_stream = (CUstream)opaque_stream;
      else 
        cu_stream = (CUstream)__kitcuda_get_thread_stream();

      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, _kitcuda_device,
                                        cu_stream));
      __kitrt_mark_mem_prefetched(vp);
      return (void*)cu_stream;
    }
  }
  KIT_NVTX_POP();
  // no prefetch, no bound stream to bound it to... 
  return nullptr;
}

void* __kitcuda_mem_host_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "unexpected null pointer!");

  KIT_NVTX_PUSH("kitcuda:mem_host_prefetch", KIT_NVTX_MEM);

  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis and
  // code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior yet.  If we have ever
  // issued a prefetch to the device (gpu) it will show here as
  // prefetched.  In this case we assume a prefetch back to the host
  // is preferred and will let it proceed.  There are obviously cases
  // where this is helpful and others where it will lead to page
  // faults and evictions.  Little work has been done with host-side
  // prefetch requests.
  size_t size;
  if (__kitrt_is_mem_prefetched(vp, &size)) {
    if (size > 0) {
      // The logic here resets the memory advice from being
      // GPU-centric to host-side preferred.  The general logic here
      // is to assume that host-side access suggests pending
      // operations after the completion of a kernel (somewhat the
      // inverse model of what happens prior to a kernel launch).
      //
      // TODO: A lot of work needs to go into seeing if we can be
      // smarter about device- and host-side prefetching.
      CU_SAFE_CALL(cuMemAdvise_p((CUdeviceptr)vp, size,
                                 CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                                 CU_DEVICE_CPU));
      // Issue a prefetch request on the stream associated with the
      // calling thread. Once issued go ahead and mark the memory as
      // no long being prefetched to the device/GPU.  This "mark" does
      // not guarantee prefetching is complete it simply flags that
      // the "instruction" has been issued by the runtime.
      CUstream cu_stream;
      if (opaque_stream) 
        cu_stream = (CUstream)opaque_stream;
      else 
        cu_stream = (CUstream)__kitcuda_get_thread_stream();

      CU_SAFE_CALL(cuMemPrefetchAsync_p((CUdeviceptr)vp, size, CU_DEVICE_CPU,
                                        cu_stream));
      __kitrt_set_mem_prefetch(vp, false);
      return cu_stream;
    }
  }
  KIT_NVTX_POP();
  return nullptr;
}

void __kitcuda_memcpy_sym_to_device(void *hostPtr, uint64_t devPtr,
                                    size_t size) {
  assert(devPtr != 0 && "unexpected null device pointer!");
  assert(hostPtr != nullptr && "unexpected null host pointer!");
  assert(size != 0 && "requested a 0 byte copy!");

  KIT_NVTX_PUSH("kitcuda:memcpy_sym_to_device", KIT_NVTX_MEM);
  CU_SAFE_CALL(cuMemcpyHtoD_v2_p(devPtr, hostPtr, size));
  KIT_NVTX_POP();
}
}
