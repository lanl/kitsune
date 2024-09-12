/*
 *===---- memory.cpp HIP memory management support-------------------------===
 *
 * Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2021, 2023. Los Alamos National Security, LLC. This software
 *  was produced under U.S. Government contract DE-AC52-06NA25396 for Los
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
 *===----------------------------------------------------------------------===
 */
#include "kithip.h"
#include "kithip_dylib.h"
#include "memory_map.h"
#include <mutex>

static std::mutex _kithip_mem_alloc_mutex;

extern "C" {

__attribute__((malloc)) void *__kithip_mem_alloc_managed(size_t size) {
  extern bool _kithip_initialized;
  if (not _kithip_initialized)
    __kithip_initialize();
  
  void *alloced_ptr;
  HIP_SAFE_CALL(hipSetDevice(__kithip_get_device_id()));
  HIP_SAFE_CALL(hipMallocManaged_p(&alloced_ptr, size, hipMemAttachGlobal));
  _kithip_mem_alloc_mutex.lock();
  __kitrt_register_mem_alloc(alloced_ptr, size);
  _kithip_mem_alloc_mutex.unlock();
  // Cheat a tad and just go ahead and issue a prefetch at allocation
  // time.  Could bite us but what the heck...
  HIP_SAFE_CALL(hipMemPrefetchAsync_p(alloced_ptr, size,
                     __kithip_get_device_id(),
                     (hipStream_t)__kithip_get_thread_stream()));
  return alloced_ptr;
}

__attribute__((malloc)) void *__kithip_mem_calloc_managed(size_t count,
                                                          size_t element_size) {
  assert(count != 0 && "zero-valued item count!");
  assert(element_size != 0 && "zero-valued element size!");

  size_t nbytes = count * element_size;
  void *memp = __kithip_mem_alloc_managed(nbytes);

  // TODO: Is there a risk of a race here?
  HIP_SAFE_CALL(
      hipMemsetD8Async_p(memp, 0, nbytes,
		 (hipStream_t)__kithip_get_thread_stream()));
  return (void *)memp;
}

__attribute__((malloc)) void *__kithip_mem_realloc_managed(void *ptr,
                                                            size_t size) {
  assert(size != 0 && "zero-valued size!");
  void *memptr = nullptr;
  size_t alloced_nbytes = 0;

  if (ptr == nullptr) {
    // just a malloc() equivalent call...
    return __kithip_mem_alloc_managed(size);
  } else {
    // Check to make sure this is a pointer we're actually managing.
    bool read_only, write_only;
    alloced_nbytes = __kitrt_get_mem_alloc_size(ptr, &read_only, &write_only);
    assert(alloced_nbytes != 0 && "kithip: realloc() on untracked allocation!");
  }

  if (size > alloced_nbytes) {
    memptr = __kithip_mem_alloc_managed(size);
    HIP_SAFE_CALL(
        hipMemcpy_p(memptr /* dest */, ptr /* source */, 
                    alloced_nbytes, hipMemcpyDefault));
    // TODO: Race?  Do we need to lock the free here?
    __kithip_mem_free(ptr);
  } else if (size < alloced_nbytes) {
    memptr = __kithip_mem_alloc_managed(size);
    HIP_SAFE_CALL(
        hipMemcpy_p(memptr /* dest */, ptr /* source */, 
                    alloced_nbytes, hipMemcpyDefault));
    // TODO: Race?  Do we need to lock the free here?
    __kithip_mem_free(ptr);
  } else
    memptr = ptr; // TODO: does this match realloc() behavior?

  return memptr;
}

void __kithip_mem_free(void *vp) {
  assert(vp && "unexpected null pointer!");
  _kithip_mem_alloc_mutex.lock();
  __kitrt_unregister_mem_alloc(vp);
  _kithip_mem_alloc_mutex.unlock();
  HIP_SAFE_CALL(hipFree_p(vp));
}

void __kithip_mem_destroy(void *vp) {
  // This entry point is used to clean up only the
  // HIP portions of an allocation -- it is used
  // by the runtime at program exit.
  HIP_SAFE_CALL(hipFree_p(vp));
}

bool __kithip_is_mem_managed(void *vp) {
  assert(vp && "unexpected null pointer!");
  assert(__kithip_is_initialized() && "kithip: runtime not initialized!");
  unsigned int is_managed;
  // NOTE: We don't wrap in a HIP-safe call here as we could be
  // passing in a bogus pointer -- if we get a HIP error we will
  // assume the pointer is unmanaged and return false accordingly.
  hipError_t r = hipPointerGetAttribute_p(&is_managed,
                                          HIP_POINTER_ATTRIBUTE_IS_MANAGED, vp);
  return (r == hipSuccess) && is_managed;
}

// NOTE: See within the code below for notes about the prefetching
// semantics.
void* __kithip_mem_gpu_prefetch(void *vp, void *opaque_stream) {
  assert(vp && "unexpected null pointer!");
  size_t size = 0;

  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis
  // and code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior.  If we have
  // ever issued a prefetch on a data region it will show here as
  // prefetched and we avoid reissuing a prefetch.  There are
  // obviously cases where this is helpful and others where it will
  // lead to page faults and evictions of pages...  At present this
  // has lead to the best general performance and reduced complexity,
  // while also maintaining correctness.
  if (not __kitrt_is_mem_prefetched(vp, &size)) {
    if (size > 0) {
      HIP_SAFE_CALL(hipMemAdvise_p(vp, size, hipMemAdviseSetPreferredLocation,
                                   __kithip_get_device_id()));
      HIP_SAFE_CALL(hipMemAdvise_p(vp, size, hipMemAdviseSetAccessedBy,
                                   __kithip_get_device_id()));
      HIP_SAFE_CALL(hipMemAdvise_p(vp, size, hipMemAdviseSetCoarseGrain,
                                   __kithip_get_device_id()));

      hipStream_t hip_stream;
      if (opaque_stream) {
        if (__kitrt_verbose_mode())
        hip_stream = (hipStream_t)opaque_stream;
      } else {
        hip_stream = (hipStream_t)__kithip_get_thread_stream();
        if (__kitrt_verbose_mode())
          fprintf(stderr, "kithip: executing prefetch-driven execution stream [stream=%p].\n",
                  (void*)hip_stream);	
      }

      if (__kitrt_verbose_mode()) 
        fprintf(stderr, "\tkithip: issue prefetch [address=%p, size=%ld, stream=%p].\n", 
                vp, size, (void*)hip_stream);	
      HIP_SAFE_CALL(hipMemPrefetchAsync_p(vp, size, __kithip_get_device_id(),
                                          hip_stream));
      __kitrt_mark_mem_prefetched(vp);
      return (void*)hip_stream;
    }
  } else {
    if (__kitrt_verbose_mode()) 
      fprintf(stderr, 
              "\tkithip: skipping previously prefetched data [address=%p, size=%ld].\n", 
              vp, size);
  }

  return nullptr;
}

void __kithip_mem_host_prefetch(void *vp) {
  assert(vp && "unexpected null pointer!");
  // TODO: Prefetching details and approaches need to be further
  // explored.  In particular, in concert with compiler analysis and
  // code generation.
  //
  // The semantics here are tricky and we don't have enough
  // information to guarantee "smart" behavior.  If we have ever
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
      // GPU-centric to host-side preferred.  The logic is
      // to assume that host-side access suggests pending
      // operations after completion of a kernel (the inverse
      // model of what happens prior to a kernel launch).
      //
      // TODO: A lot of work needs to go into seeing if we can be
      // smarter about device- and host-side prefetching.
      HIP_SAFE_CALL(hipMemAdvise_p(vp, size, hipMemAdviseSetPreferredLocation,
                                   __kithip_get_device_id()));
      // Issue a prefetch request on the stream associated with the
      // calling thread. Once issued go ahead and mark the memory as
      // no long being prefetched to the device/GPU.  This "mark" does
      // not guarantee prefetching is complete it simply flags that
      // the "instruction" has been issued by the runtime.
      HIP_SAFE_CALL(hipMemPrefetchAsync_p(vp, size, __kithip_get_device_id(),
                                 (hipStream_t)__kithip_get_thread_stream()));
      __kitrt_set_mem_prefetch(vp, false);
    }
  }
}

void __kithip_memcpy_sym_to_device(void *hostPtr, void *devPtr,
                                   size_t size) {
  assert(devPtr != 0 && "unexpected null device pointer!");
  assert(hostPtr != nullptr && "unexpected null host pointer!");
  assert(size != 0 && "requested a 0 byte copy!");
  HIP_SAFE_CALL(hipMemcpyHtoD_p(devPtr, hostPtr, size));
}

} // extern "C"
