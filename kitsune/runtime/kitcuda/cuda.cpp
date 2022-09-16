//
//===- llvm-cuda.cpp - Kitsune ABI runtime target CUDA support    ---------===//
//
// TODO:
//     - Need to update LANL/Triad Copyright notice.
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


// TODO:
//   * Need to do a better job tracking and freeing resources as necessary.
//   * Need to ponder a path for better stream usage (probably related to
//     more complex code generation on the compiler side).
//
//

#include "kitrt-debug.h"
#include "kitcuda/llvm-cuda.h"
#include "kitcuda/cuda.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <stdbool.h>

// Has the runtime been initialized (successfully)?
static bool _kitrtIsInitialized = false;

// Measure internal timing of launched kernels.
static bool _kitrtEnableTiming = false;
// Automatically report kernel execution times to stdout.
static bool   _kitrtReportTiming = false;
// Last measured kernel execution time.
static double _kitrtLastEventTime = 0.0f;


// Use heuristic-based launch parameters.
static bool _kitrtUseHeuristicLaunchParameters = false;

// Default number of threads to use per block for kernel
// launches.
static unsigned _kitrtDefaultThreadsPerBlock = 256;

// Default number of blocks per grid (allows for custom
// settings but otherwise automatically computed).
static unsigned _kitrtDefaultBlocksPerGrid = 0;

// Enable external settings for kernel launch parameters.
static bool _kitrtUseCustomLaunchParameters = false;

// CUDA device (-1 flags an uninitialized state). At present
// runtime only supports a single device.
static CUdevice  _kitrtCUdevice = -1;

// Default CUDA context (a nullptr flags an uninitialized state).
// At present the runtime only supports a single context.
static CUcontext _kitrtCUcontext = nullptr;


// Enable auto-prefetching of UVM-managed pointers.  This is a
// very simple approach that likely will have limited success
// in most use cases.
static bool _kitrtEnablePrefetch = true;


// NOTE: Over a series of CUDA releases it is worthwhile to
// check in on the header files for replacement versioned
// entry points into the driver API.  These are typically
// denoted with a '*_vN' naming scheme and don't always
// play well with older entry points.  If you suddenly
// start to see context errors this is certainly worth
// digging into.  We are vulnerable to this issue because
// we are loading dynamic symbols by name and must therefore
// match version details explicitly in the code.
#define declare(name) extern decltype(name) *name##_p = NULL;

declare(cuInit);
declare(cuDeviceGetCount);
declare(cuDeviceGet);
declare(cuCtxCreate_v3);
declare(cuDevicePrimaryCtxRetain);
declare(cuDevicePrimaryCtxRelease_v2);
declare(cuDevicePrimaryCtxReset_v2);
declare(cuCtxDestroy_v2);
declare(cuCtxSetCurrent);
declare(cuCtxPushCurrent_v2);
declare(cuCtxPopCurrent_v2);
declare(cuCtxGetCurrent);
declare(cuStreamCreate);
declare(cuStreamDestroy_v2);
declare(cuStreamSynchronize);
declare(cuLaunchKernel);
declare(cuEventCreate);
declare(cuEventRecord);
declare(cuEventSynchronize);
declare(cuEventElapsedTime);
declare(cuEventDestroy_v2);
declare(cuGetErrorName);
declare(cuGetErrorString);
declare(cuModuleLoadDataEx);
declare(cuModuleLoadData);
declare(cuModuleLoadFatBinary);
declare(cuModuleGetFunction);
declare(cuModuleUnload);

declare(cuMemAllocManaged);
declare(cuMemFree_v2);
declare(cuMemPrefetchAsync);
declare(cuMemAdvise);
declare(cuPointerGetAttribute);
declare(cuPointerSetAttribute);
declare(cuDeviceGetAttribute);
declare(cuCtxSynchronize);
declare(cuModuleGetGlobal_v2);
declare(cuMemcpy);
declare(cuMemcpyHtoD_v2);
declare(cuOccupancyMaxPotentialBlockSize);



#define CU_SAFE_CALL(x)                                      \
  {                                                          \
    CUresult result = x;                                     \
    if (result != CUDA_SUCCESS) {                            \
      const char *msg;                                       \
      cuGetErrorName_p(result, &msg);                        \
      fprintf(stderr, "kitrt %s:%d:\n", __FILE__, __LINE__); \
      fprintf(stderr, "  %s failed ('%s')\n", #x, msg);      \
      cuGetErrorString_p(result, &msg);                      \
      fprintf(stderr, "  error: '%s'\n", msg);               \
      exit(1);                                               \
    }                                                        \
  }



// The runtime maintains a map from fat binary images to CUDA modules
// (CUmodule).  This avoids a redundant load of the fat binary into a
// module when looking up kernels from the generated code.
//
// TODO: Is there a faster path here for lookup?  Is a map more
// complicated than necessary?
typedef std::map<const void*, CUmodule>  KitRTModuleMap;
static KitRTModuleMap _kitrtModuleMap;

// Alongside the module map the runtime also maintains a map from
// kernel name to CUDA function (CUfunction).  Like the modules this
// avoids a call into the module to search for the kernel.
//
// TODO: Ditto from above.  Is there a faster path here for lookup?
// Is a map more complicated than necessary?
typedef std::map<const char *, CUfunction>  KitRTKernelMap;
static KitRTKernelMap _kitrtKernelMap;


// The runtime cooperates with the compiler to manage the prefetching
// (simple for now) of UVM-allcoated data for both GPU and host and
// computations.  This is still experimental and geared specifically
// avoid forcing explicit host/device synchronization calls in
// application code.
//
// This includes tracking allocated UVM memory (via address) and also
// information about the state of the allocation (e.g., size in
// bytes).
//
// With more advanced data flow analysis by the compiler this should
// also enabled more advanced optimizations.
struct AllocMapEntry {
  size_t         size;       // size of allocated buffer in bytes.
  bool           prefetched; // data previously prefetched?
};

// The state of a UVM-allocated region of memory is tracked by the
// runtime via a map from the UVM pointer to an allocation map entry
// (see above).
typedef std::map<void *, AllocMapEntry> KitRTAllocMap;
static KitRTAllocMap _kitrtAllocMap;


/// Register a memory allocation with the runtime.  This registration
/// requires the allocated pointer and the size in bytes of the
/// allocation.
///
/// TODO: We could also consider doing this for all memory allocations
/// and then determining if we should do a memcpy() or a
/// UVM-prefetch...
static void __kitrt_registerMemAlloc(void *addr, size_t size) {
  assert(addr != nullptr && "unexpected null pointer!");
  assert(_kitrtAllocMap.find(addr) == _kitrtAllocMap.end() && "insertion of existing mem alloc pointer!");
  AllocMapEntry E;
  E.size = size;
  // TODO: Technically the residency of an allocation is undetermined
  // at the time of the allocation -- it is only upon first-touch that
  // the allocation occurs (at least for UVM memory).
  E.prefetched = false;
  _kitrtAllocMap[addr] = E;

  #ifdef _KITRT_VERBOSE_
  fprintf(stderr, "kitrt: registered memory allocation (%p).\n", addr);
  #endif
}


/// Register the allocated block of memory pointed by addr as prefetched.
static void __kitrt_registerMemPrefetched(void *addr) {
  KitRTAllocMap::iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end()) {
    cit->second.prefetched = true;
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr, "kitrt: marked allocation (%p) prefetched.\n", addr);
    #endif
  }
}


/// Return the allocated size (in bytes) of the given registered memory
/// allocation pointed to by addr.  If the given pointer is not
/// registered with the runtime zero size will be returned.
static size_t __kitrt_getMemAllocSize(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end())
    return (cit->second).size;
  else
    return 0;
}


/// Return the prefetched status of the given registered memory
/// allocation pointed to by addr. It is currently considered
/// to be a hard runtime error if the given pointer is not from
/// a registered allocation.
static bool __kitrt_isMemPrefetched(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::const_iterator cit = _kitrtAllocMap.find(addr);
  if (cit != _kitrtAllocMap.end()) {
    AllocMapEntry E = cit->second;
    #ifdef _KITRT_VERBOSE_
    if (E.prefetched)
      fprintf(stderr, "kitrt: check allocation (%p) is prefetched.\n", addr);
    else
      fprintf(stderr, "kitrt: check allocation (%p) not prefetched.\n", addr);
    #endif
    return E.prefetched;
  } else {
    return false;
  }
}



void __kitrt_cuMemNeedsPrefetch(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator it = _kitrtAllocMap.find(addr);
  if (it != _kitrtAllocMap.end()) {
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr,
            "kitrt: allocation (%p) needs prefetching (updated on host).\n",
            addr);
    #endif
    // TODO: Logic is a bit "backwards" here at first glance...
    it->second.prefetched = false;
  }
}

static bool __kitrt_unregisterMemAlloc(void *addr) {
  assert(addr != nullptr && "unexpected null pointer!");
  KitRTAllocMap::iterator it = _kitrtAllocMap.find(addr);
  if (it != _kitrtAllocMap.end()) {
    _kitrtAllocMap.erase(it);
    return true;
  } else {
    return false;
  }
}

static bool __kitrt_load_dlsyms() {

  #define DLSYM_LOAD(FN)                                       \
    if (! (FN##_p = (decltype(FN)*)dlsym(dlHandle, #FN))) {    \
      fprintf(stderr, "kitrt: failed to load dlsym 'FN##'.");  \
      return false;}

  static void *dlHandle = nullptr;
  if (dlHandle)
    return true;

  if (dlHandle = dlopen("libcuda.so", RTLD_LAZY)) {
    DLSYM_LOAD(cuInit);
    DLSYM_LOAD(cuGetErrorName);
    DLSYM_LOAD(cuGetErrorString);
    DLSYM_LOAD(cuDeviceGetCount);
    DLSYM_LOAD(cuDeviceGet);
    DLSYM_LOAD(cuDevicePrimaryCtxRetain);
    DLSYM_LOAD(cuDevicePrimaryCtxRelease_v2);
    DLSYM_LOAD(cuDevicePrimaryCtxReset_v2);
    DLSYM_LOAD(cuCtxCreate_v3);
    DLSYM_LOAD(cuCtxDestroy_v2);
    DLSYM_LOAD(cuCtxSetCurrent);
    DLSYM_LOAD(cuCtxPushCurrent_v2);
    DLSYM_LOAD(cuCtxPopCurrent_v2);
    DLSYM_LOAD(cuCtxGetCurrent);

    DLSYM_LOAD(cuMemAllocManaged);
    DLSYM_LOAD(cuMemFree_v2);
    DLSYM_LOAD(cuMemPrefetchAsync);
    DLSYM_LOAD(cuMemAdvise);

    DLSYM_LOAD(cuModuleLoadData);
    DLSYM_LOAD(cuModuleLoadDataEx);
    DLSYM_LOAD(cuModuleLoadFatBinary);
    DLSYM_LOAD(cuModuleGetFunction);
    DLSYM_LOAD(cuModuleGetGlobal_v2);
    DLSYM_LOAD(cuModuleUnload);

    DLSYM_LOAD(cuStreamCreate);
    DLSYM_LOAD(cuStreamDestroy_v2);
    DLSYM_LOAD(cuStreamSynchronize);
    DLSYM_LOAD(cuLaunchKernel);

    DLSYM_LOAD(cuEventCreate);
    DLSYM_LOAD(cuEventRecord);
    DLSYM_LOAD(cuEventSynchronize);
    DLSYM_LOAD(cuEventElapsedTime);
    DLSYM_LOAD(cuEventDestroy_v2);

    DLSYM_LOAD(cuPointerGetAttribute);
    DLSYM_LOAD(cuPointerSetAttribute);
    DLSYM_LOAD(cuDeviceGetAttribute);
    DLSYM_LOAD(cuCtxSynchronize);

    DLSYM_LOAD(cuMemcpy);
    DLSYM_LOAD(cuMemcpyHtoD_v2);
    DLSYM_LOAD(cuOccupancyMaxPotentialBlockSize);
    return true;
  } else {
    fprintf(stderr, "kitrt: Failed to load CUDA dynamic library.\n");
    fprintf(stderr, "kitrt: Is CUDA in your LD_LIBRARY_PATH?\n");
    return false;
  }
}

extern "C" {

bool __kitrt_cuInit() {
  if (_kitrtIsInitialized)
    return true;

  if (!__kitrt_load_dlsyms()) {
    fprintf(stderr, "kitrt: Unable to resolve dynamic symbols for CUDA.\n");
    fprintf(stderr, "kitrt: Check your enviornment settings and CUDA installation.\n");
    fprintf(stderr, "kitrt: aborting...\n");
    abort();
  }

  int deviceCount = 0;

  CU_SAFE_CALL(cuInit_p(0 /*, __CUDA_API_VERSION*/));
  CU_SAFE_CALL(cuDeviceGetCount_p(&deviceCount));
  if (deviceCount == 0) {
    fprintf(stderr, "kitrt: no CUDA devices found!\n");
    exit(1);
  }
  CU_SAFE_CALL(cuDeviceGet_p(&_kitrtCUdevice, 0));
  CU_SAFE_CALL(cuDevicePrimaryCtxRetain_p(&_kitrtCUcontext, _kitrtCUdevice));
  // NOTE: It seems we have to explicitly set the context but that seems
  // to be different than what the driver API docs suggest...
  CU_SAFE_CALL(cuCtxSetCurrent_p(_kitrtCUcontext));
  _kitrtIsInitialized = true;

  char *envValue;
  if ((envValue = getenv("KITRT_CU_THREADS_PER_BLOCK"))) {
    _kitrtDefaultThreadsPerBlock = atoi(envValue);
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr, "kitrt: default threads per block set to: %d\n",
            _kitrtDefaultThreadsPerBlock);
    #endif

  }

  if ((envValue = getenv("KITRT_USE_OCCUPANCY_HEURISTIC"))) {
    _kitrtUseHeuristicLaunchParameters = true;
  } else {
    _kitrtUseHeuristicLaunchParameters = false;
  }

  return _kitrtIsInitialized;
}

void __kitrt_cuDestroy() {
  // Note that this call will destroy the context regardless of how many
  // threads might be using it.  It is also assumed there will be no calls
  // using the context when destroy is called -- thus when we transition to
  // supporting more complex streams we will need to revisit the details
  // here.
  //
  if ( _kitrtIsInitialized) {
    for(auto &AM: _kitrtAllocMap) {
      cuMemFree_v2_p((CUdeviceptr)AM.first);
    }

    // Note that all resources associated with the context will be destroyed.
    CU_SAFE_CALL(cuDevicePrimaryCtxRelease_v2_p(_kitrtCUdevice));
    // This might be unfriendly in the grand scheme of things so be careful
    // how and when you call this...
    CU_SAFE_CALL(cuDevicePrimaryCtxReset_v2_p(_kitrtCUdevice));

    // We can't destroy the primary context...
    //CU_SAFE_CALL(cuCtxDestroy_v2_p(_kitrtCUcontext));

     _kitrtIsInitialized = false;
  }
}

void __kitrt_cuSetCustomLaunchParameters(unsigned BlocksPerGrid,
                                         unsigned ThreadsPerBlock) {
  _kitrtUseCustomLaunchParameters = true;
  _kitrtDefaultBlocksPerGrid = BlocksPerGrid;
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

void __kitrt_cuSetDefaultThreadsPerBlock(unsigned ThreadsPerBlock) {
  _kitrtDefaultThreadsPerBlock = ThreadsPerBlock;
}

void __kitrt_cuEnableEventTiming(unsigned report) {
  _kitrtEnableTiming = true;
  _kitrtReportTiming = report > 0;
}

void __kitrt_cuDisableEventTiming() {
  _kitrtEnableTiming = false;
  _kitrtReportTiming = false;
  _kitrtLastEventTime = 0.0;
}

void __kitrt_cuToggleEventTiming() {
  _kitrtEnableTiming = _kitrtEnableTiming ? false : true;
  _kitrtLastEventTime = 0.0;
}

double __kitrt_cuGetLastEventTime() {
  return _kitrtLastEventTime;
}

void* __kitrt_cuCreateEvent() {
  CUevent e;
  CU_SAFE_CALL(cuEventCreate_p(&e, CU_EVENT_DEFAULT));
  return (void*)e;
}

void __kitrt_cuRecordEvent(void *E) {
  assert(E && "__kitrt_cuRecordEvent() null event!");
  CU_SAFE_CALL(cuEventRecord_p((CUevent)E, 0));
}

void __kitrt_cuSynchronizeEvent(void *E) {
  assert(E && "__kitrt_cuSynchronizeEvent() null event!");
  CU_SAFE_CALL(cuEventSynchronize_p((CUevent)E));
}

void __kitrt_cuDestroyEvent(void *E) {
  assert(E && "__kitrt_cuEventDestroy() null event!");
  CU_SAFE_CALL(cuEventDestroy_v2_p((CUevent)E));
}

float __kitrt_cuElapsedEventTime(void *start, void *stop) {
  assert(start && "__kitrt_cuElapsedEventTime() null starting event");
  float msecs;
  CU_SAFE_CALL(cuEventElapsedTime_p(&msecs, (CUevent)start, (CUevent)stop));
  return(msecs/1000.0f);
}

bool __kitrt_cuIsMemManaged(void *vp) {
  assert(vp && "__kitrt_cuIsMemManaged() null data pointer!");
  CUdeviceptr devp = (CUdeviceptr)vp;
  /* For a tad bit of flexibility we don't wrap this call in a
   * safe call -- we want to return false if the given pointer
   * is "junk" as far as CUDA is concerned.
   */
  unsigned int is_managed;
  CUresult r = cuPointerGetAttribute_p(&is_managed,
                                       CU_POINTER_ATTRIBUTE_IS_MANAGED, devp);
  return (r == CUDA_SUCCESS) && is_managed;
}

void __kitrt_cuEnablePrefetch() {
  _kitrtEnablePrefetch = true;
}

void  __kitrt_cuDisablePrefetch() {
  _kitrtEnablePrefetch = false;
}

void __kitrt_cuMemPrefetchIfManaged(void *vp, size_t size) {
  if (_kitrtEnablePrefetch && __kitrt_cuIsMemManaged(vp))
    __kitrt_cuMemPrefetchAsync(vp, size);
}

void __kitrt_cuMemPrefetchAsync(void *vp, size_t size) {
  CUdeviceptr devp = (CUdeviceptr)vp;
  CU_SAFE_CALL(cuMemPrefetchAsync_p(devp, size, _kitrtCUdevice, NULL));
  __kitrt_registerMemPrefetched(vp);
}

void __kitrt_cuMemPrefetch(void *vp) {
  assert(vp && "__kitrt_cmMemPrefetch() null data pointer!");
  if (_kitrtEnablePrefetch && __kitrt_cuIsMemManaged(vp)) {
    if (not __kitrt_isMemPrefetched(vp)) {
      size_t size = __kitrt_getMemAllocSize(vp);
      if (size > 0) {
        #ifdef _KITRT_VERBOSE_
        fprintf(stderr, "kitrt: prefetch(%p).\n", vp);
        #endif
        __kitrt_cuMemPrefetchAsync(vp, size);
      }
    }
  }
}

__attribute__((malloc))
void *__kitrt_cuMemAllocManaged(size_t size) {
  if (not _kitrtIsInitialized)
    __kitrt_cuInit();

  CUdeviceptr devp;
  CU_SAFE_CALL(cuMemAllocManaged_p(&devp, size, CU_MEM_ATTACH_GLOBAL));
  __kitrt_registerMemAlloc((void*)devp, size);
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_ACCESSED_BY,
                             _kitrtCUdevice));
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                             _kitrtCUdevice));
  /*
  int enable = 1;
  CU_SAFE_CALL(cuPointerSetAttribute_p(&enable,
                                       CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                       devp));
  */
  return (void *)devp;
}

void __kitrt_cuMemFree(void *vp) {
    assert(vp && "__kitrt_cuMemFree() null data pointer!");
    if (__kitrt_unregisterMemAlloc(vp)) {
      CUdeviceptr devp = (CUdeviceptr)vp;
      CU_SAFE_CALL(cuMemFree_v2_p(devp));
    }
}

void __kitrt_cuAdviseRead(void *vp, size_t size) {
  CUdeviceptr devp = (CUdeviceptr)vp;
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_READ_MOSTLY, _kitrtCUdevice));
  CU_SAFE_CALL(cuMemAdvise_p(devp, size, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, _kitrtCUdevice));
}

void __kitrt_cuMemcpySymbolToDevice(void *hostPtr,
                                    uint64_t devPtr,
                                    size_t size) {
  assert(devPtr != 0 &&
         "__kitrt_cuMemcpySymbolToDevice() -- null device pointer!");
  assert(hostPtr != nullptr &&
         "__kitrt_cuMemcpySymbolToDevice() -- null host pointer!");
  assert(size != 0 &&
         "__kitrt_cuMemcpySymbolToDevice() -- requested a 0 byte copy!");
  CU_SAFE_CALL(cuMemcpyHtoD_v2_p(devPtr, hostPtr, size));
}

static void __kitrt_cuMaxPotentialBlockSize(int &blocksPerGrid,
                                            int &threadsPerBlock,
                                            CUfunction F,
                                            size_t numElements) {

  CU_SAFE_CALL(cuOccupancyMaxPotentialBlockSize_p(&blocksPerGrid,
                                     &threadsPerBlock, F, 0,
                                     0, // no dynamic shared memory...
                                     0));
  #ifdef _KITRT_VERBOSE_
  fprintf(stderr, "occupancy returned: %d, %d\n", blocksPerGrid,
                   threadsPerBlock);
  #endif
  blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
}

static void __kitrt_cuGetLaunchParameters(int &threadsPerBlock,
                                          int &blocksPerGrid,
                                          size_t numElements) {
  if (_kitrtUseCustomLaunchParameters) {
    threadsPerBlock = _kitrtDefaultThreadsPerBlock;
    blocksPerGrid = _kitrtDefaultBlocksPerGrid;
    // reset after the launch.
    _kitrtUseCustomLaunchParameters = false;
  } else {
    int warpSize;
    CU_SAFE_CALL(cuDeviceGetAttribute_p(&warpSize,
                                        CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                        _kitrtCUdevice));
    unsigned blockSize = 4 * warpSize;
    //assert(numElements % blockSize == 0);
    // TODO: We could do something a bit more detailed here...
    threadsPerBlock = _kitrtDefaultThreadsPerBlock;
    blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  }
}


void *__kitrt_cuCreateFBModule(const void *fatBin) {
  assert(fatBin && "request to create module from null fatbinary!");
  CUmodule module;
  CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
  //CU_SAFE_CALL(cuModuleLoadFatBinary_p(&module, fatBin));
  return (void*)module;
}

uint64_t __kitrt_cuGetGlobalSymbol(const char *SN, void *CM) {
  assert(SN && "null symbol name (SN)!");
  assert(CM && "null (opaque) CUDA module");
  CUmodule Module = (CUmodule)CM;
  // NOTE: The device pointer and size ('bytes') parameters for
  // cuModuleGetGlobal are optional.  To simplify our code gen
  // work we ignore the size parameter (which is NULL below).
  CUdeviceptr DevPtr;
  size_t bytes;
  CU_SAFE_CALL(cuModuleGetGlobal_v2_p(&DevPtr, &bytes, Module, SN));
  return DevPtr;
}

void *__kitrt_cuLaunchModuleKernel(void *mod,
                                   const char *kernelName,
                                   void **fatBinArgs,
                                   uint64_t numElements) {
  // TODO: we probably want to calculate the launch compilers during code
  // generation (when we have some more information about the actual kernel
  // code -- in that case, we should pass launch parameters to this call.
  int threadsPerBlock, blocksPerGrid;


  CUfunction kFunc;
  CUmodule module = (CUmodule)mod;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));

  if (_kitrtUseHeuristicLaunchParameters)
    __kitrt_cuMaxPotentialBlockSize(blocksPerGrid,
                                    threadsPerBlock,
                                    kFunc,
                                    numElements);
  else
    __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problem with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, 0);
  }
  #ifdef _KITRT_VERBOSE_
  fprintf(stderr, "launch parameters:\n");
  fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
  fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
  fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
  #endif

  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock, 1,
                                1, 0, nullptr, fatBinArgs, NULL));

  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, 0);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    if (_kitrtReportTiming)
      printf("%.8lg\n", msecs / 1000.0);
    _kitrtLastEventTime = msecs / 1000.0;
    cuEventDestroy_v2_p(start);
    cuEventDestroy_v2_p(stop);
  }

  return nullptr;
}


void *__kitrt_cuStreamLaunchFBKernel(const void *fatBin,
                                     const char *kernelName,
                                     void **fatBinArgs,
                                     uint64_t numElements) {
  assert(fatBin && "request to launch null fat binary image!");
  assert(kernelName && "request to launch kernel w/ null name!");
  int threadsPerBlock, blocksPerGrid;

  // TODO: We need a better path here for binding and tracking
  // allcoated resources -- as it stands we will "leak"
  // modules, streams, functions, etc.
  static bool module_built = false;
  static CUmodule module;
  if(!module_built) {
    CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
    module_built = true;
  }
  CUfunction kFunc;
  CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));
  if (_kitrtUseHeuristicLaunchParameters)
    __kitrt_cuMaxPotentialBlockSize(blocksPerGrid,
                                    threadsPerBlock,
                                    kFunc,
                                    numElements);
  else
    __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);

  CUevent start, stop;
  if (_kitrtEnableTiming) {
    // Recall that we have to take a bit of care about how we time the
    // launched kernel's execution time.  The problem with using host-device
    // synchronization points is that they can potentially stall the entire
    // GPU pipeline, which we want to avoid to enable asynchronous data
    // movement and the execution of other kernels on the GPU.
    //
    // A nice overview for measuring performance in CUDA:
    //
    //   https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
    //
    // TODO: What event creation flags do we really want here?   See:
    //
    //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
    //
    cuEventCreate_p(&start, CU_EVENT_DEFAULT);
    cuEventCreate_p(&stop, CU_EVENT_DEFAULT);
    cuEventRecord_p(start, stream);
  }

  #ifdef _KITRT_VERBOSE_
  fprintf(stderr, "launch parameters:\n");
  fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
  fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
  fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
  #endif

  CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock,
                                1, 1, 0, stream, fatBinArgs, NULL));
  if (_kitrtEnableTiming) {
    cuEventRecord_p(stop, stream);
    cuEventSynchronize_p(stop);
    float msecs = 0;
    cuEventElapsedTime_p(&msecs, start, stop);
    if (_kitrtReportTiming)
      printf("%.8lg\n", msecs / 1000.0);
    _kitrtLastEventTime = msecs / 1000.0;
    cuEventDestroy_v2_p(start);
    cuEventDestroy_v2_p(stop);
  }

  return (void *)stream;
}


// Launch a kernel on the default stream.
void *__kitrt_cuLaunchFBKernel(const void *fatBin,
			       const char *kernelName,
			       void **fatBinArgs,
			       uint64_t numElements) {
    assert(fatBin && "request to launch with null fat binary image!");
    assert(kernelName && "request to launch kernel w/ null name!");
    assert(fatBinArgs && "request to launch kernel w/ null fatbin args!");
    int threadsPerBlock, blocksPerGrid;
    CUfunction kFunc;
    CUmodule module;

    KitRTKernelMap::iterator kern_it = _kitrtKernelMap.find(kernelName);
    if (kern_it == _kitrtKernelMap.end()) {
      #ifdef _KITRT_VERBOSE_
      fprintf(stderr,
              "kitrt: module load+kernel lookup for kernel '%s'...\n",
              kernelName);
      #endif
      KitRTModuleMap::iterator mod_it = _kitrtModuleMap.find(fatBin);
      if (mod_it == _kitrtModuleMap.end()) {
        CU_SAFE_CALL(cuModuleLoadData_p(&module, fatBin));
        _kitrtModuleMap[fatBin] = module;
      } else
        module = mod_it->second;

      CU_SAFE_CALL(cuModuleGetFunction_p(&kFunc, module, kernelName));
      _kitrtKernelMap[kernelName] = kFunc;
    } else
      kFunc = kern_it->second;

    __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid,
                                  numElements);
    #ifdef _KITRT_VERBOSE_
    fprintf(stderr, "launch parameters:\n");
    fprintf(stderr, "\tnumber of overall elements: %ld\n", numElements);
    fprintf(stderr, "\tblocks/grid = %d\n", blocksPerGrid);
    fprintf(stderr, "\tthreads/block = %d\n", threadsPerBlock);
    #endif

    CUevent start, stop;
    if (_kitrtEnableTiming) {
      // An overview for measuring performance in CUDA:
      //
      // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
      // TODO: What event creation flags do we really want here? See:
      //
      //   https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html
      //
      cuEventCreate_p(&start, CU_EVENT_BLOCKING_SYNC/*DEFAULT*/);
      cuEventCreate_p(&stop, CU_EVENT_BLOCKING_SYNC/*DEFAULT*/);
      // Kick off an event prior to kernel launch...
      cuEventRecord_p(start, 0);
    }

    CU_SAFE_CALL(cuLaunchKernel_p(kFunc, blocksPerGrid, 1, 1, threadsPerBlock,
                                  1, 1, 0, nullptr, fatBinArgs, NULL));

    if (_kitrtEnableTiming) {
      cuEventRecord_p(stop, 0);
      cuEventSynchronize_p(stop);
      float msecs = 0.0;
      cuEventElapsedTime_p(&msecs, start, stop);
      _kitrtLastEventTime = msecs / 1000.0;

      if (_kitrtReportTiming)
        printf("kitrt: kernel '%s' runtime, %.8lg seconds\n",
        kernelName, _kitrtLastEventTime);
        cuEventDestroy_v2_p(start);
        cuEventDestroy_v2_p(stop);
      }
    return nullptr;  // default stream...
  }


// TODO: This call currently uses a hard-coded kernel name in the
// launch.  Once that is fixed in the ABI code, we can replace this
// call with the FB ("fat binary") launch call above...
void *__kitrt_cuLaunchELFKernel(const void *elf,
                                void **args,
                                size_t numElements) {
  CUmodule module;
  CUfunction kernel;
  CU_SAFE_CALL(cuModuleLoadDataEx_p(&module, elf, 0, 0, 0));
  CU_SAFE_CALL(cuModuleGetFunction_p(&kernel, module, "kitsune_kernel"));
  CUstream stream = nullptr;
  CU_SAFE_CALL(cuStreamCreate_p(&stream, 0));
  int threadsPerBlock, blocksPerGrid;
  __kitrt_cuGetLaunchParameters(threadsPerBlock, blocksPerGrid, numElements);
  CU_SAFE_CALL(cuLaunchKernel_p(kernel, blocksPerGrid, 1, 1, // grid dim
                                threadsPerBlock, 1, 1,       // block dim
                                0, stream,    // shared mem and stream
                                args, NULL)); // arguments
  return stream;
}

void *__kitrt_cuLaunchKernel(llvm::Module &m, void **args, size_t n) {
  std::string ptx = __kitrt_cuLLVMtoPTX(m, _kitrtCUdevice);
  void *elf = __kitrt_cuPTXtoELF(ptx.c_str());
  assert(elf && "failed to compile PTX kernel to ELF image!");
  return __kitrt_cuLaunchELFKernel(elf, args, n);
}

void __kitrt_cuStreamSynchronize(void *vs) {
  if (_kitrtEnableTiming)
    return; // TODO: Is this really safe?  We sync with events for timing.
  CU_SAFE_CALL(cuStreamSynchronize_p((CUstream)vs));
}

void __kitrt_cuCheckCtxState() {
  if (_kitrtIsInitialized) {
    CUcontext c;
    CU_SAFE_CALL(cuCtxGetCurrent_p(&c));
    if (c != _kitrtCUcontext) {
      fprintf(stderr, "kitrt: warning! current cuda context mismatch!\n");
    }
  } else {
    fprintf(stderr,
	    "kitrt: context check encountered uninitialized CUDA state!\n");
  }
}

} // extern "C"
