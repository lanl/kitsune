//
//===- llvm-cuda.cpp - Kitsune ABI runtime target CUDA support    ---------===//
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

#ifndef __KITSUNE_RUNTIME_ABI_CUDA_H__
#define __KITSUNE_RUNTIME_ABI_CUDA_H__

#include "kitrt-debug.h"
#include "llvm/IR/Module.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string>

// Dynamic symbols for CUDA calls.
#define extern_declare(name) extern decltype(name)* name##_p;
extern_declare(cuInit);
extern_declare(cuStreamCreate);
extern_declare(cuStreamDestroy_v2);
extern_declare(cuStreamSynchronize);
extern_declare(cuLaunchKernel);
extern_declare(cuDeviceGet);
extern_declare(cuGetErrorName);
extern_declare(cuModuleLoadDataEx);
extern_declare(cuModuleLoadData);
extern_declare(cuModuleGetFunction);
extern_declare(cuModuleUnload);
extern_declare(cuCtxCreate_v2);
extern_declare(cuCtxDestroy_v2);
extern_declare(cuCtxSetCurrent);
extern_declare(cuMemAllocManaged);
extern_declare(cuDeviceGetAttribute);
extern_declare(cuModuleGetGlobal);
extern_declare(cuMemcpy);
extern_declare(cuMemcpyHtoD);

#ifdef __cplusplus
extern "C" {
#endif

  /// Initialize the cuda portion of the Kitsune runtime ABI.
  /// Returns true on success and will return false or abort
  /// on error.
  bool __kitrt_cuInit();

  /// Allocate a managed memory buffer of the given size in
  /// bytes.  Upon failure the call will return a null pointer,
  /// otherwise a pointer to the allocated memory is returned.
  __attribute__((malloc))
  void *__kitrt_cuMemAllocManaged(size_t size);

  /// Launch the named kernel that is part of the "fat binary"
  /// image pointed to by 'fatBin'.  The kernel arguments and
  /// number of elements to be processed by the kernel are
  /// provided by 'kernelArgs' and 'numElements' respectively.
  /// The call returns a CUDA stream assocaited with the kernel
  /// launch; or null if the kernel failed to launch.
  void *__kitrt_cuLaunchFBKernel(const void* fatBin,
                                 const char *kernelName,
                                 void **kernelArgs,
                                 size_t numElements);

  /// Launch the named kernel from the given (opaque) CUDA module.
  /// This assumes a fat binary image has been registered/created
  /// via the runtime and the assocaited module is passed as the
  /// module parameter.  Beyond the use of a pre-existing module,
  /// this call matches the fat-binary kernel launch call above.
  void *__kitrt_cuLaunchModuleKernel(void *CM,
                                     const char *kernelName,
                                     void **kernelArgs,
                                     size_t numElements);

  /// Launch the kernel named "kitsune_kernel" that is part
  /// of the ELF image pointed to by 'elfImg'.  The kernel
  /// arguments and number of elements to be processed by
  /// the kernel are provided by 'kernelArgs' and
  /// 'numElements' respectively.  The call returns a CUDA
  /// stream assocaited with teh kernel launch; or null if the
  /// kernel failed to launch.
  void *__kitrt_cuLaunchELFKernel(const void *elfImg,
                                  void **kernelArgs,
                                  size_t numElements);

  /// Synchronize execution with the given CUDA stream
  /// object (typically a stream returned by one of the launch
  /// calls above).
  void __kitrt_cuStreamSynchronize(void *cu_stream);

  /// Convert the given PTX source, pointed to by 'PTXBuffer'
  /// into an ELF image that is suitable for launch via the
  /// CUDA API.  A pointer to the ELF image is returned upon
  /// success, otherwise null will be returned.
  void *__kitrt_cuPTXtoELF(const char *PTXBuffer);

  /// Create a CUDA module for the given fat binary.  This
  /// path is best used when global variables have to be
  /// tracked and copied between host and device.  Returns
  /// an opaque handel to the module.  Once this is created
  /// a kernel should be launched using
  /// __kitrt_cuStreamLaunchKernelFromModule().
  void *__kitrt_cuCreateFBModule(const void *fatBin);

  /// Look up the given global symbol by name in the specified
  /// CUDA module; see __kitrt_cuCreateFBModule().  The size
  /// of the global is returned in bytes (storage size).
  uint64_t __kitrt_cuGetGlobalSymbol(const char *SN, void *CM);

  /// Copy the given host-side symbol to the device.
  /// NOTE: This call assumes the device pointer was
  /// acquired using __kitrt_cuGetGlobalSymbol().
  void __kitrt_cuMemcpySymbolToDevice(void *HostPtr,
                                      uint64_t DevPtr,
                                      size_t SizeInBytes);

  /// Check the status of the runtime's CUDA context to make
  /// sure it seems to be a valid state.
  void __kitrt_cuCheckCtxState();

#ifdef __cplusplus
} // extern "C"
#endif


#ifdef __cplusplus

/// Launch the LLVM IR kernel named "kitsune_kernel" that
/// is contained with the given LLVM Module.  The kernel
/// arguments and number of elements to be processed by
/// the kernel are provided by 'kernelArgs' and
/// 'numElements' respectively. The call returns a CUDA
/// stream associated with the kernel launch; or null if
/// the kernel failed to launch.
namespace llvm {
  class Module;
}
extern "C" void *__kitrt_cuLaunchKernel(llvm::Module &M,
                                        void **args,
                                        size_t numElements);

/// Convert the given LLVM Module into PTX source and return
/// it to the caller.  Note that the string will be empty if
/// an error occurred; otherwise it will contain a PTX
/// representation of the "kitsune_kernel" (and any supporting
/// device functions) in the LLVM module.
std::string __kitrt_cuLLVMtoPTX(llvm::Module &M, CUdevice device);

#endif  // __cplusplus


#endif // __KITSUNE_RUNTIME_ABI_CUDA_H__
