//
//===- kitrt-support.cpp - Kitsune ABI runtime debug support --------------===//
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

#include <cstdio>
#include "kitrt.h"
#include "kitrt-debug.h"
#include "kitcuda/cuda.h"
#include "kithip/hip.h"

enum KitRTRuntimes {
  KITRT_CudaSupport = 0x001,
  KITRT_HipSupport = 0x002,
  KITRT_SPIRVSupport = 0x004, // TODO: should be runtime name, not IR.
  KITRT_RealmSupport = 0x008,
  KITRT_CheetahSupport = 0x010,
  KITRT_NoSupport = 0x020,
  KITRT_Uninitialized = 0x000,
};

uint32_t _kitrtEnabledRuntimes = KITRT_Uninitialized;

extern "C" {

// Get info about supported runtime targets.
bool __kitrt_isCudaSupported() {
  return _kitrtEnabledRuntimes | KITRT_CudaSupport;
}

bool __kitrt_isHipSupported() {
  return _kitrtEnabledRuntimes | KITRT_HipSupport;
}

bool __kitrt_isCheetahSupported() {
  return _kitrtEnabledRuntimes | KITRT_CheetahSupport;
}

bool __kitrt_isRealmSuported() {
  return _kitrtEnabledRuntimes | KITRT_RealmSupport;
}

bool __kitrt_runtimesInit() {

  #ifdef KITRT_CUDA_ENABLED
  if (__kitrt_cuInit()) {
    KITRT_DEBUG(kitrt::dbgs()
                << "kitrt: cuda support successfully initialized.\n");
    _kitrtEnabledRuntimes |= KITRT_CudaSupport;
  }
  #endif

  #ifdef KITRT_HIP_ENABLED
  if (__kitrt_hipInit()) {
    KITRT_DEBUG(kitrt::dbgs()
            << "kitrt: hip support successfully initialized.\n");
    _kitrtEnabledRuntimes |= KITRT_HipSupport;
  }
  #endif

  #ifdef KITRT_REALM_ENABLED
  if (__kitrt_realmInit()) {
    KITRT_DEBUG(kitrt::dbgs()
            << "kitrt: realm support successfully initialized.\n");
    _kitrtEnabledRuntimes |= KITRT_RealmSupport;
  }
  #endif

  #ifdef KITRT_CHEETAH_ENABLED
  // The Cheetah runtime (OpenCilk) is controlled outside of the
  // Kitsune runtime.  It is part of the higher level build at the
  // LLVM level and we will simple mirror that aspect here for
  // completeness.
  _kitrtEnabledRuntimes |= KITRT_CheetahSupport;
  #endif

  if (_kitrtEnabledRuntimes == KITRT_Uninitialized)
    _kitrtEnabledRuntimes = KITRT_NoSupport;

  if (__kitrt_verboseMode()) {
    fprintf(stderr,   "+=========================================+\n");
    fprintf(stderr,   "| Kitsune Runtime: Supported Targets      |\n");
    fprintf(stderr,   "|-----------------------------------------|\n");
    if (__kitrt_isCudaSupported())
      fprintf(stderr, "| Cuda            :    yes                |\n");
    else
      fprintf(stderr, "| Cuda            :     no                |\n");
    if (__kitrt_isHipSupported())
      fprintf(stderr, "| HIP             :    yes                |\n");
    else
      fprintf(stderr, "| HIP             :     no                |\n");
     if (__kitrt_isRealmSuported())
      fprintf(stderr, "| Realm           :    yes                |\n");
    else
      fprintf(stderr, "| Realm           :     no                |\n");
    if (__kitrt_isCheetahSupported())
      fprintf(stderr, "| Cheetah [sic]   :    yes                |\n");
    else
      fprintf(stderr, "| Cheetah [sic]   :     no                |\n");
    fprintf(stderr,   "|-----------------------------------------|\n\n");
  }
}

} // extern
