//===- TapirTargetIDs.h - Tapir target ID's --------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets.
//
//===----------------------------------------------------------------------===//

#ifndef TAPIR_TARGET_IDS_H_
#define TAPIR_TARGET_IDS_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace llvm {

enum class TapirTargetID {
  Off,      // Completely disabled (i.e., no -ftapir argument was present).
  None,     // Perform no lowering
  Serial,   // Lower to serial projection
  Cheetah,  // Lower to the Cheetah ABI
  Cilk,     // Lower to the Cilk Plus ABI
  Cuda,     // Lower to Cuda ABI
  OpenCilk, // Lower to OpenCilk ABI
  OpenCL,   // Lower to OpenCL ABI
  OpenMP,   // Lower to OpenMP
  GPU,   // Lower to OpenCL
  Qthreads, // Lower to Qthreads
  Realm,    // Lower to Realm
  Last_TapirTargetID
};

enum class TapirNVArchTargetID {
  Off,      // Completely disabled (i.e., -ftapir != gpu|cuda)
  SM_50,    // Maxwell -- NOTE: to be depcreated with CUDA 12.x 
  SM_52,    
  SM_53, 
  SM_60,    // Pascal 
  SM_61,
  SM_62,
  SM_70,    // Volta  
  SM_72, 
  SM_75,    // Turing 
  SM_80,    // Ampere
  // NOTE: LLVM 13.x PTX supports only up through SM_80. 
  // TODO: Update this enum when we sync w/ upstream LLVM.
  Last_TapirNVArchTargetID
};

// Tapir target options

// Virtual base class for Target-specific options.
class TapirTargetOptions {
public:
  enum TapirTargetOptionKind { TTO_OpenCilk, Last_TTO };

private:
  const TapirTargetOptionKind Kind;

public:
  TapirTargetOptionKind getKind() const { return Kind; }

  TapirTargetOptions(TapirTargetOptionKind K) : Kind(K) {}
  TapirTargetOptions(const TapirTargetOptions &) = delete;
  TapirTargetOptions &operator=(const TapirTargetOptions &) = delete;
  virtual ~TapirTargetOptions() {}

  // Top-level method for cloning TapirTargetOptions.  Defined in
  // TargetLibraryInfo.
  TapirTargetOptions *clone() const;
};

// Options for OpenCilkABI Tapir target.
class OpenCilkABIOptions : public TapirTargetOptions {
  std::string RuntimeBCPath;

  OpenCilkABIOptions() = delete;

public:
  OpenCilkABIOptions(StringRef Path)
      : TapirTargetOptions(TTO_OpenCilk), RuntimeBCPath(Path) {}

  StringRef getRuntimeBCPath() const {
    return RuntimeBCPath;
  }

  static bool classof(const TapirTargetOptions *TTO) {
    return TTO->getKind() == TTO_OpenCilk;
  }

protected:
  friend TapirTargetOptions;

  OpenCilkABIOptions *cloneImpl() const {
    return new OpenCilkABIOptions(RuntimeBCPath);
  }
};

} // end namespace llvm

#endif
