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
#include "llvm/Support/raw_ostream.h"

namespace llvm {

enum class TapirTargetID {
  None,     // Perform no lowering
  Serial,   // Lower to serial projection
  Cuda,     // Lower to Cuda ABI
  Hip,      // Lower to the Hip (AMD GPU) ABI
  GPU,      // Lower to the GPU ABI
  Lambda,   // Lower to generic Lambda ABI
  OMPTask,  // Lower to OpenMP task ABI
  OpenCilk, // Lower to OpenCilk ABI
  OpenMP,   // Lower to OpenMP (TODO: Needs to be updated)
  Qthreads, // Lower to Qthreads (TODO: Needs to be updated)
  Realm,    // Lower to Realm (TODO: Needs to be updated)
  Last_TapirTargetID
};

// Serialize the Tapir target into the given output stream. This will write a
// string representation that is compatible with the -ftapir argument used in
// clang.
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target);

enum class TapirNVArchTargetID {
  SM_50, // TODO: Remove depcreated targets based on latest CUDA releases.
  SM_52,
  SM_53,
  SM_60, // Pascal
  SM_61,
  SM_62,
  SM_70, // Volta
  SM_72,
  SM_75, // Turing
  SM_80, // Ampere
  SM_86,
  SM_90, // Hopper
  // TODO: Update this enum when we sync w/ upstream LLVM capabilities.
  Last_TapirNVArchTargetID
};

// Serialize the Tapir target into the given output stream. This will write a
// string representation that is compatible with the -ftapir argument used in
// clang.
raw_ostream &operator<<(raw_ostream &os, const TapirTargetID &Target);

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
