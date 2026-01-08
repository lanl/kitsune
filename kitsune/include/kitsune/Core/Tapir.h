//===- Tapir.h - Core Kitsune types and enums ------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the available Tapir lowering targets and other types
// that are shared between the front and middle-ends.
//
//===----------------------------------------------------------------------===//

#ifndef KITSUNE_CORE_TAPIR_H
#define KITSUNE_CORE_TAPIR_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace llvm {

/// The identifiers for the known tapir targets.
///
/// These are some useful constraints that it would be useful to maintain.
/// The Nolo tapir target does not perform any lowering - instead, the tapir
/// instructions are retained in the tapir loop even after it has been
/// "lowered". This should always have an integer value of 0. The serial tapir
/// target simply lowers the tapir loop to a serial loop. This should have an
/// integer value of 1 because it "makes sense" in this context - a serial loop
/// can be thought of as a special case of a parallel loop where only a single
/// iteration is executing at a time.
///
/// The values of these enums should not be changed unless absolutely necessary.
/// A number of tests hardcode these values, and they will be hardcoded in
/// bitcode files as well.
///
enum class TTID : uint32_t {
  /// Pseudo tapir target that does not lower tapir instructions. This is
  /// primarily useful to generate, LLVM IR containing tapir instructions.
  Nolo = 0x0,

  /// Lower to serial projection.
  Serial = 0x1,

  /// Lower to Kitsune's NVIDIA GPU runtime (cuda).
  Cuda = 0x2,

  /// Lower to Kitsune's AMD GPU runtime (hip).
  Hip = 0x4,

  /// Lower to the OpenCilk runtime.
  OpenCilk = 0x8,

  /// Lower to kitsune's JIT-enabled, GPU-agnostic runtime.
  /// FIXME: This has been disabled for now, but should be re-enabled shortly.
  GPU = 0x10,

  /// Lower to the qthreads runtime.
  /// FIXME: This is currently disabled and needs to be updated before it can be
  /// re-enabled.
  Qthreads = 0x20,

  /// Lower to Legoin's Realm runtime.
  /// FIXME: This is currently disabled and needs to be updated before it can be
  /// re-enabled.
  Realm = 0x40,

  /// Lower using a generic tapir target that uses bitcode files containing the
  /// bulk of the code used to lower a tapir loop (or other construct).
  /// FIXME: This has not been fully implemented or tested.
  Lambda = 0x80,

  /// Lowering to the OpenMP task ABI.
  /// FIXME: This has not been fully implemented or tested.
  OMPTask = 0x100,

  /// FIXME: Almost certainly obsolete.
  OpenMP = 0x200,

};

/// Convert the integer to a \ref TTID. If the integer cannot be converted to a
/// \ref TTID, return std::nullopt.
std::optional<TTID> createTTIDFrom(uint32_t u);

/// Convert the string to a \ref TTID. If the string cannot be converted to a
/// \ref TTID, return std::nullopt.
std::optional<TTID> createTTIDFrom(StringRef s);

/// An enumeration that may be set to a boolean value or unset.
enum class MaybeBool {
  Off, /// The value is set to false
  On,  /// The value is set to true
  Any  /// The value is unset
};

/// Convert the string to a \ref MaybeBool. If the string cannot be converted to
/// a \ref MaybeBool, return std::nullopt.
std::optional<MaybeBool> createMaybeBoolFrom(StringRef s);

/// The loop spawning strategy.
enum class TapirSpawnStrategy {
  Sequential,       /// Sequential (no spawning)
  DivideAndConquer, /// Divide and conquer
  GPU               /// GPU-centric spawning strategy. Currently unused.
};

} // namespace llvm

#endif // KITSUNE_CORE_TAPIR_H
