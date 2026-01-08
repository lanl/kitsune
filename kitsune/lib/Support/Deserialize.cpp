//==- FromString.cpp - Deserialization functions for Kitsune ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Functions to deserialize kitsune-specific options, attributes etc. from
/// representations that may be used in LLVM .bc or .ll files.
///
//===----------------------------------------------------------------------===//

#include "kitsune/Core/Tapir.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"

using namespace llvm;

std::optional<TTID> llvm::createTTIDFrom(StringRef s) {
  return StringSwitch<std::optional<TTID>>(s)
      .Case("nolo", TTID::Nolo)
      .Case("cuda", TTID::Cuda)
      .Case("hip", TTID::Hip)
      .Case("gpu", TTID::GPU)
      .Case("lambda", TTID::Lambda)
      .Case("omptask", TTID::OMPTask)
      .Case("opencilk", TTID::OpenCilk)
      .Case("openmp", TTID::OpenMP)
      .Case("qthreads", TTID::Qthreads)
      .Case("realm", TTID::Realm)
      .Case("serial", TTID::Serial)
      .Default(std::nullopt);
}

std::optional<TTID> llvm::createTTIDFrom(uint32_t u) {
  switch (u) {
  case 0x0:
    return TTID::Nolo;
  case 0x1:
    return TTID::Serial;
  case 0x2:
    return TTID::Cuda;
  case 0x4:
    return TTID::Hip;
  case 0x8:
    return TTID::OpenCilk;
  case 0x10:
    return TTID::GPU;
  case 0x20:
    return TTID::Qthreads;
  case 0x40:
    return TTID::Realm;
  case 0x80:
    return TTID::Lambda;
  case 0x100:
    return TTID::OMPTask;
  case 0x200:
    return TTID::OpenMP;
  default:
    return std::nullopt;
  }
}

std::optional<MaybeBool> llvm::createMaybeBoolFrom(StringRef s) {
  return StringSwitch<std::optional<MaybeBool>>(s)
      .Case("off", MaybeBool::Off)
      .Case("on", MaybeBool::On)
      .Case("any", MaybeBool::Any)
      .Default(std::nullopt);
}
