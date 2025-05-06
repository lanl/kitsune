//===--- Tapir.cpp - C Language Family Language Options ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the functions from Tapir.h
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Tapir.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Transforms/Tapir/TapirTargetIDs.h"

using namespace clang::driver;
using namespace llvm;

std::optional<TapirTargetID> clang::parseTapirTarget(const opt::ArgList &Args) {
  if (const opt::Arg *A = Args.getLastArg(options::OPT_ftapir_EQ))
    return llvm::StringSwitch<std::optional<TapirTargetID>>(A->getValue())
        .Case("none", TapirTargetID::None)
        .Case("serial", TapirTargetID::Serial)
        .Case("cuda", TapirTargetID::Cuda)
        .Case("hip", TapirTargetID::Hip)
        .Case("gpu", TapirTargetID::GPU)
        .Case("opencilk", TapirTargetID::OpenCilk)
        .Case("openmp", TapirTargetID::OpenMP)
        .Case("qthreads", TapirTargetID::Qthreads)
        .Case("realm", TapirTargetID::Realm)
        .Default(std::nullopt);
  return std::nullopt;
}

std::optional<TapirNVArchTargetID>
clang::parseTapirNVArchTarget(const opt::ArgList &Args) {
  if (const opt::Arg *A = Args.getLastArg(options::OPT_ftapir_nvarch_EQ))
    return llvm::StringSwitch<TapirNVArchTargetID>(A->getValue())
        .Case("sm_50", TapirNVArchTargetID::SM_50)
        .Case("sm_52", TapirNVArchTargetID::SM_52)
        .Case("sm_53", TapirNVArchTargetID::SM_53)
        .Case("sm_60", TapirNVArchTargetID::SM_60)
        .Case("sm_62", TapirNVArchTargetID::SM_62)
        .Case("sm_70", TapirNVArchTargetID::SM_70)
        .Case("sm_75", TapirNVArchTargetID::SM_75)
        .Case("sm_80", TapirNVArchTargetID::SM_80)
        .Case("sm_86", TapirNVArchTargetID::SM_86)
        .Case("sm_90", TapirNVArchTargetID::SM_90)
        .Default(TapirNVArchTargetID::Last_TapirNVArchTargetID);

  return std::nullopt;
}

std::optional<llvm::StringRef>
clang::getTargetConfigFileName(const opt::ArgList &Args) {
  if (std::optional<TapirTargetID> tt = parseTapirTarget(Args)) {
    switch (*tt) {
    case TapirTargetID::None:
      return "none.cfg";
    case TapirTargetID::Serial:
      return "serial.cfg";
    case TapirTargetID::Cuda:
      return "cuda.cfg";
    case TapirTargetID::Hip:
      return "hip.cfg";
    case TapirTargetID::OpenCilk:
      return "opencilk.cfg";
    case TapirTargetID::OpenMP:
      return "openmp.cfg";
    case TapirTargetID::Qthreads:
      return "qthreads.cfg";
    case TapirTargetID::Realm:
      return "realm.cfg";
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}
