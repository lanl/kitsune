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
using namespace clang;
using namespace llvm::opt;

TapirTargetID clang::parseTapirTarget(const ArgList &Args) {
  // Use Cilk if -ftapir is not specified but -fcilkplus is specified.
  if (!Args.hasArg(options::OPT_ftapir_EQ)) {
    if (Args.hasArg(options::OPT_fcilkplus))
      return TapirTargetID::Cilk;
    if (Args.hasArg(options::OPT_fopencilk))
      return TapirTargetID::OpenCilk;
    return TapirTargetID::None;
  }

  // Otherwise use the runtime specified by -ftapir.
  TapirTargetID TapirTarget = TapirTargetID::None;
  if (const Arg *A = Args.getLastArg(options::OPT_ftapir_EQ))
    TapirTarget = llvm::StringSwitch<TapirTargetID>(A->getValue())
      .Case("none", TapirTargetID::None)
      .Case("serial", TapirTargetID::Serial)
      .Case("cilk", TapirTargetID::Cilk)
      .Case("cuda", TapirTargetID::Cuda)
      .Case("opencilk", TapirTargetID::OpenCilk)      
      .Case("cuda", TapirTargetID::Cuda)
      .Case("openmp", TapirTargetID::OpenMP)
      .Case("qthreads", TapirTargetID::Qthreads)
      .Case("realm", TapirTargetID::Realm)
      .Case("opencl", TapirTargetID::OpenCL)
      .Case("cilk", TapirTargetID::Cilk)
      .Case("gpu", TapirTargetID::GPU)
      .Default(TapirTargetID::Last_TapirTargetID);

  return TapirTarget;
}

TapirNVArchTargetID clang::parseTapirNVArchTarget(const ArgList &Args) {
  TapirNVArchTargetID NVArch = TapirNVArchTargetID::Off;
  if (const Arg *A = Args.getLastArg(options::OPT_ftapir_nvarch_EQ)) 
    NVArch = llvm::StringSwitch<TapirNVArchTargetID>(A->getValue())
      .Case("sm_50", TapirNVArchTargetID::SM_50)
      .Case("sm_52", TapirNVArchTargetID::SM_52)
      .Case("sm_53", TapirNVArchTargetID::SM_53)
      .Case("sm_60", TapirNVArchTargetID::SM_60)
      .Case("sm_62", TapirNVArchTargetID::SM_62)
      .Case("sm_70", TapirNVArchTargetID::SM_70)
      .Case("sm_75", TapirNVArchTargetID::SM_75)
      .Case("sm_80", TapirNVArchTargetID::SM_80)
      .Default(TapirNVArchTargetID::Last_TapirNVArchTargetID);

  return NVArch;
}

llvm::Optional<llvm::StringRef>
clang::serializeTapirTarget(TapirTargetID Target) {
  llvm::Optional<llvm::StringRef> TapirTargetStr;
  switch (Target) {
  case TapirTargetID::None:
    TapirTargetStr = "none";
    break;
  case TapirTargetID::Serial:
    TapirTargetStr = "serial";
    break;
  case TapirTargetID::Cheetah:
    TapirTargetStr = "cheetah";
    break;
  case TapirTargetID::Cilk:
    TapirTargetStr = "cilkplus";
    break;
  case TapirTargetID::Cuda:
    TapirTargetStr = "cuda";
    break;
  case TapirTargetID::GPU:
    TapirTargetStr = "gpu";
    break;
  case TapirTargetID::OpenCilk:
    TapirTargetStr = "opencilk";
    break;
  case TapirTargetID::OpenMP:
    TapirTargetStr = "openmp";
    break;
  case TapirTargetID::Qthreads:
    TapirTargetStr = "qthreads";
    break;
  case TapirTargetID::Last_TapirTargetID:
    break;
  }
  return TapirTargetStr;
}
