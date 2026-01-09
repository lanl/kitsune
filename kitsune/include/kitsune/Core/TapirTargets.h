//===--TapirTargets.h - Helper header for tapir targets -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is simply a header file that includes headers for the known tapir
// targets to make it more convenient to include where needed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TAPIR_TARGETS_H
#define LLVM_TAPIR_TARGETS_H

#include "kitsune/Config/config.h"
#include "llvm/Transforms/Tapir/SerialABI.h"

#if KITSUNE_CUDA_ENABLED
#include "llvm/Transforms/Tapir/CudaABI.h"
#endif // KITSUNE_CUDA_ENABLED

#if KITSUNE_HIP_ENABLED
#include "llvm/Transforms/Tapir/HipABI.h"
#endif // KITSUNE_HIP_ENABLED

#if KITSUNE_LAMBDA_ENABLED
#include "llvm/Transforms/Tapir/LambdaABI.h"
#endif // KITSUNE_LAMBDA_ENABLED

#if KITSUNE_OMPTASK_ENABLED
#include "llvm/Transforms/Tapir/OMPTaskABI.h"
#endif // KITSUNE_OMPTASK_ENABLED

#if KITSUNE_OPENCILK_ENABLED
#include "llvm/Transforms/Tapir/OpenCilkABI.h"
#endif // KITSUNE_OPENCILK_ENABLED

#if KITSUNE_QTHREADS_ENABLED
#include "llvm/Transforms/Tapir/QthreadsABI.h"
#endif // KITSUNE_QTHREADS_ENABLED

#if KITSUNE_REALM_ENABLED
#include "llvm/Transforms/Tapir/RealmABI.h"
#endif // KITSUNE_REALM_ENABLED

#if KITSUNE_GPU_ENABLED
#include "llvm/Transforms/Tapir/GPUABI.h"
#endif // KITSUNE_REALM_ENABLED
       //
#endif // LLVM_TAPIR_TARGETS_H
