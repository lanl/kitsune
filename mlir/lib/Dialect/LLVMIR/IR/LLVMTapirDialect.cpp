//===- LLVMTapirDialect.cpp - MLIR LLVMSVE ops implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMTapir dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicsAArch64.h" //Presumably this needs changed

#include "mlir/Dialect/LLVMIR/LLVMTapirDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::LLVM;

Optional<MutableOperandRange>
LLVM::Tapir_sync::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return continueDestOperandsMutable();
}

Optional<MutableOperandRange>
LLVM::Tapir_reattach::getMutableSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return continueDestOperandsMutable();
}

Optional<MutableOperandRange>
LLVM::Tapir_detach::getMutableSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return index == 0 ? detachDestOperandsMutable() : continueDestOperandsMutable();
}

void LLVM::LLVMTapirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMTapir.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMTapir.cpp.inc"
