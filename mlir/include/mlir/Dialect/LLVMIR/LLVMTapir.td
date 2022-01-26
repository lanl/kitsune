//===-- LLVMArmSVE.td - LLVMARMSVE dialect op definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the LLVMTapir dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_TAPIR_OPS
#define LLVMIR_TAPIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// LLVMTapir dialect definition
//===----------------------------------------------------------------------===//

def LLVMTapir_Dialect : Dialect {
  let name = "llvm_tapir";
  let cppNamespace = "::mlir::LLVM";
}

// Base class for LLVM operations. All operations get an "llvm." prefix in
// their name automatically. LLVM operations have either zero or one result,
// this class is specialized below for both cases and should not be used
// directly.
class LLVMTapir_Op<string mnemonic, list<OpTrait> traits = []> :
    LLVM_OpBase<LLVMTapir_Dialect, mnemonic, traits>;

// Base class for LLVM intrinsic operations, should not be used directly. Places
// the intrinsic into the LLVM dialect and prefixes its name with "intr.".
class LLVMTapir_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<OpTrait> traits,
                  int numResults>
    : LLVM_IntrOpBase<LLVMTapir_Dialect, "intr." # mnem, !subst(".", "_", mnem),
                      overloadedResults, overloadedOperands, traits,
                      numResults>;

//----------------------------------------------------------------------------//
// MLIR LLVM Arm SVE intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//


// Base class for LLVM terminator operations.  All terminator operations have
// zero results and an optional list of successors.
class LLVMTapir_TerminatorOp<string mnemonic, list<OpTrait> traits = []> :
    LLVMTapir_Op<mnemonic, !listconcat(traits, [Terminator])>;

def LLVM_Tapir_detach : LLVMTapir_TerminatorOp<"detach",
    [AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins LLVM_TokenType:$syncregion,
                   Variadic<LLVM_Type>:$detachDestOperands,
                   Variadic<LLVM_Type>:$continueDestOperands);
  let successors = (successor AnySuccessor:$detachDest, AnySuccessor:$continueDest);
  let assemblyFormat = [{
    $syncregion `,`
    $detachDest (`(` $detachDestOperands^ `:` type($detachDestOperands) `)`)? `,`
    $continueDest (`(` $continueDestOperands^ `:` type($continueDestOperands) `)`)?
    attr-dict
  }];

  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}

def LLVM_Tapir_reattach : LLVMTapir_TerminatorOp<"reattach",
    [DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins LLVM_TokenType:$syncregion,
                   Variadic<LLVM_Type>:$continueDestOperands);
  let successors = (successor AnySuccessor:$continueDest);
  let assemblyFormat = [{
    $syncregion `,`
    $continueDest (`(` $continueDestOperands^ `:` type($continueDestOperands) `)`)?
    attr-dict
  }];

  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}

def LLVM_Tapir_sync : LLVMTapir_TerminatorOp<"sync",
    [DeclareOpInterfaceMethods<BranchOpInterface>,
     NoSideEffect]> {
  let arguments = (ins LLVM_TokenType:$syncregion,
                   Variadic<LLVM_Type>:$continueDestOperands);
  let successors = (successor AnySuccessor:$continueDest);
  let assemblyFormat = [{
    $syncregion `,`
    $continueDest (`(` $continueDestOperands^ `:` type($continueDestOperands) `)`)?
    attr-dict
  }];

  let builders = [LLVM_TerminatorPassthroughOpBuilder];
}

def LLVM_Tapir_createsyncregion : LLVMTapir_IntrOp<"createsyncregion", [], [], [], 1>;

#endif // LLVM_TAPIR_OPS
