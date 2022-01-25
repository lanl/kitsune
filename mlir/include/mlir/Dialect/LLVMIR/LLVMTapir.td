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
include "mlir/Dialect/LLVMIR/LLVMOps.td"

//===----------------------------------------------------------------------===//
// LLVMTapir dialect definition
//===----------------------------------------------------------------------===//

def LLVMTapir_Dialect : Dialect {
  let name = "llvm_tapir";
  let cppNamespace = "::mlir::LLVM";
}

//----------------------------------------------------------------------------//
// MLIR LLVM Arm SVE intrinsics using the MLIR LLVM Dialect type system
//----------------------------------------------------------------------------//

class LLVM_UnaryIntrinsicNoArgumentsOp<string func, list<OpTrait> traits = []> :
    LLVM_OneResultIntrOp<func, [], [0],
           !listconcat([NoSideEffect], traits)> {
}

def LLVM_Tapir_detach : LLVM_TerminatorOp<"detach",
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

def LLVM_Tapir_reattach : LLVM_TerminatorOp<"reattach",
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

def LLVM_Tapir_sync : LLVM_TerminatorOp<"sync",
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

def LLVM_Tapir_createsyncregion : LLVM_UnaryIntrinsicNoArgumentsOp<"createsyncregion">;

#endif // LLVM_TAPIR_OPS
