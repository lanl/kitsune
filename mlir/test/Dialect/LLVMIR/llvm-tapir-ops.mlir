// RUN: mlir-opt -pass-pipeline='func(canonicalize)' %s | FileCheck %s
// verify that terminators survive the canonicalizer

// CHECK-LABEL: @tapir_ops
// CHECK: llvm_tapir.createsyncregion
// CHECK: llvm_tapir.detach
// CHECK: llvm_tapir.reattach
// CHECK: llvm_tapir.sync
func @tapir_ops(%syncregion : i1) {
  llvm_tapir.createsyncregion
  llvm_tapir.detach %syncregion, ^bb1, ^bb2
^bb1
^bb2
  llvm_tapir.reattach %syncregion, ^bb3
^bb3
  llvm_tapir.sync %syncregion, ^bb4
^bb4
  llvm.return
}