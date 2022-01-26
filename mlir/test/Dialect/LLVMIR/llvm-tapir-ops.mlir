// RUN: mlir-opt %s | FileCheck %s
// verify that terminators survive the canonicalizer

// CHECK-LABEL: @tapir_ops
// CHECK: llvm.inst.createsyncregion
// CHECK: llvm_tapir.detach
// CHECK: llvm_tapir.reattach
// CHECK: llvm_tapir.sync
func @tapir_ops() {
  %sr = "llvm_tapir.intr.createsyncregion"() : () -> !llvm.token
  llvm_tapir.detach %sr, ^bb1, ^bb2
^bb1
  llvm_tapir.reattach %sr, ^bb2
^bb2
  llvm_tapir.sync %sr, ^bb3
^bb3
  llvm.return
}
