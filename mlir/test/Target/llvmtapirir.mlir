// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @tapir_ops() {
//CHECK: @llvm.syncregion.start()
  %sr = llvm_tapir.tapir_createsyncregion : !llvm.token
  llvm_tapir.detach %sr, ^bb1, ^bb2
^bb1:
  llvm_tapir.reattach %sr, ^bb2
^bb2:
  llvm_tapir.sync %sr, ^bb3
^bb3:
  llvm.return
}
