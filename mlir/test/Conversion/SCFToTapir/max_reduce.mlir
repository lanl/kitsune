// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-tapir %s | FileCheck %s




// CHECK-LABEL:     func.func @reduction_0(%arg0: f32, %arg1: f32) -> f32 attributes {passthrough = ["reduction", "noinline"]} {
// CHECK:             %0 = arith.cmpf ogt, %arg0, %arg1 : f32
// CHECK:             %1 = arith.select %0, %arg0, %arg1 : f32
// CHECK:             return %1 : f32
// CHECK:           }
// CHECK:           func @simple_parallel_max_reduce_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> f32 {
// CHECK:             %0 = llvm_tapir.tapir_syncregion_start : !llvm.token
// CHECK:             %1 = alloca() : memref<f32>
// CHECK:             store %arg3, %1[] : memref<f32>
// CHECK:             br ^bb1(%arg0 : index)
// CHECK:           ^bb1(%2: index):  // 2 preds: ^bb0, ^bb3
// CHECK:             %3 = cmpi slt, %2, %arg1 : index
// CHECK:             cond_br %3, ^bb2, ^bb5
// CHECK:           ^bb2:  // pred: ^bb1
// CHECK:             llvm_tapir.detach %0, ^bb4, ^bb3
// CHECK:           ^bb3:  // 2 preds: ^bb2, ^bb4
// CHECK:             %4 = addi %2, %arg2 : index
// CHECK:             br ^bb1(%4 : index)
// CHECK:           ^bb4:  // pred: ^bb2
// CHECK:             %cst = constant 4.200000e+01 : f32
// CHECK:             %5 = load %1[] : memref<f32>
// CHECK:             %6 = call @reduction_0(%5, %cst) : (f32, f32) -> f32
// CHECK:             store %6, %1[] : memref<f32>
// CHECK:             llvm_tapir.reattach %0, ^bb3
// CHECK:           ^bb5:  // pred: ^bb1
// CHECK:             llvm_tapir.sync %0, ^bb6
// CHECK:           ^bb6:  // pred: ^bb5
// CHECK:             %7 = load %1[] : memref<f32>
// CHECK:             return %7 : f32
// CHECK:           }


func.func @simple_parallel_max_reduce_loop(%arg0: index, %arg1: index, %arg2: index, %arg3: f32) -> f32 {

  %0 = scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) init(%arg3) -> f32 {
    %cst = arith.constant 42.0 : f32
    scf.reduce(%cst) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.cmpf "ogt", %lhs, %rhs : f32
      %2 = arith.select %1, %lhs, %rhs : f32
      scf.reduce.return %2 : f32
    }
  }
  return %0 : f32
 }

