// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-tapir -split-input-file %s | FileCheck %s

// CHECK-LABEL: parallel_loop

func.func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %step = arith.constant 1 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step) {
    "unknown.op"() : () -> ()
  }
  return
}

func.func @simple_parallel_reduce_loop(%arg0: index, %arg1: index,
                              %arg2: index, %arg3: f32) -> f32 {

  %0 = scf.parallel (%i) = (%arg0) to (%arg1) step (%arg2) init(%arg3) -> f32 {
    %cst = arith.constant 42.0 : f32
    scf.reduce(%cst) : f32 {
    ^bb0(%lhs: f32, %rhs: f32):
      %1 = arith.mulf %lhs, %rhs : f32
      scf.reduce.return %1 : f32
    }
  }
  func.return %0 : f32
 }
