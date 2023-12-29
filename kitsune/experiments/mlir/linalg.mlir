// RUN: mlir-opt %s -convert-linalg-to-loops | FileCheck %s
// RUN: mlir-opt %s -convert-linalg-to-parallel-loops | FileCheck --check-prefix=CHECKPARALLEL %s

// Test that we can lower all the way to LLVM without crashing, don't check results here.
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-linalg-to-llvm -o=/dev/null 2>&1

// CHECK: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

// CHECKPARALLEL: #[[$stride1Dilation1:.*]] = affine_map<(d0, d1) -> (d0 + d1)>

func.func @matmul(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi8>, %M: index, %N: index, %K: index) {
  %z = arith.constant 0 : index
  %A = memref.view %arg0[%z][%M, %K] : memref<?xi8> to memref<?x?xf32>
  %B = memref.view %arg1[%z][%K, %N] : memref<?xi8> to memref<?x?xf32>
  %C = memref.view %arg2[%z][%M, %N] : memref<?xi8> to memref<?x?xf32>
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
               outs(%C: memref<?x?xf32>)
  return
}
