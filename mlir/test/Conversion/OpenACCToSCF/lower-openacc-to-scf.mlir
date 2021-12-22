// RUN: mlir-opt -split-input-file %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -split-input-file %s | mlir-opt -allow-unregistered-dialect  | FileCheck %s
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

func @compute1(%A: memref<10x10xf32>, %B: memref<10x10xf32>) -> memref<10x10xf32> {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

  acc.parallel {
    acc.loop gang vector {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        %a = load %A[%arg3, %arg3] : memref<10x10xf32>
        %b = mulf %a, %c10 : f32
        store %b, %B[%arg3, %arg3] : memref<10x10xf32>
      }
      acc.yield
    }
    acc.yield
  }
  return
}

//  CHECK-LABEL: func @compute1(
//  CHECK-NEXT:   [[c0:%.*]] = constant 0 : index
//  CHECK-NEXT:   [[c10:%.*]] = constant 10 : index
//  CHECK-NEXT:   [[c1:%.*]] = constant 1 : index
//  CHECK-NEXT:   scf.parallel ([[%arg3:%.*]]) = ([[c0:%.*]]) to ([[c10:%.*]]) step ([[c1:%.*]]) {
//  CHECK-NEXT:     [[%a:%.*]] = load [[%A:%.*]][[[%arg3:%.*]], [[%arg3:%.*]]] : memref<10x10xf32>
//  CHECK-NEXT:     [[%b:%.*]] = mulf [[%a:%.*]], [[%c10:%.*]] : f32
//  CHECK-NEXT:     store [[%b:%.*]], [[%B:%.*]][[[%arg3:%.*]], [[%arg3:%.*]]] : memref<10x10xf32>
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return
//  CHECK-NEXT: }
