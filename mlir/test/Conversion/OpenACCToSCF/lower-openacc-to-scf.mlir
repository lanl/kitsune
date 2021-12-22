// RUN: mlir-opt -lower-openacc %s | FileCheck %s

func private @body(index) -> ()
func @compute1() -> () {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index

  acc.parallel {
    acc.loop gang vector {
      scf.for %i = %c0 to %c10 step %c1 {
        call @body(%i) : (index) -> ()
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
//  CHECK-NEXT:   scf.parallel ([[i:%.*]]) = ([[c0]]) to ([[c10]]) step ([[c1]]) {
//  CHECK-NEXT:     call @body([[i]]) : (index) -> ()
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return 
//  CHECK-NEXT: }
