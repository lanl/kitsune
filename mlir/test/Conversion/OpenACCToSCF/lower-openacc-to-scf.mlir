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

//  CHECK-LABEL: func @compute1
//  CHECK:   scf.parallel ([[i:%.*]]) = (%c0) to (%c10) step (%c1) {
//  CHECK-NEXT:     call @body([[i]]) : (index) -> ()
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return 
//  CHECK-NEXT: }

func private @body2(index, index) -> ()
func @compute2(%l:index, %h:index, %s1:index, %s2:index) -> () {
  acc.parallel {
    acc.loop gang vector {
      scf.for %i = %l to %h step %s1 {
        scf.for %j = %l to %h step %s2 {
          call @body2(%i, %j) : (index, index) -> ()
        }
      }
      acc.yield
    } attributes { collapse = 2 }
    acc.yield
  }
  return
}

//  CHECK: func @compute2([[l:%.*]]: index, [[h:%.*]]: index, [[s1:%.*]]: index, [[s2:%.*]]: index)
//  CHECK-NEXT:   scf.parallel ([[i:%.*]], [[j:%.*]]) = ([[l]], [[l]]) to ([[h]], [[h]]) step ([[s1]], [[s2]]) {
//  CHECK-NEXT:     call @body2([[i]], [[j]])
//  CHECK-NEXT:     scf.yield
//  CHECK-NEXT:   }
//  CHECK-NEXT:   return 
//  CHECK-NEXT: }


