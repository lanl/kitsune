// RUN: fir-opt --split-input-file --cfg-conversion %s | FileCheck %s 

// CHECK-LABEL: @f4
func private @f4(i32) -> i1

// do_concurrent_loop 
// CHECK-LABEL: @x4
func @x4(%lo : index, %up : index) {
  %c1 = arith.constant 1 : index
  // CHECK: scf.parallel (%arg2) = (%arg0) to (%arg1) step (%c1) {
  fir.do_concurrent_loop %i = %lo to %up step %c1 {
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: fir.call @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
  }
  // CHECK: return
  return
}

