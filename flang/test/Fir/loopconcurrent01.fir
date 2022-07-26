func private @f4(i32) -> i1

// do_concurrent_loop 
// CHECK-LABEL: @x4
// CHECK-SAME: (i64 %[[lo:.*]],
func @x4(%lo : index, %up : index) {
  %c1 = arith.constant 1 : index
  // CHECK: icmp sgt i64 %[[count]],
  fir.do_concurrent_loop %i = %lo to %up step %c1 {
    // CHECK: trunc i64 %[[i]] to i32
    %i1 = fir.convert %i : (index) -> i32
    // CHECK: call i1 @f4
    %ok = fir.call @f4(%i1) : (i32) -> i1
  }
  // CHECK: ret void
  return
}

