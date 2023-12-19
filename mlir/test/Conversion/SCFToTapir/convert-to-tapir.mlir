// RUN: mlir-opt -allow-unregistered-dialect -convert-scf-to-tapir %s | FileCheck %s

// CHECK-LABEL:   func @parallel_loop(
// CHECK-SAME:                        [[VAL_0:%.*]]: index, [[VAL_1:%.*]]: index, [[VAL_2:%.*]]: index, [[VAL_3:%.*]]: index, [[VAL_4:%.*]]: index) {
// CHECK:           [[VAL_5:%.*]] = constant 1 : index
// CHECK:           [[VAL_13:%.*]] = "llvm_tapir.intr.createsyncregion"() : () -> !llvm.token
// CHECK:           br ^bb1([[VAL_0]] : index)
// CHECK:         ^bb1([[VAL_6:%.*]]: index):
// CHECK:           [[VAL_7:%.*]] = cmpi slt, [[VAL_6]], [[VAL_2]] : index
// CHECK:           cond_br [[VAL_7]], ^bb2, ^bb9
// CHECK:         ^bb2:
// CHECK:           br ^bb3([[VAL_1]] : index)
// CHECK:         ^bb3([[VAL_8:%.*]]: index):
// CHECK:           [[VAL_9:%.*]] = cmpi slt, [[VAL_8]], [[VAL_3]] : index
// CHECK:           cond_br [[VAL_9]], ^bb4, ^bb7
// CHECK:         ^bb4:
// CHECK:           llvm_tapir.detach [[VAL_13]], ^bb6, ^bb5
// CHECK:         ^bb5:
// CHECK:           [[VAL_11:%.*]] = addi [[VAL_8]], [[VAL_5]] : index
// CHECK:           br ^bb3([[VAL_11]] : index)
// CHECK:         ^bb6:
// CHECK:           "unknown.op"() : () -> ()
// CHECK:           llvm_tapir.reattach [[VAL_13]], ^bb5
// CHECK:         ^bb7: 
// CHECK:           llvm_tapir.sync [[VAL_13]], ^bb8
// CHECK:         ^bb8:
// CHECK:           [[VAL_12:%.*]] = addi [[VAL_6]], [[VAL_4]] : index
// CHECK:           br ^bb1([[VAL_12]] : index)
// CHECK:         ^bb9:
// CHECK:           llvm_tapir.sync [[VAL_13]], ^bb10
// CHECK:         ^bb10:
// CHECK:           return
// CHECK:         }

func @parallel_loop(%arg0 : index, %arg1 : index, %arg2 : index,
                        %arg3 : index, %arg4 : index) {
  %step = constant 1 : index
  scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
                                          step (%arg4, %step) {
    "unknown.op"() : () -> ()
  }
  return
}

