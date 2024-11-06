// RUN: %kitxx -### -ftapir=cuda %s 2>&1 | FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-D_tapir_cuda_target"

// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
// CHECK: -lkitrt
// CHECK-SAME: -lcudart
// CHECK-SAME: -lcuda
