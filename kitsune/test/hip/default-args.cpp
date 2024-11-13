// RUN: %kitxx -### -ftapir=hip %s 2>&1 | FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-D_tapir_hip_target"

// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
// CHECK: -lkitrt
// CHECK-SAME: -lamdhip64
