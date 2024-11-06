// RUN: %kitxx -### -ftapir=opencilk %s 2>&1 | FileCheck %s

// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
// CHECK: -lopencilk
// CHECK-SAME: -lkitrt
