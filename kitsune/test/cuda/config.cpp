// Check that the default target-specific configuration file is always found.
// RUN: %kitxx -### -ftapir=cuda %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT-CONFIG

// Check that providing a custom config directory without a target-specific
// configuration file is ok.
// RUN: %kitxx -### -ftapir=cuda --config-kitsune-dir=%S/../ %s 2>&1 | FileCheck %s -check-prefix=CHECK-CUSTOM-NOEXIST

// Check that providing a custom config directory with a target-specific
// configuration file leads to the file being found and the contents used and
// the default options are preserved.
// RUN: %kitxx -### -ftapir=cuda --config-kitsune-dir=%S %s 2>&1 | FileCheck %s -check-prefix=CHECK-CUSTOM

// CHECK-DEFAULT-CONFIG: Configuration file: {{.*}}/cuda.cfg
// CHECK-CUSTOM-NOEXIST-NOT: Configuration file: {{.*}}/cuda.cfg
// CHECK-CUSTOM: Configuration file: {{.*}}/cuda/cuda.cfg
// CHECK-CUSTOM: "-cc1"
// CHECK-CUSTOM-SAME: "-D" "some_preprocessor_flag"
// CHECK-CUSTOM-SAME: "-D_tapir_cuda_target"
// CHECK-CUSTOM-SAME: "-Wsome_compiler_flag"
// CHECK-CUSTOM: {{^[ ]*"[^"]+/[l]?}}ld{{[.]?[^ ]*}}"

// It is a pain to check for the actual linker executable. There are far too
// many options depending on the platform, so just check the next line for the
// expected linker flags.
// CHECK-CUSTOM: "-some_linker_flag"
// CHECK-CUSTOM-SAME: -lkitrt
// CHECK-CUSTOM-SAME: -lcudart
// CHECK-CUSTOM-SAME: -lcuda
