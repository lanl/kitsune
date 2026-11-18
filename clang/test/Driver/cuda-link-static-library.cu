// A link job made up only of objects and static libraries has no CUDA source
// input, so `--cuda-link` is what enables CUDA offloading for it. The
// requested architectures must then reach the linker wrapper so device code is
// extracted from the static libraries.

// RUN: touch %t.o %t.a
// RUN: %clang -### --target=x86_64-linux-gnu --cuda-link -nocudalib \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   --offload-arch=sm_35 --offload-arch=sm_70 %t.o %t.a 2>&1 \
// RUN: | FileCheck -check-prefix=CUDA-LINK %s

// CUDA-LINK: clang-linker-wrapper{{.*}}"--should-extract=sm_35"{{.*}}"--should-extract=sm_70"

// RUN: %clang -ccc-print-bindings --target=x86_64-linux-gnu --cuda-link \
// RUN:   -nocudalib --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   --offload-arch=sm_70 %t.o %t.a 2>&1 \
// RUN: | FileCheck -check-prefix=CUDA-LINK-BINDINGS %s

// CUDA-LINK-BINDINGS-NOT: "nvptx64-nvidia-cuda"
// CUDA-LINK-BINDINGS: "x86_64-unknown-linux-gnu" - "Offload::Linker", inputs: ["{{.*}}.o", "{{.*}}.a"], output: "a.out"

// Without `--cuda-link` nothing enables CUDA, so the architectures are unused.
// RUN: %clang -### --target=x86_64-linux-gnu --offload-link -nocudalib \
// RUN:   --cuda-path=%S/Inputs/CUDA/usr/local/cuda \
// RUN:   --offload-arch=sm_70 %t.o %t.a 2>&1 \
// RUN: | FileCheck -check-prefix=NO-CUDA-LINK %s

// NO-CUDA-LINK: clang-linker-wrapper
// NO-CUDA-LINK-NOT: --should-extract
