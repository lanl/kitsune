//===-- kit-config.cpp - Kitsune configuration utility --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tool encapsulates information about Kitsune's configuration. This is
// an analog of the LLVM project's llvm-config utility, but the scope is very
// narrow. For most things, llvm-config should be used.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Config/config.h"
#include "llvm/Config/config.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void usage(bool exitWithFailure = true) {
  errs() << "\
usage: kit-config <OPTION>... \n\
\n\
Get configuration information about Kitsune\n\
\n\
Options:\n\
  --cuda-prefix      The cuda prefix (only relevant if the Cuda tapir target was bbuilt)\n\
  --cuda-target      Has the Cuda Tapir target been built (ON or OFF).\n\
  --help             Print a summary of kit-config arguments.\n\
  --hip-prefix       The rocm prefix (only relevant if the Hip tapir target was built)\n\
  --hip-target       Has the Hip Tapir target been built (ON or OFF).\n\
  --kokkos-mode      Is Kokkos mode enabled (ON or OFF).\n\
  --opencilk-target  Has the OpenCilk Tapir target been built (ON or OFF).\n\
  --openmp-target    Has the OpenMP Tapir target been built (ON or OFF).\n\
  --qthreads-target  Has the Qthreads Tapir target been built (ON or OFF).\n\
  --realm-target     Has the Realm Tapir target been built (ON or OFF).\n\
  --tapir-targets    List all tapir targets that have been built.\n\
  --version          Print LLVM version.\n";
  if (exitWithFailure)
    exit(1);
}

int main(int argc, char **argv) {
  bool hasAnyOption = false;

  raw_ostream &os = outs();
  for (int i = 1; i != argc; ++i) {
    StringRef Arg = argv[i];

    if (Arg.starts_with("-")) {
      hasAnyOption = true;
      if (Arg == "--version") {
        os << PACKAGE_VERSION << "\n";
      } else if (Arg == "--help") {
        usage(false);
      } else if (Arg == "--cuda-prefix") {
        if (KITSUNE_CUDA_ENABLED)
          os << KITSUNE_CUDA_PREFIX << "\n";
      } else if (Arg == "--cuda-target") {
        os << (KITSUNE_CUDA_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--hip-prefix") {
        if (KITSUNE_HIP_ENABLED)
          os << KITSUNE_HIP_PREFIX << "\n";
      } else if (Arg == "--hip-target") {
        os << (KITSUNE_HIP_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--kokkos-mode") {
        os << (KITSUNE_KOKKOS_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--opencilk-target") {
        os << (KITSUNE_OPENCILK_ENABLED ? "ON" : "OFF")  << "\n";
      } else if (Arg == "--openmp-target") {
        os << (KITSUNE_OPENMP_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--qthreads-target") {
        os << (KITSUNE_QTHREADS_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--realm-target") {
        os << (KITSUNE_REALM_ENABLED ? "ON" : "OFF") << "\n";
      } else if (Arg == "--tapir-targets") {
        os << KITSUNE_ENABLED_TAPIR_TARGETS << "\n";
      } else {
        usage();
      }
    } else {
      usage();
    }
  }

  if (!hasAnyOption)
    usage();

  return 0;
}
