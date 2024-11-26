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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static void usage(bool exitWithFailure = true) {
  errs() << "\
usage: kit-config <OPTION>... \n\
\n\
Get configuration information about Kitsune\n\
\n\
Options:\n\
  --c-frontend       Path to Kitsune's C frontend if it has been enabled\n\
  --c++-frontend     Path to Kitsune's C++ frontend if it has been enabled\n\
  --cuda-prefix      The cuda prefix (only relevant if the Cuda tapir target was bbuilt)\n\
  --cuda-target      Has the Cuda Tapir target been built (ON or OFF)\n\
  --fortran-frontend Path to Kitsune's Fortran frontend if it has been enabled\n\
  --frontends        The frontends that have been enabled\n\
  --help             Print a summary of kit-config arguments\n\
  --hip-prefix       The rocm prefix (only relevant if the Hip tapir target was built)\n\
  --hip-target       Has the Hip Tapir target been built (ON or OFF)\n\
  --kokkos-mode      Is Kokkos mode enabled (ON or OFF)\n\
  --opencilk-target  Has the OpenCilk Tapir target been built (ON or OFF)\n\
  --openmp-target    Has the OpenMP Tapir target been built (ON or OFF)\n\
  --qthreads-target  Has the Qthreads Tapir target been built (ON or OFF)\n\
  --realm-target     Has the Realm Tapir target been built (ON or OFF)\n\
  --tapir-targets    List all tapir targets that have been built\n\
  --version          Print LLVM version\n";
  if (exitWithFailure)
    exit(1);
}

// Get the path to the given frontend. @ref kitConfig is the full path to the
// this kitConfig executable. The frontend is assumed to be in the same
// directory as kitConfig.
static std::string frontendPath(StringRef kitConfig, StringRef frontend) {
  StringRef bin = sys::path::parent_path(kitConfig);
  SmallString<256> path(bin);

  sys::path::append(path, frontend);
  return path.c_str();
}

// Get the full path to this executable.
static std::string getExe(const char *argv0) {
  return sys::fs::getMainExecutable(argv0, (void *)(intptr_t)getExe);
}

int main(int argc, char **argv) {
  bool hasAnyOption = false;

  raw_ostream &os = outs();
  for (int i = 1; i != argc; ++i) {
    StringRef arg = argv[i];

    if (arg.starts_with("-")) {
      hasAnyOption = true;
      if (arg == "--version") {
        os << PACKAGE_VERSION << "\n";
      } else if (arg == "--help") {
        usage(false);
      } else if (arg == "--c-frontend") {
        if (KITSUNE_C_ENABLED)
          os << frontendPath(getExe(argv[0]), KITSUNE_C_FRONTEND) << "\n";
      } else if (arg == "--c++-frontend") {
        if (KITSUNE_CXX_ENABLED)
          os << frontendPath(getExe(argv[0]), KITSUNE_CXX_FRONTEND) << "\n";
      } else if (arg == "--cuda-prefix") {
        if (KITSUNE_CUDA_ENABLED)
          os << KITSUNE_CUDA_PREFIX << "\n";
      } else if (arg == "--cuda-target") {
        os << (KITSUNE_CUDA_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--fortran-frontend") {
        if (KITSUNE_Fortran_ENABLED)
          os << frontendPath(getExe(argv[0]), KITSUNE_Fortran_FRONTEND) << "\n";
      } else if (arg == "--frontends") {
        if (KITSUNE_C_ENABLED)
          os << "C ";
        if (KITSUNE_CXX_ENABLED)
          os << "C++ ";
        if (KITSUNE_Fortran_ENABLED)
          os << "Fortran ";
        os << "\n";
      } else if (arg == "--hip-prefix") {
        if (KITSUNE_HIP_ENABLED)
          os << KITSUNE_HIP_PREFIX << "\n";
      } else if (arg == "--hip-target") {
        os << (KITSUNE_HIP_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--kokkos-mode") {
        os << (KITSUNE_KOKKOS_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--opencilk-target") {
        os << (KITSUNE_OPENCILK_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--openmp-target") {
        os << (KITSUNE_OPENMP_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--qthreads-target") {
        os << (KITSUNE_QTHREADS_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--realm-target") {
        os << (KITSUNE_REALM_ENABLED ? "ON" : "OFF") << "\n";
      } else if (arg == "--tapir-targets") {
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
