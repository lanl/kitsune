/*
 * Copyright (c) 2020 Triad National Security, LLC
 *                         All rights reserved.
 *
 * This file is part of the kitsune/llvm project.  It is released under
 * the LLVM license.
 */

#ifndef KITSUNE_CONFIG_H
#define KITSUNE_CONFIG_H

#define KITSUNE_VERSION_MAJOR @KITSUNE_VERSION_MAJOR@
#define KITSUNE_VERSION_MINOR @KITSUNE_VERSION_MINOR@
#define KITSUNE_VERSION_PATCH @KITSUNE_VERSION_PATCH@
#define KITSUNE_PACKAGE_VERSION "@KITSUNE_PACKAGE_VERSION@"

// -----------------------------------------------------------------------------

#define KITSUNE_LIBNAME "@KITSUNE_LIBNAME@"
#define KITSUNE_LIBNAME_STATIC "@KITSUNE_LIBNAME_STATIC@"

// The default GCC installation directory to use. If this is non-null, Kitsune
// will not scan for a GCC installation. This value can be overridden by passing
// --gcc-install-dir on the command line
#define KITSUNE_GCC_INSTALL_DIR "@KITSUNE_GCC_INSTALL_DIR@"

// When using a fixed number of threads per block, the maximum value that can be
// specified.
constexpr unsigned KITSUNE_MAX_FIXED_THREADS_PER_BLOCK = 1024;

// -----------------------------------------------------------------------------

// A space-separated string of all known Tapir targets, including those that
// have not been enabled
#define KITSUNE_KNOWN_TAPIR_TARGETS "@KITSUNE_KNOWN_TAPIR_TARGETS_STR@"

// A space-separated string with the Tapir targets that have been enabled in
// this build
#define KITSUNE_ENABLED_TAPIR_TARGETS "@KITSUNE_ENABLED_TAPIR_TARGETS_STR@"

// Kitsune language support that has been enabled
#cmakedefine01 KITSUNE_C_ENABLED
#cmakedefine01 KITSUNE_CXX_ENABLED
#cmakedefine01 KITSUNE_Fortran_ENABLED

// The names of the Kitsune frontends
#define KITSUNE_C_FRONTEND "@KITSUNE_C_FRONTEND@"
#define KITSUNE_CXX_FRONTEND "@KITSUNE_CXX_FRONTEND@"
#define KITSUNE_Fortran_FRONTEND "@KITSUNE_Fortran_FRONTEND@"

// -----------------------------------------------------------------------------

// A space-separated string of all languages for which Kitsune frontends exist
#define KITSUNE_KNOWN_LANGS "@KITSUNE_KNOWN_LANGS_STR@"

// A space-separate dstring of all enabled languages
#define KITSUNE_ENABLED_LANGS "@KITSUNE_ENABLED_LANGS_STR@"

// -----------------------------------------------------------------------------

// Has kokkos support been enabled
#cmakedefine01 KITSUNE_KOKKOS_ENABLED

// Extra preprocessor flags needed when compiling in kokkos mode.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_KOKKOS_EXTRA_PREPROCESSOR_FLAGS                                \
  "@KITSUNE_KOKKOS_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling in kokkos mode.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_KOKKOS_EXTRA_COMPILER_FLAGS                                    \
  "@KITSUNE_KOKKOS_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling in kokkos mode.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_KOKKOS_EXTRA_LINKER_FLAGS "@KITSUNE_KOKKOS_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// A space-separated string of all known Tapir targets, including those that
// have not been enabled
#define KITSUNE_ALL_TAPIR_TARGETS "@KITSUNE_ALL_TAPIR_TARGETS_STR@"

// A space-separated string with the Tapir targets that have been enabled in
// this build
#define KITSUNE_ENABLED_TAPIR_TARGETS "@KITSUNE_ENABLED_TAPIR_TARGETS_STR@"

// -----------------------------------------------------------------------------

// Is the cuda tapir target enabled
#cmakedefine01 KITSUNE_CUDA_ENABLED

// Extra preprocessor flags needed when compiling with the cuda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_CUDA_EXTRA_PREPROCESSOR_FLAGS                                  \
  "@KITSUNE_CUDA_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the cuda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_CUDA_EXTRA_COMPILER_FLAGS "@KITSUNE_CUDA_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the cuda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_CUDA_EXTRA_LINKER_FLAGS "@KITSUNE_CUDA_EXTRA_LINKER_FLAGS@"

#define KITSUNE_CUDA_PREFIX "@KITSUNE_CUDA_PREFIX@"
#define KITSUNE_CUDA_VERSION_MAJOR @KITSUNE_CUDA_VERSION_MAJOR@
#define KITSUNE_CUDA_VERSION_MINOR @KITSUNE_CUDA_VERSION_MINOR@
#define KITSUNE_CUDA_VERSION_PATCH @KITSUNE_CUDA_VERSION_PATCH@
#define KITSUNE_CUDA_VERSION_STRING "@KITSUNE_CUDA_VERSION_STRING@"
#define KITSUNE_CUDA_BINARY_DIR "@KITSUNE_CUDA_BINARY_DIR@"
#define KITSUNE_CUDA_LIBRARY_DIR "@KITSUNE_CUDA_LIBRARY_DIR@"
#define KITSUNE_CUDA_LIBCUDA_DIR "@KITSUNE_CUDA_LIBCUDA_DIR@"
#define KITSUNE_CUDA_PTXAS "@KITSUNE_CUDA_PTXAS@"
#define KITSUNE_CUDA_FATBINARY "@KITSUNE_CUDA_FATBINARY@"
#define KITSUNE_CUDA_ARCH_DEFAULT "@KITSUNE_CUDA_ARCH_DEFAULT@"

#define KITSUNE_CUDA_LIBNAME_CUDA "@KITSUNE_CUDA_LIBNAME_CUDA@"
#define KITSUNE_CUDA_LIBNAME_CUDART "@KITSUNE_CUDA_LIBNAME_CUDART@"
#define KITSUNE_CUDA_LIBNAME_CUDART_STATIC                                     \
  "@KITSUNE_CUDA_LIBNAME_CUDART_STATIC@"

// The name of the symbol containing the cuda device code (the fat binary)
#define KITSUNE_CUDA_FATBIN_NAME "__kitsune_fatbin_cuda"

// -----------------------------------------------------------------------------

// Is the hip tapir target enabled
#cmakedefine01 KITSUNE_HIP_ENABLED

// Extra preprocessor flags needed when compiling with the hip tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_HIP_EXTRA_PREPROCESSOR_FLAGS                                   \
  "@KITSUNE_HIP_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the hip tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_HIP_EXTRA_COMPILER_FLAGS "@KITSUNE_HIP_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the hip tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_HIP_EXTRA_LINKER_FLAGS "@KITSUNE_HIP_EXTRA_LINKER_FLAGS@"

#define KITSUNE_HIP_PREFIX "@KITSUNE_HIP_PREFIX@"
#define KITSUNE_HIP_VERSION_MAJOR @KITSUNE_HIP_VERSION_MAJOR@
#define KITSUNE_HIP_VERSION_MINOR @KITSUNE_HIP_VERSION_MINOR@
#define KITSUNE_HIP_VERSION_PATCH @KITSUNE_HIP_VERSION_PATCH@
#define KITSUNE_HIP_VERSION_STRING "@KITSUNE_HIP_VERSION_STRING@"
#define KITSUNE_HIP_LIBRARY_DIR "@KITSUNE_HIP_LIBRARY_DIR@"
#define KITSUNE_HIP_BITCODE_DIR "@KITSUNE_HIP_BITCODE_DIR@"
#define KITSUNE_HIP_ARCH_DEFAULT "@KITSUNE_HIP_ARCH_DEFAULT@"

#define KITSUNE_HIP_LIBNAME_AMDHIP "@KITSUNE_HIP_LIBNAME_AMDHIP@"

// The name of the symbol containing the hip device code (the fat binary)
#define KITSUNE_HIP_FATBIN_NAME "__kitsune_fatbin_hip"

// -----------------------------------------------------------------------------

// Is the lambda tapir target enabled
#cmakedefine01 KITSUNE_LAMBDA_ENABLED

// Extra preprocessor flags needed when compiling with the lambda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_LAMBDA_EXTRA_PREPROCESSOR_FLAGS                                \
  "@KITSUNE_LAMBDA_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the lambda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_LAMBDA_EXTRA_COMPILER_FLAGS                                    \
  "@KITSUNE_LAMBDA_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the lambda tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_LAMBDA_EXTRA_LINKER_FLAGS "@KITSUNE_LAMBDA_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// Is the omptask tapir target enabled
#cmakedefine01 KITSUNE_OMPTASK_ENABLED

// Extra preprocessor flags needed when compiling with the omptask tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OMPTASK_EXTRA_PREPROCESSOR_FLAGS                               \
  "@KITSUNE_OMPTASK_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the omptask tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OMPTASK_EXTRA_COMPILER_FLAGS                                   \
  "@KITSUNE_OMPTASK_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the omptask tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OMPTASK_EXTRA_LINKER_FLAGS                                     \
  "@KITSUNE_OMPTASK_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// Is the opencilk tapir target enabled
#cmakedefine01 KITSUNE_OPENCILK_ENABLED

// Extra preprocessor flags needed when compiling with the opencilk tapir
// target. These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENCILK_EXTRA_PREPROCESSOR_FLAGS                              \
  "@KITSUNE_OPENCILK_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the opencilk tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENCILK_EXTRA_COMPILER_FLAGS                                  \
  "@KITSUNE_OPENCILK_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the opencilk tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENCILK_EXTRA_LINKER_FLAGS                                    \
  "@KITSUNE_OPENCILK_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// is the openmp tapir target enabled
#cmakedefine01 KITSUNE_OPENMP_ENABLED

// Extra preprocessor flags needed when compiling with the openmp tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS                                \
  "@KITSUNE_OPENMP_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the openmp tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS                                    \
  "@KITSUNE_OPENMP_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the openmp tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_OPENMP_EXTRA_LINKER_FLAGS "@KITSUNE_OPENMP_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// Is the qthreads tapir target enabled
#cmakedefine01 KITSUNE_QTHREADS_ENABLED

// Extra preprocessor flags needed when compiling with the qthreads tapir
// target. These are in addition to any flags automatically added by Kitsune
#define KITSUNE_QTHREADS_EXTRA_PREPROCESSOR_FLAGS                              \
  "@KITSUNE_QTHREADS_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the qthreads tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_QTHREADS_EXTRA_COMPILER_FLAGS                                  \
  "@KITSUNE_QTHREADS_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the qthreads tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_QTHREADS_EXTRA_LINKER_FLAGS                                    \
  "@KITSUNE_QTHREADS_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// Is the realm tapir target enabled
#cmakedefine01 KITSUNE_REALM_ENABLED

// Extra preprocessor flags needed when compiling with the realm tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_REALM_EXTRA_PREPROCESSOR_FLAGS                                 \
  "@KITSUNE_REALM_EXTRA_PREPROCESSOR_FLAGS@"

// Extra compiler flags needed when compiling with the realm tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_REALM_EXTRA_COMPILER_FLAGS                                     \
  "@KITSUNE_REALM_EXTRA_COMPILER_FLAGS@"

// Extra linker flags needed when compiling with the realm tapir target.
// These are in addition to any flags automatically added by Kitsune
#define KITSUNE_REALM_EXTRA_LINKER_FLAGS "@KITSUNE_REALM_EXTRA_LINKER_FLAGS@"

// -----------------------------------------------------------------------------

// Is the gpu target enabled 
#cmakedefine01 KITSUNE_GPU_ENABLED

#endif // KITSUNE_CONFIG_H
