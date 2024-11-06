#
# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
cmake_minimum_required(VERSION 3.20)

if (KITSUNE_KOKKOS_ENABLED)
  configure_file(kokkos.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/kokkos.cfg)
endif ()

configure_file(none.cfg.in
  ${LLVM_BINARY_DIR}/share/kitsune/none.cfg)

configure_file(serial.cfg.in
  ${LLVM_BINARY_DIR}/share/kitsune/serial.cfg)

if (KITSUNE_CUDA_ENABLED)
  configure_file(cuda.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/cuda.cfg)
endif()

if (KITSUNE_HIP_ENABLED)
  configure_file(hip.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/hip.cfg)
endif()

if (KITSUNE_OPENCILK_ENABLED)
  configure_file(opencilk.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/opencilk.cfg)
endif()

if (KITSUNE_OPENMP_ENABLED)
  configure_file(openmp.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/openmp.cfg)
endif()

if (KITSUNE_QTHREADS_ENABLED)
  configure_file(qthreads.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/qthreads.cfg)
endif()

if (KITSUNE_REALM_ENABLED)
  configure_file(realm.cfg.in
    ${LLVM_BINARY_DIR}/share/kitsune/realm.cfg)
endif()

install(DIRECTORY ${LLVM_BINARY_DIR}/share/kitsune/
  DESTINATION share/kitsune
  COMPONENT install-clang-cfgs
  FILES_MATCHING
  PATTERN "*.cfg")
