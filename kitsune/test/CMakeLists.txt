# Test runner infrastructure for Kitsune. This configures the Kitsune test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

llvm_canonicalize_cmake_booleans(
  KITSUNE_KOKKOS_ENABLED
  KITSUNE_CUDA_ENABLED
  KITSUNE_HIP_ENABLED
  KITSUNE_OPENCILK_ENABLED
  KITSUNE_OPENMP_ENABLED
  KITSUNE_QTHREADS_ENABLED
  KITSUNE_REALM_ENABLED
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  PATHS
  "LLVM_SOURCE_DIR"
  "LLVM_BINARY_DIR"
  "LLVM_TOOLS_DIR"
  "LLVM_LIBS_DIR"
  "SHLIBDIR"
  "LLVM_LIT_TOOLS_DIR"
  "LLVM_EXTERNAL_LIT"
  "KITSUNE_BINARY_DIR"
  "KITSUNE_SOURCE_DIR"
  "CURRENT_TOOLS_DIR"
  "CMAKE_LIBRARY_OUTPUT_DIRECTORY"
)

set(KITSUNE_TEST_PARAMS
  USE_Z3_SOLVER=0
)

list(APPEND KITSUNE_TEST_DEPS
  ${KITSUNE_C_FRONTEND}
  ${KITSUNE_CXX_FRONTEND}
  clang
  llvm-config
  FileCheck count not
  opt
)

if(TARGET llvm-lto)
  list(APPEND KITSUNE_TEST_DEPS llvm-lto)
endif()

add_custom_target(kitsune-test-depends DEPENDS ${KITSUNE_TEST_DEPS})
set_target_properties(kitsune-test-depends PROPERTIES FOLDER "Kitsune tests")

add_lit_testsuite(check-kitsune "Running the Kitsune regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS ${KITSUNE_TEST_PARAMS}
  DEPENDS ${KITSUNE_TEST_DEPS}
)
set_target_properties(check-kitsune PROPERTIES FOLDER "Kitsune tests")

add_lit_testsuites(KITSUNE ${CMAKE_CURRENT_SOURCE_DIR}
  PARAMS ${KITSUNE_TEST_PARAMS}
  DEPENDS ${KITSUNE_TEST_DEPS}
  FOLDER "Kitsune tests"
)
