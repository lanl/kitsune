#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#

# TODO: This is geared for an in-tree build, need to update if we 
# want to enable building examples independently of the overall 
# toolchain. 
#
# A small collection of Kokkos examples.
#
# NOTE: Internal builds of these examples are not geared for 
# benchmarking and performance comparisons for numerous reasons
# (e.g., kokkos is only configure in the serial mode within the 
# internal builds, compiler flags may not have all the desired 
# impacts in how they are set vs. LLVM's overall build settings, 
# etc.).
# 
# TODO: There are some cases where we can not successfully use 
# the -fkokkos-no-init flag.  There is currently no easy way to 
# distinguish these caes without some more cmake work.  The work
# to address this from within the comiler is currently something 
# we need to work on to make supporting both examples and real 
# codes successfully... 
project(KokkosExamples CXX)

set(_ld_library_path "")
set(_llvm_version "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
message(STATUS "Kokkos examples: LLVM version: ${_llvm_version}")
message(STATUS "   LLVM target triple: ${LLVM_DEFAULT_TARGET_TRIPLE}")

# TODO: Not sure this (always? ever?) works.
set(_new_ld_lib_path ${LLVM_BINARY_DIR}/lib/clang/${_llvm_version}/lib/${LLVM_DEFAULT_TARGET_TRIPLE})
if (DEFINED ENV{LD_LIBRARY_PATH})
  set(_save_ld_library_path $ENV{LD_LIBRARY_PATH})
  string(FIND _save_ld_library_path ${_new_ld_lib_path} _ld_index)
  if (_ld_index EQUAL -1)
    set(_ld_library_path $ENV{LD_LIBRARY_PATH})
    string(APPEND _ld_library_path ":")
  endif()
endif()
string(APPEND _ld_library_path ${_new_ld_lib_path})
set(ENV{LD_LIBRARY_PATH} ${_ld_library_path})
message(STATUS "setting LD_LIBRARY_PATH: ${_ld_library_path}.")
unset(_ld_library_path)
unset(_llvm_version)
unset(_new_ld_lib_path)

set(_example_srcs
  complex.cpp
  matmult1.cpp
  normalize.cpp
#  raytrace.cpp
  vecadd.cpp)

# Build the pure-kokkos serial target version of all the examples first.
# NOTE: Do not build kokkos with parallel "back-ends" as part of the 
# overall kitsune+tapir configure and build.  For performance comparisons 
# and other related work you should use a separate kokkos build. 
foreach(src_file IN ITEMS ${_example_srcs})
  get_filename_component(exec_name kokkos.${src_file} NAME_WLE)
  message(STATUS "  ${src_file} --> ${exec_name}")
  add_executable(${exec_name} ${src_file})
  target_include_directories(${exec_name} 
    BEFORE 
    PRIVATE 
    # ugh... uninstalled kokkos paths are a bit painful... 
    ${KITSUNE_INCLUDE_DIR}
    ${KITSUNE_KOKKOS_SRC_DIR}/core/src
    ${KITSUNE_KOKKOS_SRC_DIR}/containers/src)
  target_link_libraries(${exec_name} PRIVATE kokkoscore)
  set_target_properties(${exec_name} 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/examples/kitsune/)
  add_dependencies(${exec_name} kokkos)
  #
  install(TARGETS ${exec_name} 
          DESTINATION ${CMAKE_INSTALL_PREFIX}/kitsune/examples/)
endforeach()

# Build a version of the examples for each runtime target that has 
# been enabled. We can get a list of all the support runtime targets
# in a format that is compatible for use with -ftapir=target. 
get_tapir_rt_targets(_tapir_rt_targets)

message(STATUS "  enabled runtime targets: ${_tapir_rt_targets}")
foreach(rt IN ITEMS ${_tapir_rt_targets})
  foreach(src_file IN ITEMS ${_example_srcs})
    get_filename_component(exec_name ${src_file} NAME_WLE)
    set(exec_name "kokkos.${exec_name}.${rt}")
    message(STATUS "  ${src_file} --> ${exec_name}")
    add_executable(${exec_name} ${src_file})
    target_include_directories(${exec_name} 
      BEFORE 
      PRIVATE
      ${KITSUNE_INCLUDE_DIR}
      ${KITSUNE_KOKKOS_SRC_DIR}/core/src
      ${KITSUNE_KOKKOS_SRC_DIR}/containers/src)
    target_compile_options(${exec_name} 
      PRIVATE
      -v
      -fkokkos
      -fkokkos-no-init 
      -ftapir=${rt}
      --config ${LLVM_BINARY_DIR}/share/kitsune/${rt}.cfg)
    target_link_options(${exec_name} 
      PRIVATE 
      -ftapir=${rt}
      --config ${LLVM_BINARY_DIR}/share/kitsune/${rt}.cfg)
    add_tapir_dependency(${exec_name} ${rt} tapir_target_libs)
    add_dependencies(${exec_name} kokkos)
    target_link_libraries(${exec_name} kokkoscore ${tapir_target_libs})
    set_target_properties(${exec_name}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY  ${LLVM_BINARY_DIR}/examples/kitsune)
    #  
    install(TARGETS ${exec_name}
      DESTINATION ${LLVM_BINARY_DIR}/kitsune/examples/)
  endforeach()
endforeach()

# Do some clean up... 
unset(_example_srcs)
unset(_tapir_rt_targets)

if (_save_ld_library_path)
  set(ENV{LD_LIBRARY_PATH} ${_save_ld_library_path})
  unset(_save_ld_library_path)
endif()

