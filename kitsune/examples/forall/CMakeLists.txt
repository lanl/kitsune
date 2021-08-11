
#
# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#

# A small collection of forall examples. 
#
# Note: Internal builds of these examples are not necessarily 
# geared for benchmarking and performance comparisons as they
# currently inherit build flags from the larger llvm 
# configuration. 

project(ForAllExamples C CXX)

set(_ld_library_path "")
set(_llvm_version "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
message(STATUS "ForAll examples: LLVM version: ${_llvm_version}")
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

# Build a version of the examples for each runtime target that has 
# been enabled. We can get a list of all the support runtime targets
# in a format that is compatible for use with -ftapir=target. 
get_tapir_rt_targets(_tapir_rt_targets)

# Manually add the serial target to our testing scope... 
list(APPEND _tapir_rt_targets "serial") 

message(STATUS "  enabled runtime targets: ${_tapir_rt_targets}")
foreach(rt IN ITEMS ${_tapir_rt_targets})
  foreach(src_file IN ITEMS ${_example_srcs})
    get_filename_component(exec_name ${src_file} NAME_WLE)
    set(exec_name "forall.${exec_name}.${rt}")
    message(STATUS "  ${src_file} --> ${exec_name}")
    add_executable(${exec_name} ${src_file})
    target_include_directories(${exec_name} 
      BEFORE 
      PRIVATE
      ${KITSUNE_INCLUDE_DIR})
    target_compile_options(${exec_name} 
      PRIVATE
      -v
      -ftapir=${rt}
      --config ${LLVM_BINARY_DIR}/share/kitsune/${rt}.cfg)
    target_link_options(${exec_name} 
      PRIVATE 
      -ftapir=${rt}
      --config ${LLVM_BINARY_DIR}/share/kitsune/${rt}.cfg)
    # TODO: for runtimes build within the toolchain components we
    # really need to add a set of libraries per runtime. Ideally 
    # the configure files will pick this up but it seems to still 
    # be a bit brittle. 
    target_link_libraries(${exec_name})
    set_target_properties(${exec_name}
      PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY  ${LLVM_BINARY_DIR}/examples/kitsune)
    #  
    install(TARGETS ${exec_name}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/kitsune/examples/)
  endforeach()
endforeach()
# Do some clean up... 
unset(_example_srcs)
unset(_tapir_rt_targets)

if (_save_ld_library_path)
  set(ENV{LD_LIBRARY_PATH} ${_save_ld_library_path})
  unset(_save_ld_library_path)
endif()
