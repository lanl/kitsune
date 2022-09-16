# Copyright (c) 2020 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#
include(FetchContent)

set(KITSUNE_REALM_SOURCE_DIR ${LLVM_BINARY_DIR}/legion)
set(KITSUNE_LEGION_INCLUDE_DIR ${KITSUNE_REALM_SOURCE_DIR}/runtime)
set(KITSUNE_REALM_INCLUDE_DIR ${KITSUNE_LEGION_SRC_DIR}/realm)
set(KITSUNE_REALM_BINARY_DIR ${LLVM_BINARY_DIR}/runtimes/runtimes-bins/_deps/realm-build)
FetchContent_Declare(realm
  GIT_REPOSITORY   https://github.com/StanfordLegion/legion.git
  GIT_TAG          master
  SOURCE_DIR       ${KITSUNE_REALM_SOURCE_DIR}
  UPDATE_DISCONNECTED TRUE)
set(Legion_BUILD_REALM_ONLY OFF)
set(Legion_USE_LVVM OFF)
set(Legion_LINK_LLVM_LIBS OFF)
set(Legion_USE_CUDA OFF)
set(Legion_USE_HIP OFF)
FetchContent_MakeAvailable(realm)

set(KITSUNE_REALM_ABI_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_REALM_ABI_INCLUDE_DIR ${KITSUNE_REALM_ABI_DIR})

include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${Realm_INCLUDE_DIR})

set(REALM_ABI_SRCS realm_abi.cpp)
set(REALM_ABI_HEADERS realm_abi.h)

# TODO: Do we want an option for shared library support?
add_library(realm-abi STATIC ${REALM_ABI_SRCS})
target_include_directories(realm-abi PRIVATE
  ${KITSUNE_LEGION_INCLUDE_DIR}
  ${KITSUNE_REALM_INCLUDE_DIR}
  ${KITSUNE_REALM_BINARY_DIR}/runtime)
set_target_properties(realm-abi
  PROPERTIES
  # TODO: Not sure we need both these for PIC...
  POSITION_INDEPENDENT_CODE   TRUE
  COMPILE_FLAGS   "-fPIC")
#add_dependencies(realm-abi Realm RealmRuntime)
add_library(RealmABI ALIAS realm-abi)

install(TARGETS realm-abi
  LIBRARY DESTINATION lib COMPONENT realm-abi
  ARCHIVE DESTINATION lib COMPONENT realm-abi)

#if (KITSUNE_INCLUDE_TESTS)
#  add_subdirectory(tests)
#endif()
