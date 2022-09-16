# Copyright (c) 2021 Triad National Security, LLC
#                         All rights reserved.
#
# This file is part of the kitsune/llvm project.  It is released under
# the LLVM license.
#
#

set(KITSUNE_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KITSUNE_RUNTIME_INCLUDE_DIR ${KITSUNE_RUNTIME_DIR})
option(KITRT_ENABLE_DEBUG "Enable debug mode features for the runtime." OFF)
option(KITRT_ENABLE_VERBOSE "Enable verbose mode features for the runtime." ON)

include_directories(${KITSUNE_SOURCE_DIR}/include)
include_directories(${KITSUNE_RUNTIME_INCLUDE_DIR})

if (KITRT_ENABLE_DEBUG)
  add_definitions(-DKITRT_DEBUG)
endif()

if (KITRT_ENABLE_VERBOSE)
  add_definitions(-D_KITRT_VERBOSE_)
endif()

set(KITRT_HDRS
    kitrt.h
    kitrt-debug.h)

set(KITRT_SRCS
    kitrt.cpp
    kitrt-debug.cpp
    kitrt-support.cpp)

if(KITSUNE_RT_ENABLE_CUDA)
  message(STATUS "kitsune runtime: CUDA support enabled...")
  find_package(CUDAToolkit REQUIRED)
  if(CUDAToolkit_FOUND)
    message(ERROR "kitsune runtime: Unable to find CUDAtoolkit!")
  endif()

  add_definitions(-DKITRT_CUDA_ENABLED)

  list(APPEND KITRT_HDRS
       kitcuda/cuda.h
       kitcuda/llvm-cuda.h)

  list(APPEND KITRT_SRCS
       kitcuda/cuda.cpp
       kitcuda/llvm-cuda.cpp)

  link_directories(${CUDAToolkit_LIBRARY_DIR})
  include_directories(${CUDAToolkit_INCLUDE_DIRS}
                      ${CMAKE_CURRENT_SOURCE_DIR}/kitcuda)
  set(KITRT_LINK_LIBS "-lnvptxcompiler_static -lcudart -lcuda")
  endif()

  add_library(kitrt SHARED
              ${KITRT_SRCS})

  target_include_directories(kitrt
    BEFORE PRIVATE
    ${LLVM_MAIN_INCLUDE_DIR}
    ${LLVM_INCLUDE_DIR}
    ${KITSUNE_SOURCE_DIR}/include
    ${KITSUNE_RUNTIME_INCLUDE_DIR}
    )

  target_link_libraries(kitrt
   PUBLIC
   LLVM -lrt -ldl -lpthread
   ${KITRT_LINK_LIBS})

  install(TARGETS kitrt
    LIBRARY DESTINATION lib COMPONENT kitrt
    ARCHIVE DESTINATION lib COMPONENT kitrt)

  install(FILES ${KITRT_HDRS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt)
  install(DIRECTORY kitcuda kithip kitrealm
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kitrt/
    FILES_MATCHING PATTERN "*.h"
    )


