find_package(CUDAToolkit)

set(SOURCES
  DRFScopedNoAliasAA.cpp
  LambdaABI.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  OMPTaskABI.cpp
  Outline.cpp
  GPUABI.cpp
  SerializeSmallTasks.cpp
  Tapir.cpp
  TapirGPUUtils.cpp
  TapirToTarget.cpp
  TapirLoopInfo.cpp
)

if (KITSUNE_CUDA_ENABLED)
  list(APPEND SOURCES CudaABI.cpp)
endif ()

if (KITSUNE_HIP_ENABLED)
  list(APPEND SOURCES HipABI.cpp)
endif ()

if (KITSUNE_OPENCILK_ENABLED)
  list(APPEND SOURCES OpenCilkABI.cpp)
endif ()

if (KITSUNE_OPENMP_ENABLED)
  list(APPEND SOURCES OpenMPABI.cpp)
endif ()

if (KITSUNE_QTHREADS_ENABLED)
  list(APPEND SOURCES QthreadsABI.cpp)
endif ()

if (KITSUNE_REALM_ENABLED)
  list(APPEND SOURCES RealmABI.cpp)
endif ()

# Some ABI targets are always enabled.
list(APPEND SOURCES SerialABI.cpp)

# TODO: Perhaps we need to do the same with LambdaABI and OMPTaskABI.

add_llvm_component_library(LLVMTapirOpts
  ${SOURCES}

  # LLVM discourages the use of this, but we definitely don't want to build
  # ABI targets unnecessarily (besides which, getting them to compile can be
  # fairly difficult without target-specific headers such as those from a Cuda
  # installation which may be in a non-standard location). Plus, they use them
  # in certain places anyway.
  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Tapir

  DEPENDS
  intrinsics_gen
  LLVMLinker
  LLVMBitWriter

  COMPONENT_NAME
  TapirOpts

  LINK_COMPONENTS
  Analysis
  Core
  IRReader
  Linker
  MC
  Scalar
  Support
  TransformUtils
  Vectorize
  Passes
)

if (CUDAToolkit_FOUND)
  message(STATUS "Tapir ABI has CUDA version info: ${CUDAToolkit_VERSION}")
  add_compile_definitions(
    CUDATOOLKIT_VERSION=${CUDAToolkit_VERSION}
    CUDATOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    CUDATOOLKIT_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}
  )
  set(TAPIR_CUDAABI_DEFAULT_TARGET "sm_80" CACHE STRING
	 "The default CUDA device target for Tapir-transformed code")
  add_compile_definitions(
    _CUDAABI_DEFAULT_TARGET_ARCH="${TAPIR_CUDAABI_DEFAULT_TARGET}"
  )
endif()
