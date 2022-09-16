find_package(CUDAToolkit)

add_llvm_component_library(LLVMTapirOpts
  CilkABI.cpp
  CilkRTSCilkFor.cpp
  CudaABI.cpp
  DRFScopedNoAliasAA.cpp
  LoopSpawningTI.cpp
  LoopStripMine.cpp
  LoopStripMinePass.cpp
  LoweringUtils.cpp
  OpenCilkABI.cpp
  OpenMPABI.cpp
  OpenCLABI.cpp
  Outline.cpp
  QthreadsABI.cpp
  RealmABI.cpp
  GPUABI.cpp
  SerialABI.cpp
  SerializeSmallTasks.cpp
  Tapir.cpp
  TapirToTarget.cpp
  TapirLoopInfo.cpp

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
  Scalar
  Support
  TransformUtils
  Vectorize

  )

if (CUDAToolkit_FOUND)
  message(STATUS "Tapir ABI has CUDA version info: ${CudaToolkit_VERSION}")
  add_compile_definitions(
    CUDATOOLKIT_VERSION=${CUDAToolkit_VERSION}
    CUDATOOLKIT_VERSION_MAJOR=${CUDAToolkit_VERSION_MAJOR}
    CUDATOOLKIT_VERSION_MINOR=${CUDAToolkit_VERSION_MINOR}
  )
endif()

if(KITSUNE_ENABLE_OPENCL_TARGET)
  target_link_libraries(LLVMTapirOpts LLVMSPIRVLib)
endif()
