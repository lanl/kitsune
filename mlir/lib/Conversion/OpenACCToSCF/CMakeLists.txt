add_mlir_conversion_library(MLIROpenACCToSCF
  OpenACCToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIROpenACC
  MLIRSCF
  MLIRTransforms
  )

mlir_tablegen(OpenACCToStandard.cpp.inc -gen-rewriters)
add_public_tablegen_target(OpenACCToStandardIncGen)

add_mlir_conversion_library(MLIROpenACCToStandard
  ConvertOpenACCConstraints.cpp
  OpenACCToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToStandard

  DEPENDS
  MLIRConversionPassIncGen
  OpenACCToStandardIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIROpenACC
  MLIRTensor
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
