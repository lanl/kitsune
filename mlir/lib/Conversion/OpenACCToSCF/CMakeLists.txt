mlir_tablegen(OpenACCToStandard.cpp.inc -gen-rewriters)
add_public_tablegen_target(OpenACCToStandardIncGen)

add_mlir_conversion_library(MLIROpenACCToStandard
  ConvertOpenACCConstraints.cpp
  OpenACCToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/OpenACCToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIROpenACC
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
