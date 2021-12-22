//===- OpenACCToSCF.h - Conversion utils from shape to std dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_OPENACCTOSCF_H_
#define MLIR_CONVERSION_OPENACCTOSCF_H_

#include <memory>

namespace mlir {

class Pass; 
class FuncOp;
class MLIRContext;
class ModuleOp;
template <typename T>
class OperationPass;
class OwningRewritePatternList;

void populateOpenACCToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

std::unique_ptr<Pass> createConvertOpenACCToSCFPass();

void populateConvertOpenACCConstraintsConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOSCF_H_
