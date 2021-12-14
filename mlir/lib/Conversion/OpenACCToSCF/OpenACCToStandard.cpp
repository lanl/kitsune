//===- OpenACCToStandard.cpp - conversion from OpenACC to Standard dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToStandard/OpenACCToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/OpenACC/IR/OpenACC.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::acc;
using namespace mlir::scf;

/// Conversion patterns.
namespace {
class LoopOpConversion : public OpConversionPattern<LoopOp> {
public:
  using OpConversionPattern<LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult
LoopOpConversion::matchAndRewrite(LoopOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  LoopOp::Adaptor transformed(operands);

  // TODO:Create a scf::ParallelOp from the LoopOp
  rewriter.replaceOp(op, {transformed.inputs().front()});
  return success();
}


void mlir::populateOpenACCToStandardConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  populateWithGenerated(ctx, patterns);
  patterns.insert<
      LoopOpConversion,
      >;
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertOpenACCToStandardPass() {
  return std::make_unique<ConvertOpenACCToStandardPass>();
}
