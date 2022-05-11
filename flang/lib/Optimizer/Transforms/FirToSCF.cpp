//===- FirToSCF.cpp - conversion from Fir to SCF dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FirToSCF/FirToSCF.h"

//#include "PassDetail.h"
//#include "mlir/IR/BlockAndValueMapping.h"
//#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
//delete this #include "mlir/Dialect/OpenACC/OpenACC.h"
//fail #include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Conversion patterns.
/*namespace {
class LoopOpConversion : public OpConversionPattern<fir::DoLoopOp> {
public:
  using OpConversionPattern<fir::DoLoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(fir::DoLoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class ParallelOpConversion : public OpConversionPattern<fir::DoConcurrentLoopOp> {
public:
  using OpConversionPattern<fir::DoConcurrentLoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(fir::DoConcurrentLoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

namespace {

class ConvertFirToSCFPass
    : public ConvertFirToSCFBase<ConvertFirToSCFPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateFirToSCFConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target
        .addLegalDialect<scf::SCFDialect, StandardOpsDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//Will need to change this to a while loop because there is no collapse value.
// Converts acc::parallel { acc::loop { scf::for { body } } } to scf::parallel { body }
LogicalResult
ParallelOpConversion::matchAndRewrite(fir::DoConcurrentLoopOp op, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // We only continue if first op in the loop is an scf::for and the first op in
  // a parallel is a loop. If not, the transform is a noop
  auto loop = dyn_cast<acc::LoopOp>(op.region().front().begin());
  if(!loop) return success();
  auto fop = dyn_cast<scf::ForOp>(loop.region().front().begin());
  if(!fop) return success(); 

  SmallVector<Value, 8> steps = {fop.step()} ; 
  SmallVector<Value, 8> ivs = {fop.getInductionVar()};  
  SmallVector<Value, 8> upperBoundTuple = {fop.upperBound()};
  SmallVector<Value, 8> lowerBoundTuple = {fop.lowerBound()};
  
  if(auto collapse = loop.collapse()){ 
    for(uint64_t i=0; i < *collapse - 1; i++){
      fop = dyn_cast<scf::ForOp>(fop.region().front().begin());
      if(!fop) return failure();
      steps.push_back(fop.step());
      upperBoundTuple.push_back(fop.upperBound());
      lowerBoundTuple.push_back(fop.lowerBound()); 
      ivs.push_back(fop.getInductionVar()); 
    }
  }

  // Both scf::ForOps and scf::ParallelOps must be single-block, so we only need
  // to clone that block.
  scf::ParallelOp par = rewriter.create<scf::ParallelOp>(
    op.getLoc(), lowerBoundTuple, upperBoundTuple, steps, op.reductionOperands(), 
    [&](OpBuilder& ob, Location l, ValueRange newivs, ValueRange otherargs){
      BlockAndValueMapping map;
      for(auto dim : llvm::zip(ivs, newivs)){
        Value iv, newiv;
        std::tie(iv, newiv) = dim; 
        map.map(iv, newiv);
      }
      map.map(fop.getBody(), ob.getBlock()); 
      for(auto &op : *fop.getBody()){
        auto newop = ob.clone(op, map); 
        map.map(op.getResults(), newop->getResults()); 
      }
    }); 

  rewriter.replaceOp(op, par.results());

  return success(); 
}

// Not clear what a standalone loop means
LogicalResult
LoopOpConversion::matchAndRewrite(acc::LoopOp loop, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {

  return success();
}

void mlir::populateFirToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<Pass>
mlir::createConvertFirToSCFPass() {
  return std::make_unique<ConvertFirToSCFPass>();
}*/