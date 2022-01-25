//===- OpenACCToSCF.cpp - conversion from OpenACC to SCF dialect ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Conversion/OpenACCToSCF/OpenACCToSCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Pattern to transform the `ifCond` on operation without region into a scf.if
/// and move the operation into the `then` region.
template <typename OpTy>
class ExpandIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early exit if there is no condition.
    if (!op.ifCond())
      return success();

    // Condition is not a constant.
    if (!op.ifCond().template getDefiningOp<arith::ConstantOp>()) {
      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), TypeRange(),
                                             op.ifCond(), false);
      rewriter.updateRootInPlace(op, [&]() { op.ifCondMutable().erase(0); });
      auto thenBodyBuilder = ifOp.getThenBodyBuilder();
      thenBodyBuilder.setListener(rewriter.getListener());
      thenBodyBuilder.clone(*op.getOperation());
      rewriter.eraseOp(op);
    }

    return success();
  }
};
} // namespace

void mlir::populateOpenACCToSCFConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ExpandIfCondition<acc::EnterDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::ExitDataOp>>(patterns.getContext());
  patterns.add<ExpandIfCondition<acc::UpdateOp>>(patterns.getContext());
}

namespace {
struct ConvertOpenACCToSCFPass
    : public ConvertOpenACCToSCFBase<ConvertOpenACCToSCFPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertOpenACCToSCFPass::runOnOperation() {
  auto op = getOperation();
  auto *context = op.getContext();

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);
  populateOpenACCToSCFConversionPatterns(patterns);

  target.addLegalDialect<scf::SCFDialect>();
  target.addLegalDialect<acc::OpenACCDialect>();

  target.addDynamicallyLegalOp<acc::EnterDataOp>(
      [](acc::EnterDataOp op) { return !op.ifCond(); });

  target.addDynamicallyLegalOp<acc::ExitDataOp>(
      [](acc::ExitDataOp op) { return !op.ifCond(); });

  target.addDynamicallyLegalOp<acc::UpdateOp>(
      [](acc::UpdateOp op) { return !op.ifCond(); });

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertOpenACCToSCFPass() {
=======
/// Conversion patterns.
namespace {
class LoopOpConversion : public OpConversionPattern<acc::LoopOp> {
public:
  using OpConversionPattern<acc::LoopOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(acc::LoopOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

class ParallelOpConversion : public OpConversionPattern<acc::ParallelOp> {
public:
  using OpConversionPattern<acc::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(acc::ParallelOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

namespace {

class ConvertOpenACCToSCFPass
    : public ConvertOpenACCToSCFBase<ConvertOpenACCToSCFPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populateOpenACCToSCFConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target
        .addLegalDialect<scf::SCFDialect, StandardOpsDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

// Converts acc::parallel { acc::loop { scf::for { body } } } to scf::parallel { body }
LogicalResult
ParallelOpConversion::matchAndRewrite(acc::ParallelOp op, ArrayRef<Value> operands,
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
  
  /*
  scf::ParallelOp par = rewriter.create<scf::ParallelOp>(
    op.getLoc(), lowerBoundTuple, upperBoundTuple, steps); 

  BlockAndValueMapping map;
  for(auto dim : llvm::zip(ivs, par.getInductionVars())){
    Value iv, newiv;
    std::tie(iv, newiv) = dim; 
    map.map(iv, newiv);
  }

  rewriter.cloneRegionBefore(op.region(), par.region(), par.region().begin(), map); 
  rewriter.setInsertionPointToStart(parBody); 
  for(auto &op : *fop.getBody()){
    auto newop = rewriter.clone(op, map); 
    map.map(op.getResults(), newop->getResults()); 
  }
  */
  //rewriter.eraseOp(&*par.end()); 
  //rewriter.eraseBlock(fop.getBody()); 
  //rewriter.cloneRegionBefore(fop.region(), par.region(),
  //                           std::next(par.region().begin()), map);

  //rewriter.eraseOp(op); 

  //par.dump(); 

  return success(); 
}

// Not clear what a standalone loop means
LogicalResult
LoopOpConversion::matchAndRewrite(acc::LoopOp loop, ArrayRef<Value> operands,
                                 ConversionPatternRewriter &rewriter) const {

  return success();
}

void mlir::populateOpenACCToSCFConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ParallelOpConversion>(ctx);
}

std::unique_ptr<Pass>
mlir::createConvertOpenACCToSCFPass() {
>>>>>>> 5126676d858e (One dimensional openacc.loop to scf.parallel)
  return std::make_unique<ConvertOpenACCToSCFPass>();
}
