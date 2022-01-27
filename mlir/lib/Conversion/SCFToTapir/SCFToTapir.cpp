//===- SCFToTapir.cpp - ControlFlow to CFG conversion ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.for, scf.if and loop.terminator
// ops into standard CFG ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToTapir/SCFToTapir.h"
#include "../PassDetail.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTapirDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToTapirPass : public SCFToTapirBase<SCFToTapirPass> {
  void runOnOperation() override;
};

}

struct ParallelLowering : public OpRewritePattern<mlir::scf::ParallelOp> {
  using OpRewritePattern<mlir::scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::scf::ParallelOp parallelOp,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
ParallelLowering::matchAndRewrite(ParallelOp parallelOp,
                                  PatternRewriter &rewriter) const {
  Location loc = parallelOp.getLoc();
  SmallVector<ForOp, 4> forLoops; 

  auto *ctx = parallelOp.getContext(); 
  auto sr = rewriter.create<LLVM::Tapir_createsyncregion>(loc, LLVM::LLVMTokenType::get(ctx)); 
  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to scf.for ops and have those lowered in
  // a further rewrite. If a parallel loop contains reductions (and thus returns
  // values), forward the initial values for the reductions down the loop
  // hierarchy and bubble up the results by modifying the "yield" terminator.
  SmallVector<Value, 4> iterArgs = llvm::to_vector<4>(parallelOp.initVals());
  SmallVector<Value, 4> ivs;
  ivs.reserve(parallelOp.getNumLoops());
  bool first = true;
  SmallVector<Value, 4> loopResults(iterArgs);
  for (auto loop_operands :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.lowerBound(),
                 parallelOp.upperBound(), parallelOp.step())) {
    Value iv, lower, upper, step;
    std::tie(iv, lower, upper, step) = loop_operands;
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step, iterArgs);
    ivs.push_back(forOp.getInductionVar());
    auto iterRange = forOp.getRegionIterArgs();
    iterArgs.assign(iterRange.begin(), iterRange.end());

    if (first) {
      // Store the results of the outermost loop that will be used to replace
      // the results of the parallel loop when it is fully rewritten.
      loopResults.assign(forOp.result_begin(), forOp.result_end());
      first = false;
    } else if (!forOp.getResults().empty()) {
      // A loop is constructed with an empty "yield" terminator if there are
      // no results.
      rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
      rewriter.create<scf::YieldOp>(loc, forOp.getResults());
    }

    forLoops.push_back(forOp); 
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  // First, merge reduction blocks into the main region.
  SmallVector<Value, 4> yieldOperands;
  yieldOperands.reserve(parallelOp.getNumResults());
  for (auto &op : *parallelOp.getBody()) {
    auto reduce = dyn_cast<ReduceOp>(op);
    if (!reduce)
      continue;

    Block &reduceBlock = reduce.reductionOperator().front();
    Value arg = iterArgs[yieldOperands.size()];
    yieldOperands.push_back(reduceBlock.getTerminator()->getOperand(0));
    rewriter.eraseOp(reduceBlock.getTerminator());
    rewriter.mergeBlockBefore(&reduceBlock, &op, {arg, reduce.operand()});
    rewriter.eraseOp(reduce);
  }

  // Then merge the loop body without the terminator.
  rewriter.eraseOp(parallelOp.getBody()->getTerminator());
  Block *newBody = rewriter.getInsertionBlock();
  if (newBody->empty())
    rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
  else
    rewriter.mergeBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                              ivs);

  // Finally, create the terminator if required (for loops with no results, it
  // has been already created in loop construction).
  if (!yieldOperands.empty()) {
    rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
    rewriter.create<scf::YieldOp>(loc, yieldOperands);
  }

  rewriter.replaceOp(parallelOp, loopResults);

  // Now we have a set of nested for loops that we know can be executed in
  // parallel.  
  for(auto i = forLoops.begin(); i != forLoops.end(); i++){
    // We handle the special case of the last element of the loop for inserting
    // Tapir instructions: 
    bool innerMost = *i == *forLoops.rbegin(); 
    ForOp &forOp = *i; 
    Location loc = forOp.getLoc();
    
    rewriter.setInsertionPoint(forOp); 

    // Start by splitting the block containing the 'scf.for' into two parts.
    // The part before will get the init code, the part after will be the end
    // point.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPosition = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPosition);

    // Use the first block of the loop body as the condition block since it is the
    // block that has the induction variable and loop-carried values as arguments.
    // Split out all operations from the first block into a new block. Move all
    // body blocks from the loop body region to the region containing the loop.
    auto *conditionBlock = &forOp.region().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &forOp.region().back();
    rewriter.inlineRegionBefore(forOp.region(), endBlock);
    auto iv = conditionBlock->getArgument(0);

    // Append the induction variable stepping logic to the last body block and
    // branch back to the condition block. Loop-carried values are taken from
    // operands of the loop terminator.
    Operation *terminator = lastBodyBlock->getTerminator();
    if(innerMost){
      auto *detachedBlock = rewriter.splitBlock(firstBodyBlock, firstBodyBlock->begin()); 
      auto *reattachBlock = rewriter.splitBlock(lastBodyBlock, lastBodyBlock->end()); 
      rewriter.setInsertionPointToEnd(firstBodyBlock);
      rewriter.create<LLVM::Tapir_detach>(loc, sr, ArrayRef<Value>(), ArrayRef<Value>(), detachedBlock, reattachBlock); 
      rewriter.setInsertionPointToEnd(detachedBlock);
      rewriter.create<LLVM::Tapir_reattach>(loc, sr, ArrayRef<Value>(), reattachBlock); 
      rewriter.setInsertionPointToStart(reattachBlock); 
    } else {
      rewriter.setInsertionPointToEnd(lastBodyBlock); 
    } 
      
    auto step = forOp.step();
    auto stepped = rewriter.create<AddIOp>(loc, iv, step).getResult();
    if (!stepped)
      return failure();

    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    rewriter.create<BranchOp>(loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    Value lowerBound = forOp.lowerBound();
    Value upperBound = forOp.upperBound();
    if (!lowerBound || !upperBound)
      return failure();

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    SmallVector<Value, 8> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = forOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    rewriter.create<BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison =
        rewriter.create<CmpIOp>(loc, CmpIPredicate::slt, iv, upperBound);

    rewriter.create<CondBranchOp>(loc, comparison, firstBodyBlock,
                                  ArrayRef<Value>(), endBlock, ArrayRef<Value>());
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());

    auto syncBlock = rewriter.splitBlock(endBlock, endBlock->begin()); 
    rewriter.setInsertionPointToEnd(endBlock);
    rewriter.create<LLVM::Tapir_sync>(loc, sr, ArrayRef<Value>(), syncBlock); 
  }

  return success();
}

void mlir::populateParallelToTapirConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ParallelLowering>(ctx);
}

void SCFToTapirPass::runOnOperation() {
  OwningRewritePatternList patterns;
  populateParallelToTapirConversionPatterns(patterns, &getContext());
  // Configure conversion to lower out scf.for, scf.if, scf.parallel and
  // scf.while. Anything else is fine.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ParallelOp>();
  //target.addLegalDialect<LLVM::LLVMDialect>();
  //target.addLegalDialect<LLVM::LLVMTapirDialect>();

  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToTapirPass() {
  return std::make_unique<SCFToTapirPass>();
}
