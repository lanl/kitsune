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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTapirDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOTAPIR
#include "mlir/Conversion/Passes.h.inc"
}

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToTapirPass 
    : public impl::SCFToTapirBase<SCFToTapirPass> {
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
  auto sr = rewriter.create<LLVM::Tapir_syncregion_start>(loc, LLVM::LLVMTokenType::get(ctx)); 
  // For a parallel loop, we essentially need to create an n-dimensional loop
  // nest. We do this by translating to scf.for ops and have those lowered in
  // a further rewrite. 
  
  // For reductions, we create an alloca for each and insert the appropriate
  // loads and stores
  SmallVector<Value, 4> reductionAllocas; 
  Value one = rewriter.create<LLVM::ConstantOp>(
	loc, rewriter.getIntegerType(64), rewriter.getI64IntegerAttr(1));
  //Value index = rewriter.create<ConstantIndexOp>(loc, 0); 
  for(auto initVal : parallelOp.getInitVals()){
    auto ptrTp = LLVM::LLVMPointerType::get(initVal.getType()); 
    Value ra = rewriter.create<LLVM::AllocaOp>(loc, ptrTp, one, 0);
    //auto ra = rewriter.create<memref::AllocaOp>(loc, ptrTp);
    //Value index = rewriter.create<ConstantIndexOp>(loc, 0); 
    rewriter.create<LLVM::StoreOp>(loc, initVal, ra);  
    reductionAllocas.push_back(ra); 
  }
  SmallVector<Value, 4> ivs;
  ivs.reserve(parallelOp.getNumLoops());
  for (auto loop_operands :
       llvm::zip(parallelOp.getInductionVars(), parallelOp.getLowerBound(),
                 parallelOp.getUpperBound(), parallelOp.getStep())) {
    Value iv, lower, upper, step;
    std::tie(iv, lower, upper, step) = loop_operands;
    ForOp forOp = rewriter.create<ForOp>(loc, lower, upper, step);
    ivs.push_back(forOp.getInductionVar());

    forLoops.push_back(forOp); 
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  auto ib = rewriter.getInsertionBlock(); 
  auto ip = rewriter.getInsertionPoint(); 
  // First, merge reduction blocks into the main region.
  int rai=0; 
  for (auto &op : *parallelOp.getBody()) {
    auto reduce = dyn_cast<ReduceOp>(op);
    if (!reduce)
      continue;

    rewriter.setInsertionPointAfter(&op); 

    Block &reduceBlock = reduce.getReductionOperator().front();
    auto arg = rewriter.create<LLVM::LoadOp>(loc, reductionAllocas[rai]); 

    // Outlined the reduction op to a function
    rewriter.setInsertionPoint(op.getParentOfType<func::FuncOp>()); 
    auto type = reduce.getOperand().getType(); 
    FunctionType funtype = FunctionType::get(ctx, { type, type}, type ); 
    auto outlinedFunc = rewriter.create<func::FuncOp>(loc, "reduction_" + std::to_string(rai), funtype);
    outlinedFunc->setAttr("passthrough", ArrayAttr::get(ctx,
      {StringAttr::get(ctx, "reduction"),
       StringAttr::get(ctx, "noinline")} 
      )); 
    
    // copy instructions to outlined func
    rewriter.setInsertionPointToStart(outlinedFunc.addEntryBlock()); 
    IRMapping bvm; 
    for(auto it : llvm::zip(reduceBlock.getArguments(), outlinedFunc.getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));
    for(Operation &op : reduceBlock.without_terminator())
      rewriter.clone(op, bvm); 
    
    Operation *term = reduceBlock.getTerminator(); 
    rewriter.create<LLVM::ReturnOp>(loc, bvm.lookup(term->getOperand(0))); 
    
    // insert call to outlined func and store result in alloca
    rewriter.setInsertionPointAfter(arg); 
    ValueRange values( {arg, reduce.getOperand() }); 
    auto rv = rewriter.create<func::CallOp>(loc, outlinedFunc, values); 
    rewriter.create<LLVM::StoreOp>(loc, rv->getResult(0), reductionAllocas[rai++]); 

    rewriter.eraseOp(reduce);

  }

  rewriter.setInsertionPoint(ib, ip); 
  // Then merge the loop body without the terminator.
  rewriter.eraseOp(parallelOp.getBody()->getTerminator());
  Block *newBody = rewriter.getInsertionBlock();
  if (newBody->empty())
    rewriter.mergeBlocks(parallelOp.getBody(), newBody, ivs);
  else
    rewriter.mergeBlockBefore(parallelOp.getBody(), newBody->getTerminator(),
                              ivs);

  // Now we have a set of nested for loops that we know can be executed in
  // parallel.  
  for(auto i = forLoops.begin(); i != forLoops.end(); i++){
    // We handle the special case of the last element of the loop for inserting
    // Tapir instructions: 
    bool innerMost = *i == *forLoops.rbegin(); 
    bool outerMost = *i == *forLoops.begin(); 
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
    auto *conditionBlock = &forOp.getRegion().front();
    auto *firstBodyBlock =
        rewriter.splitBlock(conditionBlock, conditionBlock->begin());
    auto *lastBodyBlock = &forOp.getRegion().back();
    rewriter.inlineRegionBefore(forOp.getRegion(), endBlock);
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
      
    auto step = forOp.getStep();
    auto stepped = rewriter.create<arith::AddIOp>(loc, iv, step).getResult();
    if (!stepped)
      return failure();

    SmallVector<Value, 8> loopCarried;
    loopCarried.push_back(stepped);
    loopCarried.append(terminator->operand_begin(), terminator->operand_end());
    rewriter.create<cf::BranchOp>(loc, conditionBlock, loopCarried);
    rewriter.eraseOp(terminator);

    // Compute loop bounds before branching to the condition.
    rewriter.setInsertionPointToEnd(initBlock);
    Value lowerBound = forOp.getLowerBound();
    Value upperBound = forOp.getUpperBound();
    if (!lowerBound || !upperBound)
      return failure();

    // The initial values of loop-carried values is obtained from the operands
    // of the loop operation.
    SmallVector<Value, 8> destOperands;
    destOperands.push_back(lowerBound);
    auto iterOperands = forOp.getIterOperands();
    destOperands.append(iterOperands.begin(), iterOperands.end());
    rewriter.create<cf::BranchOp>(loc, conditionBlock, destOperands);

    // With the body block done, we can fill in the condition block.
    rewriter.setInsertionPointToEnd(conditionBlock);
    auto comparison =
        rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iv, upperBound);

    auto cb = rewriter.create<cf::CondBranchOp>(loc, comparison, firstBodyBlock,
                                  ArrayRef<Value>(), endBlock, ArrayRef<Value>());
    cb->setAttr(mlir::LLVM::LLVMDialect::getLoopAttrName(),  DictionaryAttr::get(ctx, {
	{StringAttr::get(ctx, "disable_unroll"), BoolAttr::get(ctx, true)}, 
	{StringAttr::get(ctx, "tapir.loop.spawn.strategy"), 
		IntegerAttr::get(IntegerType::get(ctx, 32), 1)}})); 
    // The result of the loop operation is the values of the condition block
    // arguments except the induction variable on the last iteration.
    rewriter.replaceOp(forOp, conditionBlock->getArguments().drop_front());

    
    // We only need to sync after the outermost loop
    if(outerMost){
      auto syncBlock = rewriter.splitBlock(endBlock, endBlock->begin()); 
      rewriter.setInsertionPointToEnd(endBlock);
      rewriter.create<LLVM::Tapir_sync>(loc, sr, ArrayRef<Value>(), syncBlock); 

      // Insert loads after the sync
      rewriter.setInsertionPointToStart(syncBlock); 
      SmallVector<Value,4> loopResults(reductionAllocas); 
      for(size_t i = 0; i < reductionAllocas.size(); i++){
        loopResults[i] = rewriter.create<LLVM::LoadOp>(loc, reductionAllocas[i]); 
      }
      rewriter.replaceOp(parallelOp, loopResults);
    }
  }

  return success();
}

void mlir::populateParallelToTapirConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ParallelLowering>(patterns.getContext());
}

void SCFToTapirPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  populateParallelToTapirConversionPatterns(patterns);

  // Configure conversion to lower out scf.parallel
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ParallelOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::createLowerToTapirPass() {
  return std::make_unique<SCFToTapirPass>();
}
