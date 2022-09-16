/**
 ***************************************************************************
 * TODO: Need to update LANL/Triad Copyright notice...
 *
 * Copyright (c) 2017, Los Alamos National Security, LLC.
 * All rights reserved.
 *
 *  Copyright 2010. Los Alamos National Security, LLC. This software was
 *  produced under U.S. Government contract DE-AC52-06NA25396 for Los
 *  Alamos National Laboratory (LANL), which is operated by Los Alamos
 *  National Security, LLC for the U.S. Department of Energy. The
 *  U.S. Government has rights to use, reproduce, and distribute this
 *  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 *  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 *  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 *  derivative works, such modified software should be clearly marked,
 *  so as not to confuse it with the version available from LANL.
 *
 *  Additionally, redistribution and use in source and binary forms,
 *  with or without modification, are permitted provided that the
 *  following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of Los Alamos National Security, LLC, Los
 *      Alamos National Laboratory, LANL, the U.S. Government, nor the
 *      names of its contributors may be used to endorse or promote
 *      products derived from this software without specific prior
 *      written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 *  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 *  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 *  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 *  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 *  SUCH DAMAGE.
 *
 ***************************************************************************/
#include "clang/AST/Attr.h"
#include "clang/Basic/Attributes.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "CodeGenFunction.h"
#include "CGCleanup.h"
#include "clang/AST/StmtKitsune.h"
#include "llvm/IR/ValueMap.h"


using namespace clang;
using namespace CodeGen;


LoopAttributes::LSStrategy
CodeGenFunction::GetTapirStrategyAttr(ArrayRef<const Attr *> Attrs) {

  LoopAttributes::LSStrategy Strategy = LoopAttributes::SEQ;

  auto curAttr = Attrs.begin();

  while(curAttr != Attrs.end()) {

    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirStrategy) {
      const auto *SAttr = cast<const TapirStrategyAttr>(*curAttr);

      switch(SAttr->getTapirStrategyType()) {
        case TapirStrategyAttr::SEQ:
          Strategy = LoopAttributes::SEQ;
          break;
        case TapirStrategyAttr::DAC:
          Strategy = LoopAttributes::DAC;
          break;
        case TapirStrategyAttr::GPU:
          Strategy = LoopAttributes::GPU;
          break;
        default:
          llvm_unreachable("all strategies should be handled before this!");
          break;
      }
    }
  }
  return Strategy;
}

LoopAttributes::LTarget
CodeGenFunction::GetTapirTargetAttr(ArrayRef<const Attr *> Attrs) {

  auto curAttr = Attrs.begin();

  // TODO: Need to make sure the default matches build parameters!
  LoopAttributes::LTarget Target = LoopAttributes::CilkRT;

  while(curAttr != Attrs.end()) {

    const attr::Kind AttrKind = (*curAttr)->getKind();

    if (AttrKind == attr::TapirTarget) {
      const auto *TAttr = cast<const TapirTargetAttr>(*curAttr);

      switch(TAttr->getTapirTargetAttrType()) {
        case TapirTargetAttr::CheetahRT:
          Target = LoopAttributes::CheetahRT;
          break;
        case TapirTargetAttr::CilkRT:
          Target = LoopAttributes::CilkRT;
          break;
        case TapirTargetAttr::CudaRT:
          Target = LoopAttributes::CudaRT;
          break;
        case TapirTargetAttr::HipRT:
          Target = LoopAttributes::HipRT;
          break;
        case TapirTargetAttr::OmpRT:
          Target = LoopAttributes::OmpRT;
          break;
        case TapirTargetAttr::QthreadsRT:
         Target = LoopAttributes::QthreadsRT;
         break;
        case TapirTargetAttr::RealmRT:
          Target = LoopAttributes::RealmRT;
          break;
        case TapirTargetAttr::RocmRT:
          Target = LoopAttributes::RocmRT;
          break;
        case TapirTargetAttr::SequentialRT:
          Target = LoopAttributes::SequentialRT;
          break;
        case TapirTargetAttr::ZeroRT:
          Target = LoopAttributes::ZeroRT;
          break;
        default:
          llvm_unreachable("All target attributes should be handled here!");
          break;
      }
    }
    curAttr++;
  }
  return Target;
}


// Stolen from CodeGenFunction.cpp
static void EmitIfUsed(CodeGenFunction &CGF, llvm::BasicBlock *BB) {
  if (!BB) return;
  if (!BB->use_empty())
    return CGF.CurFn->getBasicBlockList().push_back(BB);
  delete BB;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      CGM.getIntrinsic(llvm::Intrinsic::syncregion_start), SV, AllocaInsertPt);
  return SRStart;
}

/// EmitSyncStmt - Emit a sync node.
void CodeGenFunction::EmitSyncStmt(const SyncStmt &S) {
  llvm::BasicBlock *ContinueBlock = createBasicBlock("sync.continue");

  // If this code is reachable then emit a stop point (if generating
  // debug info). We have to do this ourselves because we are on the
  // "simple" statement path.
  if (HaveInsertPoint())
    EmitStopPoint(&S);

  Builder.CreateSync(ContinueBlock,
    getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  // PushDetachScope();
  SyncRegion* SR = getOrCreateLabeledSyncRegion(S.getSyncVar());
  //StartLabeledDetach(SR);

  llvm::BasicBlock* DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock* ContinueBlock = createBasicBlock("det.cont");

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "",
                                             DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock,
                           SR->getSyncRegionStart());


  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock,
                             SR->getSyncRegionStart());

  llvm::Instruction* ptr = AllocaInsertPt;
  AllocaInsertPt = OldAllocaInsertPt;
  ptr->eraseFromParent();

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::SetAllocaInsertPoint(llvm::Value* v, llvm::BasicBlock* bb){
  AllocaInsertPt = new llvm::BitCastInst(v, Int32Ty, "", bb);
}

// Emit a load of the induction variable
// It has a side effect of erasing the mapping in the
// LocalDeclMap but keeping track of the original mapping
// as well as the new RValue after the load. This is all
// a precursor to capturing the IV by value in the body emission.
void CodeGenFunction::EmitIVLoad(const VarDecl* LoopVar,
                                DeclMapByValueTy& IVDeclMap) {

  // The address corresponding to the IV
  Address IVAddress = LocalDeclMap.find(LoopVar)->second;

  // Remove the IV mapping from the LocalDeclMap
  LocalDeclMap.erase(LoopVar);

  // Get the type
  QualType type = LoopVar->getType();

  // Create the vector of values

  llvm::SmallVector<llvm::Value *, 4> ValueVec;

  // Emit all the shallow copy loads and update
  switch (getEvaluationKind(type)) {
    case TEK_Scalar: {
      LValue IVLV = MakeAddrLValue(IVAddress, type);
      RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc());
      ValueVec.push_back(IVRV.getScalarVal());
      break;
    }
    case TEK_Complex: {
      ComplexPairTy Val =
        EmitLoadOfComplex(MakeAddrLValue(IVAddress, type),
                          LoopVar->getBeginLoc());
      ValueVec.push_back(Val.first);
      ValueVec.push_back(Val.second);
      break;
    }
    case TEK_Aggregate: {
      if (const llvm::StructType *STy = dyn_cast<llvm::StructType>(IVAddress.getElementType())) {
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          Address EltPtr = Builder.CreateStructGEP(IVAddress, i);
          llvm::Value *Elt = Builder.CreateLoad(EltPtr);
          ValueVec.push_back(Elt);
        }
      } else {
        LValue IVLV = MakeAddrLValue(IVAddress, type);
        RValue IVRV = EmitLoadOfLValue(IVLV, LoopVar->getBeginLoc());
        ValueVec.push_back(IVRV.getScalarVal());
      }
      break;
    }
  }

  // Capture the mapping from LoopVar to the old address and new vector of Value*'s
  IVDeclMap.insert({LoopVar, {IVAddress, ValueVec}});
}

// Emit a thread safe copy of the induction variable and set it's value
// to the current value of the induction variable

void CodeGenFunction::EmitThreadSafeIV(const VarDecl* IV, const llvm::SmallVector<llvm::Value*,4>& Values){

  // emit the thread safe induction variable and cleanups
  AutoVarEmission LVEmission = EmitAutoVarAlloca(*IV);
  EmitAutoVarCleanups(LVEmission);
  QualType type = IV->getType();

  // get the address of the emission
  Address Loc = LVEmission.getObjectAddress(*this);

  // turn the address into an LValue
  LValue LV = MakeAddrLValue(Loc, type);

  // Make sure the LValue isn't garbage collected
  LV.setNonGC(true);

  switch (getEvaluationKind(type)) {
    case TEK_Scalar: {
      EmitStoreOfScalar(Values[0], LV, true);
      break;
    }
    case TEK_Complex: {
      ComplexPairTy Val = {Values[0], Values[1]};
      EmitStoreOfComplex(Val, LV, true);
      break;
    }
    case TEK_Aggregate: {
      if (const llvm::StructType *STy =dyn_cast<llvm::StructType>(Loc.getElementType())) {
        for (unsigned i = 0, e = STy->getNumElements(); i != e; ++i) {
          Address EltPtr = Builder.CreateStructGEP(Loc, i);
          llvm::Value *Elt = Values[i];
          Builder.CreateStore(Elt, EltPtr);
        }
      } else {
        EmitStoreOfScalar(Values[0], LV, /*isInit*/ true);
      }
      break;
    }
  }

}

// Restore the original mapping between the Vardecl and its address
void CodeGenFunction::RestoreDeclMap(const VarDecl* IV, const Address IVAddress){

  // remove the mapping to the thread safe induction variable
  LocalDeclMap.erase(IV);

  // restore the original mapping
  LocalDeclMap.insert({IV, IVAddress});
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                  ArrayRef<const Attr *> ForallAttr) {

  LoopAttributes::LTarget TT = GetTapirTargetAttr(ForallAttr);
  if (TT == LoopAttributes::CudaRT) {
    fprintf(stderr, "Found a cuda attributed forall statement.\n");
  }

  /////////////////////////////////////////////////////////////////////////////
  // Code Modifications necessary for implementing parallel loops not required
  // by serial loops.

  // New basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock* Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest Sync = getJumpDestInCurrentScope("forall.sync");

  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV vector}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the initialization before the loop.
  EmitStmt(S.getInit());

  // In a parallel loop there will always be a condition block
  // so there is no need to test
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *CondBlock = Condition.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(),
                 ForallAttr, SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // In a parallel loop, there will always be an increment block
  JumpDest Increment = getJumpDestInCurrentScope("forall.inc");

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  // Create a cleanup scope for the condition variable cleanups.
  // We don't need this unless we allow condition scope variables
  LexicalScope ConditionScope(*this, S.getSourceRange());

  // If the for statement has a condition scope, emit the local variable
  // declaration.
  // Presently, we don't support condition variables, but we should :-)
  if (S.getConditionVariable()) {
    EmitDecl(*S.getConditionVariable());
  }

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(Sync);
  }

  /////////////////////////////////////////////////////////////////////////////
  // The following block of code emits the detach block for parallel execution
  // along with its Tapir terminator. This is where we capture the induction
  // variable by value and store it on the stack of the calling thread.

  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getInit());

  // Create threadsafe induction variables before the detach and put them in IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.getBlock(), SRStart);

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(ForBody);

  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

    ///////////////////////////////////////////////////////////////////////////
    // In this block of code, we change the alloca insert point so that the
    // alloca's happen after the detach and within the body block. This makes
    // sure each thread has its own local copy of the induction variable. We
    // also need to store the thread safe value from the calling thread into
    // this local copy. In EmitThreadSafeIV, we use AutoVarAlloca so any codegen
    // in the body automatically and correctly mapped to the local thread
    // safe copy of the induction variable.

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap)
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    // End of parallel modification code block
    ///////////////////////////////////////////////////////////////////////////

    EmitStmt(S.getBody());
  }

  /////////////////////////////////////////////////////////////////////////////
  // In this block of code, we need to unwind the codegen of the induction
  // variable from the current local thread safe copy back to the original
  // induction variable. We also need to emit the reattach block and reset the
  // alloca insertion point.

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap)
    RestoreDeclMap(ivp.first, ivp.second.first);

  // emit the reattach block
  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment.getBlock(), SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt;

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  // Emit the increment.
  EmitBlock(Increment.getBlock());
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  ConditionScope.ForceCleanup();

  EmitStopPoint(&S);
  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the Sync block and terminator
  EmitBlock(Sync.getBlock());
  Builder.CreateSync(LoopExit.getBlock(), SRStart);

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void
CodeGenFunction::EmitCXXForallRangeStmt(const CXXForallRangeStmt &S,
                                        ArrayRef<const Attr *> ForallAttr) {

  /////////////////////////////////////////////////////////////////////////////
  // Code modifications necessary for implementing parallel loops not required
  // by serial loops.

  // new basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock* Detach = createBasicBlock("forall.detach");
  JumpDest Reattach = getJumpDestInCurrentScope("forall.reattach");
  JumpDest LoopExit = getJumpDestInCurrentScope("forall.sync");

  // Declarations for capturing the IV vardecl to old and new llvm Values as
  // well as the alloca insertion point which we need to change and change back
  DeclMapByValueTy IVDeclMap; // map from Vardecl to {IV, thread safe IV}
  llvm::AssertingVH<llvm::Instruction> OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);

  // emit the sync region
  PushSyncRegion();
  llvm::Instruction *SRStart = EmitSyncRegionStart();
  CurSyncRegion->setSyncRegionStart(SRStart);
  LoopStack.setSpawnStrategy(LoopAttributes::DAC);

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  llvm::BasicBlock *End = createBasicBlock("forall.end");

  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the first pieces before the loop.
  if (S.getInit())
    EmitStmt(S.getInit());
  EmitStmt(S.getRangeStmt());
  EmitStmt(S.getBeginStmt());
  EmitStmt(S.getEndStmt());
  EmitStmt(S.getIndexStmt());
  EmitStmt(S.getIndexEndStmt());

  // In a parallel loop there will always be a condition block
  // so there is no need to test
  llvm::BasicBlock *CondBlock = createBasicBlock("forall.cond");
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(),
                 ForallAttr, SourceLocToDebugLoc(R.getBegin()),
                 SourceLocToDebugLoc(R.getEnd()));

  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // The loop body, consisting of the specified body and the loop variable.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // The body is executed if the expression, contextually converted
  // to bool, is true.
  llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  llvm::MDNode *Weights = createProfileWeightsForLoop(
      S.getCond(), getProfileCount(S.getBody()));
  Builder.CreateCondBr(BoolCondVal, Detach, ExitBlock, Weights);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  /////////////////////////////////////////////////////////////////////////////
  // The following block of code emits the detach block for parallel execution
  // along with its Tapir terminator. This is where we capture the induction
  // variable by value and store it on the stack of the calling thread.

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getIndexStmt());

  // Create threadsafe induction variables before the detach and put them in IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // Create a block for the increment. In case of a 'continue', we jump there.
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, S.getSourceRange());

    ///////////////////////////////////////////////////////////////////////////
    // In this block of code, we change the alloca insert point so that the
    // alloca's happen after the detach and within the body block. This makes
    // sure each thread has its own local copy of the induction variable. We
    // also need to store the thread safe value from the calling thread into
    // this local copy. In EmitThreadSafeIV, we use AutoVarAlloca so any codegen
    // in the body automatically and correctly mapped to the local thread
    // safe copy of the induction variable.

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap)
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    // End of parallel modification code block
    ///////////////////////////////////////////////////////////////////////////

    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  /////////////////////////////////////////////////////////////////////////////
  // In this block of code, we need to unwind the codegen of the induction
  // variable from the current local thread safe copy back to the original
  // induction variable. We also need to emit the reattach block and reset the
  // alloca insertion point.

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap)
    RestoreDeclMap(ivp.first, ivp.second.first);

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt;

  // End of parallel modification code block
  /////////////////////////////////////////////////////////////////////////////

  EmitStopPoint(&S);
  // If there is an increment, emit it next.
  EmitBlock(Increment);
  EmitStmt(S.getInc());

  BreakContinueStack.pop_back();

  EmitBranch(CondBlock);

  ForScope.ForceCleanup();

  LoopStack.pop();

  // Emit the Sync block and terminator
  EmitBlock(LoopExit.getBlock());
  Builder.CreateSync(End, SRStart);

  EmitBlock(End, true);
}

