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
#include "CGCleanup.h"
#include "CodeGenFunction.h"
#include "kitsune/Frontend/KitsuneOptions.h"
#include "clang/AST/Attr.h"
#include "clang/AST/StmtKitsune.h"
#include "clang/Basic/AttrKinds.h"
#include "clang/Basic/Attributes.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/IR/FixedPointBuilder.h"

using namespace clang;
using namespace CodeGen;

// KITSUNE FIXME: This seems to be unused. Is this actually the case? Also, is
// there is a bug in the attribute handling loop since curAttr is never
// incremented? Or is there some mysterious C++ black magic at work again?
llvm::TapirSpawnStrategy
CodeGenFunction::GetTapirStrategyAttr(ArrayRef<const Attr *> Attrs) {
  auto curAttr = Attrs.begin();
  while (curAttr != Attrs.end()) {
    const attr::Kind AttrKind = (*curAttr)->getKind();
    if (AttrKind == attr::TapirStrategy) {
      const auto *SAttr = cast<TapirStrategyAttr>(*curAttr);
      switch (SAttr->getTapirStrategyType()) {
      case TapirStrategyAttr::SEQ:
        return llvm::TapirSpawnStrategy::Sequential;
      case TapirStrategyAttr::DAC:
        return llvm::TapirSpawnStrategy::DivideAndConquer;
      case TapirStrategyAttr::GPU:
        return llvm::TapirSpawnStrategy::GPU;
      }
      llvm_unreachable("GetTapirStrategyAttr: TapirStrategyAttr not handled");
    }
  }
  return llvm::TapirSpawnStrategy::Sequential;
}

// If a tapir target attribute exists, it will override the tapir target
// specified on the command line - if any. If a tapir target attribute does not
// exist and one was specified on the command line, that will be returned.
std::optional<llvm::TTID>
CodeGenFunction::GetTapirTargetAttr(ArrayRef<const Attr *> Attrs) {
  // FIXME KITSUNE: This will check for the first occurrence of the tapir target
  // attribute and break immediately if it finds it. Is this what we actually
  // want?
  for (const Attr *curAttr : Attrs) {
    if (curAttr->getKind() == attr::TapirTarget) {
      switch (cast<const TapirTargetAttr>(curAttr)->getTapirTargetAttrType()) {
      case TapirTargetAttr::Nolo:
        return llvm::TTID::Nolo;
      case TapirTargetAttr::Serial:
        return llvm::TTID::Serial;
      case TapirTargetAttr::Cuda:
        return llvm::TTID::Cuda;
      case TapirTargetAttr::Hip:
        return llvm::TTID::Hip;
      case TapirTargetAttr::GPU:
        return llvm::TTID::GPU; 
      case TapirTargetAttr::OpenCilk:
        return llvm::TTID::OpenCilk;
      case TapirTargetAttr::OpenMP:
        return llvm::TTID::OpenMP;
      case TapirTargetAttr::Qthreads:
        return llvm::TTID::Qthreads;
      case TapirTargetAttr::Realm:
        return llvm::TTID::Realm;
      }
      llvm_unreachable("GetTapirTargetAttr: TTID not handled");
    }
  }
  return CGM.getKitsuneOpts().getTTID();
}

unsigned CodeGenFunction::GetKitsuneLaunchAttr(ArrayRef<const Attr *> Attrs) {
  for (const auto *curAttr : Attrs) {
    if (curAttr->getKind() == attr::KitsuneLaunch) {
      unsigned ThreadsPerBlock =
          cast<const KitsuneLaunchAttr>(curAttr)->getThreadsPerBlock();
      return ThreadsPerBlock;
    }
  }
  // missing attribute -- zero threads per block using runtime settings.
  return 0;
}

llvm::Instruction *CodeGenFunction::EmitLabeledSyncRegionStart(StringRef SV) {
  // Start the sync region.  To ensure the syncregion.start call dominates all
  // uses of the generated token, we insert this call at the alloca insertion
  // point.
  llvm::Function *Func = CGM.getIntrinsic(llvm::Intrinsic::syncregion_start);
  llvm::Instruction *SRStart = llvm::CallInst::Create(
      Func->getFunctionType(), Func, SV, AllocaInsertPt->getIterator());
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

  Builder.CreateSync(
      ContinueBlock,
      getOrCreateLabeledSyncRegion(S.getSyncVar())->getSyncRegionStart());
  EmitBlock(ContinueBlock);
}

void CodeGenFunction::EmitSpawnStmt(const SpawnStmt &S) {
  // Set up to perform a detach.
  SyncRegion *SR = getOrCreateLabeledSyncRegion(S.getSyncVar());

  llvm::BasicBlock *DetachedBlock = createBasicBlock("det.achd");
  llvm::BasicBlock *ContinueBlock = createBasicBlock("det.cont");

  auto OldAllocaInsertPt = AllocaInsertPt;
  llvm::Value *Undef = llvm::UndefValue::get(Int32Ty);
  AllocaInsertPt = new llvm::BitCastInst(Undef, Int32Ty, "", DetachedBlock);

  Builder.CreateDetach(DetachedBlock, ContinueBlock, SR->getSyncRegionStart());

  EmitBlock(DetachedBlock);
  EmitStmt(S.getSpawnedStmt());

  Builder.CreateReattach(ContinueBlock, SR->getSyncRegionStart());

  llvm::Instruction *ptr = AllocaInsertPt;
  AllocaInsertPt = OldAllocaInsertPt;
  ptr->eraseFromParent();

  EmitBlock(ContinueBlock);
}

void CodeGenFunction::SetAllocaInsertPoint(llvm::Value *v,
                                           llvm::BasicBlock *bb) {
  AllocaInsertPt = new llvm::BitCastInst(v, Int32Ty, "", bb);
}

// Emit a load of the induction variable
// It has a side effect of erasing the mapping in the
// LocalDeclMap but keeping track of the original mapping
// as well as the new RValue after the load. This is all
// a precursor to capturing the IV by value in the body emission.
void CodeGenFunction::EmitIVLoad(const VarDecl *LoopVar,
                                 DeclMapByValueTy &IVDeclMap) {
  // The address corresponding to the IV
  Address IVAddress = LocalDeclMap.find(LoopVar)->second;
  LocalDeclMap.erase(LoopVar);
  QualType type = LoopVar->getType();
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
    ComplexPairTy Val = EmitLoadOfComplex(MakeAddrLValue(IVAddress, type),
                                          LoopVar->getBeginLoc());
    ValueVec.push_back(Val.first);
    ValueVec.push_back(Val.second);
    break;
  }
  case TEK_Aggregate: {
    if (const llvm::StructType *STy =
            dyn_cast<llvm::StructType>(IVAddress.getElementType())) {
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

  // Capture the mapping from LoopVar to the old address and new vector of
  // Value*'s
  IVDeclMap.insert({LoopVar, {IVAddress, ValueVec}});
}

// Emit a thread safe copy of the induction variable and set it's value
// to the current value of the induction variable
void CodeGenFunction::EmitThreadSafeIV(
    const VarDecl *IV, const llvm::SmallVector<llvm::Value *, 4> &Values) {
  // emit the thread safe induction variable and cleanups
  AutoVarEmission LVEmission = EmitAutoVarAlloca(*IV);
  EmitAutoVarCleanups(LVEmission);
  QualType type = IV->getType();
  Address Loc = LVEmission.getObjectAddress(*this);
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
    if (const llvm::StructType *STy =
            dyn_cast<llvm::StructType>(Loc.getElementType())) {
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

// Restore the original mapping between the thread safe induction variable and
// its address, then restore the original mapping.
void CodeGenFunction::RestoreDeclMap(const VarDecl *IV,
                                     const Address IVAddress) {
  LocalDeclMap.erase(IV);
  LocalDeclMap.insert({IV, IVAddress});
}

void CodeGenFunction::EmitForallStmt(const ForallStmt &S,
                                     ArrayRef<const Attr *> ForallAttr) {
  assert(CGM.getKitsuneOpts().getTTID().has_value() &&
         "TTID not set in Kitsune options");
  llvm::TTID TT = *CGM.getKitsuneOpts().getTTID();
  if (std::optional<llvm::TTID> AttrTT = GetTapirTargetAttr(ForallAttr))
    TT = *AttrTT;
  if (TT != llvm::TTID::Nolo)
    LoopStack.setLoopTarget(TT);

  // New basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
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
  LoopStack.setSpawnStrategy(llvm::TapirSpawnStrategy::DivideAndConquer);
  // See if we have any launch attributes to handle before we start loop body.
  if (TT == llvm::TTID::Cuda || TT == llvm::TTID::Hip) {
    unsigned ThreadsPerBlock = GetKitsuneLaunchAttr(ForallAttr);
    if (ThreadsPerBlock > 0)
      LoopStack.setLoopThreadsPerBlock(ThreadsPerBlock);
  }

  JumpDest LoopExit = getJumpDestInCurrentScope("forall.end");
  LexicalScope ForScope(*this, S.getSourceRange());

  // Evaluate the initialization before the loop.
  EmitStmt(S.getInit());

  // TODO: explain more
  // We assume that the boolean is a binary operator and pre-compute RHS
  auto* BO = dyn_cast<BinaryOperator>(S.getCond()); 
  llvm::Value *RHS = EmitScalarExpr(BO->getRHS()); 


  // In a parallel loop there will always be a condition block
  // so there is no need to test
  JumpDest Condition = getJumpDestInCurrentScope("forall.cond");
  llvm::BasicBlock *CondBlock = Condition.getBlock();
  EmitBlock(CondBlock);

  const SourceRange &R = S.getSourceRange();
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), ForallAttr,
                 SourceLocToDebugLoc(R.getBegin()),
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
  if (S.getConditionVariable())
    EmitDecl(*S.getConditionVariable());

  llvm::BasicBlock *ExitBlock = LoopExit.getBlock();
  // If there are any cleanups between here and the loop-exit scope,
  // create a block to stage a loop exit along.
  if (ForScope.requiresCleanups())
    ExitBlock = createBasicBlock("forall.cond.cleanup");

  // As long as the condition is true, iterate the loop.
  llvm::BasicBlock *ForBody = createBasicBlock("forall.body");

  // C99 6.8.5p2/p4: The first substatement is executed if the expression
  // compares unequal to 0.  The condition must be a scalar type.
  
  //llvm::Value *BoolCondVal = EvaluateExprAsBool(S.getCond());
  llvm::Value *LHS = EmitScalarExpr(BO->getLHS()); 
  QualType LHSTy = BO->getLHS()->getType();
  llvm::Value *BoolCondVal;
  if (LHSTy->hasSignedIntegerRepresentation()) {
    switch(BO->getOpcode()) {
      case BO_GT:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_SGT, LHS, RHS, "cmp");
        break;
      case BO_GE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_SGE, LHS, RHS, "cmp");
        break;
      case BO_LT:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_SLT, LHS, RHS, "cmp");
        break;
      case BO_LE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_SLE, LHS, RHS, "cmp");
        break;
      case BO_NE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_NE, LHS, RHS, "cmp");
        break;
      case BO_EQ:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_EQ, LHS, RHS, "cmp");
        break;
      default:
        llvm_unreachable("Invalid comparison in forall");
    }
  } else {
    switch(BO->getOpcode()){
      case BO_GT:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_UGT, LHS, RHS, "cmp");
        break;
      case BO_GE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_UGE, LHS, RHS, "cmp");
        break;
      case BO_LT:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_ULT, LHS, RHS, "cmp");
        break;
      case BO_LE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_ULE, LHS, RHS, "cmp");
        break;
      case BO_NE:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_NE, LHS, RHS, "cmp");
        break;
      case BO_EQ:
        BoolCondVal = Builder.CreateICmp(llvm::ICmpInst::ICMP_EQ, LHS, RHS, "cmp");
        break;
      default:
        llvm_unreachable("Invalid comparison in forall");
    }
  }

  Builder.CreateCondBr(
      BoolCondVal, Detach, Sync.getBlock(),
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody())));

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(Sync);
  }

  // Emits the detach block for parallel execution along with its Tapir
  // terminator. This is where we capture the induction variable by value and
  // store it on the stack of the calling thread.
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init. This is guaranteed to exist.
  const DeclStmt *DS = cast<DeclStmt>(S.getInit());

  // Create threadsafe induction variables before the detach and put them in
  // IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment.getBlock(), SRStart);

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  {
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    RunCleanupsScope BodyScope(*this);

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

    EmitStmt(S.getBody());
  }

  // Unwind the codegen of the induction variable from the current local thread
  // safe copy back to the original induction variable. We also need to emit the
  // reattach block and reset the alloca insertion point.

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
  PopSyncRegion();

  // Emit the fall-through block.
  EmitBlock(LoopExit.getBlock(), true);
}

void CodeGenFunction::EmitCXXForallRangeStmt(
    const CXXForallRangeStmt &S, ArrayRef<const Attr *> ForallAttr) {
  assert(CGM.getKitsuneOpts().getTTID().has_value() &&
         "TTID not set in Kitsune options");
  llvm::TTID TT = *CGM.getKitsuneOpts().getTTID();
  if (std::optional<llvm::TTID> AttrTT = GetTapirTargetAttr(ForallAttr))
    TT = *AttrTT;
  if (TT != llvm::TTID::Nolo)
    LoopStack.setLoopTarget(TT);

  if (TT == llvm::TTID::Cuda || TT == llvm::TTID::Hip) {
    unsigned ThreadsPerBlock = GetKitsuneLaunchAttr(ForallAttr);
    if (ThreadsPerBlock > 0)
      LoopStack.setLoopThreadsPerBlock(ThreadsPerBlock);
  }

  // Code modifications necessary for implementing parallel loops not required
  // by serial loops.

  // new basic blocks and jump destinations with Tapir terminators
  llvm::BasicBlock *Detach = createBasicBlock("forall.detach");
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
  LoopStack.setSpawnStrategy(llvm::TapirSpawnStrategy::DivideAndConquer);

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
  LoopStack.push(CondBlock, CGM.getContext(), CGM.getCodeGenOpts(), ForallAttr,
                 SourceLocToDebugLoc(R.getBegin()),
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
  llvm::MDNode *Weights =
      createProfileWeightsForLoop(S.getCond(), getProfileCount(S.getBody()));
  Builder.CreateCondBr(BoolCondVal, Detach, ExitBlock, Weights);

  if (ExitBlock != LoopExit.getBlock()) {
    EmitBlock(ExitBlock);
    EmitBranchThroughCleanup(LoopExit);
  }

  // Emits the detach block for parallel execution along with its Tapir
  // terminator. This is where we capture the induction variable by value and
  // store it on the stack of the calling thread.

  // Emit the (currently empty) detach block
  EmitBlock(Detach);

  // Extract the DeclStmt from the statement init
  const DeclStmt *DS = cast<DeclStmt>(S.getIndexStmt());

  // Create threadsafe induction variables before the detach and put them in
  // IVDeclMap
  for (auto *DI : DS->decls())
    EmitIVLoad(dyn_cast<VarDecl>(DI), IVDeclMap);

  // Create a block for the increment. In case of a 'continue', we jump there.
  llvm::BasicBlock *Increment = createBasicBlock("forall.inc");

  // create the detach terminator
  Builder.CreateDetach(ForBody, Increment, SRStart);

  EmitBlock(ForBody);
  incrementProfileCounter(&S);

  // Store the blocks to use for break and continue.
  BreakContinueStack.push_back(BreakContinue(LoopExit, Reattach));

  {
    // Create a separate cleanup scope for the loop variable and body.
    LexicalScope BodyScope(*this, S.getSourceRange());

    // Change the alloca insert point so that the alloca's happen after the
    // detach and within the body block. This makes sure each thread has its own
    // local copy of the induction variable. We also need to store the thread
    // safe value from the calling thread into this local copy. In
    // EmitThreadSafeIV, we use AutoVarAlloca so any codegen in the body
    // automatically and correctly mapped to the local thread safe copy of the
    // induction variable.

    // change the alloca insert point to the body block
    SetAllocaInsertPoint(Undef, ForBody);

    // emit the thread safe induction variables and initialize them by value
    for (const auto &ivp : IVDeclMap)
      EmitThreadSafeIV(ivp.first, ivp.second.second);

    EmitStmt(S.getLoopVarStmt());
    EmitStmt(S.getBody());
  }

  // Unwind the codegen of the induction variable from the current local thread
  // safe copy back to the original induction variable. We also need to emit the
  // reattach block and reset the alloca insertion point.

  // Restore induction variable mappings after emitting body, and before
  // the increment
  for (const auto &ivp : IVDeclMap)
    RestoreDeclMap(ivp.first, ivp.second.first);

  EmitBlock(Reattach.getBlock());
  Builder.CreateReattach(Increment, SRStart);

  // reset the alloca insertion point
  AllocaInsertPt->removeFromParent();
  AllocaInsertPt = OldAllocaInsertPt;

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
  PopSyncRegion();

  EmitBlock(End, true);
}
