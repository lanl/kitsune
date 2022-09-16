//===- GPUABI.h - Interface to the Kitsune GPU back end ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune GPU ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//
#ifndef GPU_ABI_H_
#define GPU_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {

class DataLayout;
class TargetMachine;
class LLVMLoop; 

class GPUABI : public TapirTarget {
  LLVMLoop *LOP = nullptr;
public:
  GPUABI(Module &M) : TapirTarget(M) {}
  ~GPUABI() {}
  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final {}
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops)
    override final;
  void postProcessHelper(Function &F) override final;
  void preProcessOutlinedTask(Function &F, Instruction *DetachPt,
                              Instruction *TaskFrameCreate, bool IsSpawner,
                              BasicBlock *TFEntry) override final;
  void postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                               Instruction *TaskFrameCreate, bool IsSpawner,
                               BasicBlock *TFEntry) override final;
  void preProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final;
  void postProcessRootSpawner(Function &F, BasicBlock *TFEntry) override final;

  void processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT)
    override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
    override final;
};

class LLVMLoop : public LoopOutlineProcessor {
  friend class GPUABI; 

private:
  static unsigned NextKernelID;
  unsigned MyKernelID;
  Module LLVMM;
  TargetMachine *LLVMTargetMachine;
  GlobalVariable *LLVMGlobal;

  FunctionCallee GetThreadIdx = nullptr;
  FunctionCallee GPUInit = nullptr;
  FunctionCallee GPULaunchKernel = nullptr;
  FunctionCallee GPUWaitKernel = nullptr;

  SmallVector<Value *, 5> OrderedInputs; 
public:
  LLVMLoop(Module &M);

  void setupLoopOutlineArgs(
      Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
      ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
      const SmallVectorImpl<Value *> &LCInputs,
      const ValueSet &TLInputsFixed)
    override final;
  unsigned getIVArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args) const
    override final;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                               DominatorTree &DT) override final;
};
}

#endif
/*
#include "llvm/Transforms/Tapir/LoopSpawningTI.h"
#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm; 

class GPU : public LoopOutlineProcessor {
public:
  GPU(Module &M) : LoopOutlineProcessor(M) {}
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;  
  GlobalVariable* LLVMKernel; 
};
*/
