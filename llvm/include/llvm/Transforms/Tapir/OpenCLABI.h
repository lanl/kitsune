//===- OpenCLABI.h - Interface to the Kitsune OpenCL back end ------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune OpenCL ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for NVIDIA GPU code.
//
//===----------------------------------------------------------------------===//
#ifndef OpenCL_ABI_H_
#define OpenCL_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"

namespace llvm {

class DataLayout;
class TargetMachine;
class SPIRVLoop; 

class OpenCLABI : public TapirTarget {
  SPIRVLoop *LOP = nullptr;
public:
  OpenCLABI(Module &M) : TapirTarget(M) {}
  ~OpenCLABI() {}
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

class SPIRVLoop : public LoopOutlineProcessor {
  friend class OpenCLABI; 

private:
  static unsigned NextKernelID;
  unsigned MyKernelID;
  Module SPIRVM;
  TargetMachine *SPIRVTargetMachine;
  GlobalVariable *SPIRVGlobal;

  FunctionCallee GetThreadIdx = nullptr;
  FunctionCallee GetBlockIdx = nullptr;
  FunctionCallee GetBlockDim = nullptr;
  FunctionCallee KitsuneOpenCLInit = nullptr;
  FunctionCallee KitsuneGPUInitKernel = nullptr;
  FunctionCallee KitsuneGPUMemWrite = nullptr;
  FunctionCallee KitsuneGPUMemRead = nullptr;
  FunctionCallee KitsuneGPUSetArg = nullptr;
  FunctionCallee KitsuneGPUSetRunSize = nullptr;
  FunctionCallee KitsuneGPURunKernel = nullptr;
  FunctionCallee KitsuneGPUFinish = nullptr;

  SmallVector<Value *, 5> OrderedInputs; 
public:
  SPIRVLoop(Module &M);

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

class OpenCL : public LoopOutlineProcessor {
public:
  OpenCL(Module &M) : LoopOutlineProcessor(M) {}
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                          ValueToValueMapTy &VMap) override final;  
  GlobalVariable* SPIRVKernel; 
};
*/
