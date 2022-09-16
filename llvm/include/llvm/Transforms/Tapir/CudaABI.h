//===- CudaABI.h - Interface to the Kitsune Cuda back end -----*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the Tapir to CUDA transformation that targets the direct CUDA
// support in the Kitsune runtime API.  The transform currently targets
// ahead-of-time (non-JIT) code generation and should allow cross-compilation
// on systems without a local GPU -- as long as the CUDA toolchain is
// available.
//
//===----------------------------------------------------------------------===//
#ifndef TapirCuda_ABI_H_
#define TapirCuda_ABI_H_

#include "llvm/Transforms/Tapir/LoweringUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Support/ToolOutputFile.h"

namespace llvm {

class TargetMachine;
class CudaLoop;

typedef std::unique_ptr<ToolOutputFile> CudaABIOutputFile;

class CudaABI : public TapirTarget {

public:
  CudaABI(Module &M);
  ~CudaABI();

  Value *lowerGrainsizeCall(CallInst *GrainsizeCall) override final;
  void lowerSync(SyncInst &SI) override final;

  void addHelperAttributes(Function &F) override final;
  void preProcessFunction(Function &F, TaskInfo &TI,
                          bool OutliningTapirLoops) override final;
  void postProcessFunction(Function &F, bool OutliningTapirLoops) override final;
  void postProcessHelper(Function &F) override final;

  void preProcessOutlinedTask(Function &F,
                              Instruction *DetachPt,
                              Instruction *TaskFrameCreate,
                              bool IsSpawner,
                              BasicBlock *TFEntry) override final;

  void postProcessOutlinedTask(Function &F,
                               Instruction *DetachPt,
                               Instruction *TaskFrameCreate,
                               bool IsSpawner,
                               BasicBlock *TFEntry) override final;

  void preProcessRootSpawner(Function &F,
                             BasicBlock *TFEntry) override final;
  void postProcessRootSpawner(Function &F,
                              BasicBlock *TFEntry) override final;

  void processSubTaskCall(TaskOutlineInfo &TOI,
                          DominatorTree &DT) override final;

  void postProcessModule() override final;

  LoopOutlineProcessor *getLoopOutlineProcessor(const TapirLoopInfo *TL)
                          override final;

  void pushPTXFilename(const std::string &PTXFilename);

  std::unique_ptr<Module>& getLibDeviceModule();

      void pushGlobalVariable(GlobalVariable *GV);
  bool hasGlobalVariables() const {
    return !GlobalVars.empty();
  }
  int globalVarCount() const {
    return GlobalVars.size();
  }

  private:
    CudaABIOutputFile generatePTX();
    CudaABIOutputFile assemblePTXFile(CudaABIOutputFile &PTXFile);
    CudaABIOutputFile createFatbinaryFile(CudaABIOutputFile &AsmFile);
    GlobalVariable *embedFatbinary(CudaABIOutputFile &FatbinaryFile);
    void registerFatbinary(GlobalVariable *RawFatbinary);
    void finalizeLaunchCalls(Module &M, GlobalVariable *Fatbin);
    void bindGlobalVariables(Value *CM, IRBuilder<> &B);
    Function *createCtor(GlobalVariable *Fatbinary, GlobalVariable *Wrapper);
    Function *createDtor(GlobalVariable *FBHandle);

    std::unique_ptr<Module> LibDeviceModule;

        typedef std::list<std::string> StringListTy;
    StringListTy ModulePTXFileList;
    typedef std::list<GlobalVariable *> GlobalVarListTy;
    GlobalVarListTy GlobalVars;

    Module   KM;
    TargetMachine *PTXTargetMachine;
};

/// The loop outline process for transforming a Tapir parallel loop
/// represention into a Cuda runtime and PTX --> fat binary kernel
/// execution.
///
///  * The loop processor requires a CUDA install and that the 'ptxas'
///    and 'fatbinary' executables are in the user's path.  While it
///    is tempting to inline direct CUDA calls into the transform this
///    has two drawbacks:
///
///      1. CMake dependencies on Cuda would need to be added.
///      2. GPU would be required at compile time (i.e., no cross
///         compilation support).
///
class CudaLoop : public LoopOutlineProcessor {
  friend class CudaABI;

private:
  CudaABI *TTarget = nullptr;
  static unsigned NextKernelID;    // Give the generated kernel a unique ID.
  unsigned KernelID;               // Unique ID for this transformed loop.
  std::string KernelName;          // A unique name for the kernel.
  Module  &KernelModule;           // PTX module holds the generated kernel(s).

  FunctionCallee GetThreadIdx = nullptr;

  // Cuda/PTX thread index access.
  Function *CUThreadIdxX  = nullptr,
           *CUThreadIdxY  = nullptr,
           *CUThreadIdxZ  = nullptr;
  // Cuda/PTX block index and dimensions access.
  Function *CUBlockIdxX   = nullptr,
           *CUBlockIdxY   = nullptr,
           *CUBlockIdxZ   = nullptr;
  Function *CUBlockDimX   = nullptr,
           *CUBlockDimY   = nullptr,
           *CUBlockDimZ   = nullptr;
  // Cuda/PTX grid dimensions access.
  Function *CUGridDimX    = nullptr,
           *CUGridDimY    = nullptr,
           *CUGridDimZ    = nullptr;

  // Cuda thread synchronize
  Function *CUSyncThreads = nullptr;

  FunctionCallee KitCudaLaunchFn = nullptr;
  FunctionCallee KitCudaLaunchModuleFn = nullptr;
  FunctionCallee KitCudaWaitFn   = nullptr;
  FunctionCallee KitCudaMemPrefetchFn = nullptr;
  FunctionCallee KitCudaCreateFBModuleFn = nullptr;
  FunctionCallee KitCudaGetGlobalSymbolFn = nullptr;
  FunctionCallee KitCudaMemcpySymbolToDeviceFn = nullptr;
  SmallVector<Value *, 5> OrderedInputs;

public:
  CudaLoop(Module &M,   // Input module (host side)
           Module &KM,  // Target module for CUDA code
           const std::string &KernelName, // CUDA kernel name
           CudaABI *TT, // Target
           bool MakeUniqueName = true);
  ~CudaLoop();

  void setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                            SmallVectorImpl<Value *> &HelperInputs,
                            ValueSet &InputSet,
                            const SmallVectorImpl<Value *> &LCArgs,
                            const SmallVectorImpl<Value *> &LCInputs,
                            const ValueSet &TLInputsFixed) override final;

  unsigned getIVArgIndex(const Function &F, const ValueSet &Args)
                         const override final;

  unsigned getLimitArgIndex(const Function &F, const ValueSet &Args)
                            const override final;

  std::string getKernelName() const { return KernelName; }

  unsigned getKernelID() const {
    return KernelID;
  }

  void preProcessTapirLoop(TapirLoopInfo &TL,
                           ValueToValueMapTy &VMap) override;
  void postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo & Out,
                          ValueToValueMapTy &VMap) override final;
  void processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo & TOI,
                               DominatorTree &DT) override final;
  void transformForPTX(Function &F);

  Function *resolveLibDeviceFunction(Function *F);
};

}

#endif
