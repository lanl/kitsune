//===- GPUABI.cpp - Lower Tapir to the Kitsune GPU back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune GPU ABI to convert Tapir instructions to
// calls into the Kitsune runtime system for GPU LLVM code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Tapir/GPUABI.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/TargetRegistry.h"
#include <sstream>
#include <fstream>

using namespace llvm;

#define DEBUG_TYPE "gpuabi"

Value *GPUABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);

  // Replace uses of grainsize intrinsic call with this grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  return Grainsize;
}

void GPUABI::lowerSync(SyncInst &SI) {
  // currently a no-op...
}

void GPUABI::preProcessOutlinedTask(llvm::Function&, llvm::Instruction*, llvm::Instruction*, bool, BasicBlock*){}
void GPUABI::postProcessOutlinedTask(llvm::Function&, llvm::Instruction*, llvm::Instruction*, bool, BasicBlock*){}
void GPUABI::preProcessRootSpawner(llvm::Function&, BasicBlock *TFEntry){}
void GPUABI::postProcessRootSpawner(llvm::Function&, BasicBlock *TFEntry){}

void GPUABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
}

void GPUABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
}

void GPUABI::postProcessHelper(Function &F) {
}

void GPUABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
}

LoopOutlineProcessor *
GPUABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  if(!LOP) 
    return new LLVMLoop(M);
  return LOP;
}

// Static counter for assigning IDs to kernels.
unsigned LLVMLoop::NextKernelID = 0;

LLVMLoop::LLVMLoop(Module &M)
    : LoopOutlineProcessor(M, LLVMM), LLVMM("kernelModule", M.getContext()) {
  ValueToValueMapTy VMap; 
  // LLVMMptr = CloneModule(M, vmap, [](const GlobalValue* gv) { return false; });  
  // And named metadata....
  for (const NamedMDNode &NMD : M.named_metadata()) {
    NamedMDNode *NewNMD = LLVMM.getOrInsertNamedMetadata(NMD.getName());
    for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i)
      NewNMD->addOperand(MapMetadata(NMD.getOperand(i), VMap));
  }
  // Assign an ID to this kernel.
  MyKernelID = NextKernelID++;

  // Setup an LLVM triple.
  Triple LLVMTriple("spir64-unknown-unknown");
  LLVMM.setTargetTriple(LLVMTriple.str());

  // Insert runtime-function declarations in LLVM host modules.
  Type *LLVMInt32Ty = Type::getInt32Ty(LLVMM.getContext());
  Type *LLVMInt64Ty = Type::getInt64Ty(LLVMM.getContext());
  GetThreadIdx = LLVMM.getOrInsertFunction("gtid", LLVMInt32Ty);
  Function* getid = LLVMM.getFunction("gtid"); 

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *Int8Ty = Type::getInt8Ty(M.getContext());
  Type *Int32Ty = Type::getInt32Ty(M.getContext());
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  GPUInit = M.getOrInsertFunction("initRuntime", VoidTy);
  GPULaunchKernel = M.getOrInsertFunction("launchBCKernel", VoidPtrTy, VoidPtrTy, Int64Ty, VoidPtrPtrTy, Int64Ty);
  GPUWaitKernel = M.getOrInsertFunction("waitKernel", VoidTy, VoidPtrTy);
}

void LLVMLoop::setupLoopOutlineArgs(
    Function &F, ValueSet &HelperArgs, SmallVectorImpl<Value *> &HelperInputs,
    ValueSet &InputSet, const SmallVectorImpl<Value *> &LCArgs,
    const SmallVectorImpl<Value *> &LCInputs, const ValueSet &TLInputsFixed) {
  // Add the loop control inputs.

  // The first parameter defines the extent of the index space, i.e., the number
  // of threads to launch.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The second parameter defines the start of the index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }
  // The third parameter defines the grainsize, if it is not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("runStride");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop control inputs.
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V); 
    HelperInputs.push_back(V);
  }

  for(Value *V : HelperInputs){
    OrderedInputs.push_back(V);
  }
}

unsigned LLVMLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned LLVMLoop::getLimitArgIndex(const Function &F, const ValueSet &Args)
  const {
  // The argument for the loop limit is the first input.
  return 0;
}

void LLVMLoop::postProcessOutline(TapirLoopInfo &TL, TaskOutlineInfo &Out,
                                   ValueToValueMapTy &VMap) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  //Type *Int64Ty = Type::getInt64Ty(Ctx);
  //Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();


  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);
  Instruction *ClonedSyncReg = cast<Instruction>(
      VMap[T->getDetach()->getSyncRegion()]);

  // We no longer need the cloned sync region.
  ClonedSyncReg->eraseFromParent();

  // Set the helper function to have external linkage.
  // Get the thread ID for this invocation of Helper.
  IRBuilder<> B(Entry->getTerminator());
  Value *ThreadIdx = B.CreateCall(GetThreadIdx);
  //Value *BlockIdx = B.CreateCall(GetBlockIdx, ConstantInt::get(Int32Ty, 0));
  //Value *BlockDim = B.CreateCall(GetBlockDim, ConstantInt::get(Int32Ty, 0));
  Value *ThreadID = B.CreateIntCast(ThreadIdx, PrimaryIV->getType(), false);


  Function *Helper = Out.Outline;
  Helper->setName("kitsune_kernel"); 
  // Fix argument pointer types to global, nocapture
  // TODO: read/write attributes?
  LLVM_DEBUG(dbgs() << "Function type after globalization of argument pointers << " << *Helper->getType() << "\n"); 
  LLVM_DEBUG(dbgs() << "LLVMM after globalization of argument pointers << " << *Helper->getParent() << "\n"); 

  // Verify that the Thread ID corresponds to a valid iteration.  Because Tapir
  // loops use canonical induction variables, valid iterations range from 0 to
  // the loop limit with stride 1.  The End argument encodes the loop limit.
  // Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    auto OutlineArgsIter = Helper->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter;

    // Get the grainsize value, which is either constant or the third LC arg.
    // ReplaceInstWithInst(gep, GetElementPtrInst::Create(
    if (unsigned ConstGrainsize = TL.getGrainsize())
      Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    else
      // Grainsize argument is the third LC arg.
      Grainsize = &*++(++OutlineArgsIter);
  }
  ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEndGrain = B.CreateAdd(ThreadID, Grainsize);
  Value *Cmp = B.CreateICmp(ICmpInst::ICMP_ULT, ThreadEndGrain, End); 
  Value *ThreadEnd = B.CreateSelect(Cmp, ThreadEndGrain, End); 
  Value *Cond = B.CreateICmpUGE(ThreadID, ThreadEnd);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != ThreadEnd)
    ++TripCountIdx;
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  assert(ClonedCond->getOperand(TripCountIdx) == ThreadEnd &&
         "End argument not used in condition");

}

void LLVMLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  //Task *T = TL.getTask();
  //Instruction *ReplCall = cast<CallBase>(TOI.ReplCall);
  LLVM_DEBUG(dbgs() << "Running processOutlinedLoopCall: " << LLVMM);
  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock* RCBB = TOI.ReplCall->getParent(); 
  BasicBlock* NBB = RCBB->splitBasicBlock(TOI.ReplCall); 
  TOI.ReplCall->eraseFromParent(); 

  IRBuilder<> B(&NBB->front());

  // Compile the kernel 
  //LLVMM.getFunctionList().remove(TOI.Outline); 
  //TOI.Outline->eraseFromParent(); 
  LLVMContext &LLVMCtx = LLVMM.getContext();

  ValueToValueMapTy VMap; 
  // We recursively add definitions and declarations to the device module
  SmallVector<Function*> todo;
  todo.push_back(LLVMM.getFunction("kitsune_kernel"));
  while(!todo.empty()){
    auto *F = todo.back(); 
    todo.pop_back(); 
    for(auto &BB : *F){
      for(auto &I : BB){
        if(auto *CI = dyn_cast<CallInst>(&I)){
          if(Function *f = CI->getCalledFunction()){
            if(f->getParent() != &LLVMM){
              // TODO: improve check for function, could be overloaded
              auto *deviceF = LLVMM.getFunction(f->getName()); 
              if(!deviceF){
                if(f->getParent() == &M){
                  deviceF = Function::Create(f->getFunctionType(), f->getLinkage(), f->getName(), LLVMM);  
                  VMap[f] = deviceF; 
                  auto *NewFArgIt = deviceF->arg_begin();
                  for (auto &Arg : f->args()) {
                    auto ArgName = Arg.getName();
                    NewFArgIt->setName(ArgName);
                    VMap[&Arg] = &(*NewFArgIt++);
                  }
                  SmallVector<ReturnInst*, 8> Returns;
                  CloneFunctionInto(deviceF, f, VMap, CloneFunctionChangeType::DifferentModule, Returns); 
                  // GPU calls are slow as balls, try to force inlining
                  deviceF->addFnAttr(Attribute::AlwaysInline); 
                  todo.push_back(deviceF);
                }
              } 
              CI->setCalledFunction(deviceF); 
            }
          }
        }
        for(auto &op : I.operands()){
          if(GlobalVariable *GV = dyn_cast<GlobalVariable>(op)) {
            if(GV->getParent() == &M){
              GlobalVariable *NewGV = new GlobalVariable(
                  LLVMM, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
                  (Constant *)nullptr, GV->getName(), (GlobalVariable *)nullptr,
                  GV->getThreadLocalMode(), GV->getType()->getAddressSpace());
              NewGV->copyAttributesFrom(GV);
              VMap[op] = NewGV;
              const Comdat *SC = GV->getComdat();
              if (!SC)
                return;
              Comdat *DC = NewGV->getParent()->getOrInsertComdat(SC->getName());
              DC->setSelectionKind(SC->getSelectionKind());
              NewGV->setComdat(DC);
              NewGV->setLinkage(GV->getLinkage()); 
              NewGV->setInitializer(GV->getInitializer()); 
              op = NewGV; 
            }
          }
        }
      }
    }
  }

  SmallVector<Metadata *, 3> AV;
  //AV.push_back(ValueAsMetadata::get(Helper));
  //AV.push_back(MDString::get(LLVMCtx, "kernel"));
  //AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(LLVMCtx),
  //                                                   1)));
  //Annotations->addOperand(MDNode::get(Ctx, AV));

  legacy::PassManager *PassManager = new legacy::PassManager;


  // Add in our optimization passes

  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createReassociatePass());
  PassManager->add(createGVNPass());
  PassManager->add(createCFGSimplificationPass());
  PassManager->add(createLoopVectorizePass());
  PassManager->add(createSLPVectorizerPass());
  //PassManager->add(createBreakCriticalEdgesPass());
  //PassManager->add(createConstantPropagationPass());
  PassManager->add(createDeadCodeEliminationPass());
  PassManager->add(createDeadStoreEliminationPass());
  //PassManager->add(createInstructionCombiningPass());
  PassManager->add(createCFGSimplificationPass());
  PassManager->add(createDeadCodeEliminationPass());
  PassManager->add(createVerifierPass());
  PassManager->run(LLVMM);

  delete PassManager;

  LLVM_DEBUG(dbgs() << "LLVM Module after passes: " << LLVMM);

  // generate llvm kernel code
  SmallVector<char, 1<<20> mbuf;
  BitcodeWriter bcw(mbuf);
  bcw.writeModule(LLVMM); 
  bcw.writeStrtab(); 

  char* heapbuf = new char[mbuf.size()];
  for(int i=0; i<mbuf.size(); i++) { heapbuf[i] = mbuf[i]; } 

  std::string strbuf(mbuf.data(), mbuf.size()); 
  std::ofstream out("compile_time.bc"); 
  out << strbuf;
  out.close();

  Constant *LLVMBC = ConstantDataArray::getRaw(StringRef(heapbuf, mbuf.size()), mbuf.size(), Int8Ty);
  LLVMGlobal = new GlobalVariable(M, LLVMBC->getType(), true,
                                 GlobalValue::PrivateLinkage, LLVMBC,
                                 "gpu_" + Twine("kitsune_kernel"));

  //Value* TripCount = isSRetInput(TOI.InputSet[0]) ? TOI.InputSet[1] : TOI.InputSet[0]; 
  //Value *RunStart = ReplCall->getArgOperand(getIVArgIndex(*Parent,
  //                                                        TOI.InputSet));
  //Value *TripCount = ReplCall->getArgOperand(getLimitArgIndex(*Parent,
  //                                                            TOI.InputSet));

  Value *KernelID = ConstantInt::get(Int32Ty, MyKernelID);
  Value *LLVMPtr = B.CreateBitCast(LLVMGlobal, VoidPtrTy);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  Constant *kernelSize = ConstantInt::get(Int64Ty, 
    LLVMGlobal->getInitializer()->getType()->getArrayNumElements()); 
  BasicBlock &EBB = Parent->getEntryBlock(); 
  IRBuilder<> EB(&EBB.front()); 
  EB.CreateCall(GPUInit, {});

  ArrayType* arrayType = ArrayType::get(VoidPtrTy, OrderedInputs.size()); 
  Value* argArray = B.CreateAlloca(arrayType); 
  int i=0;
  for (Value *V : OrderedInputs) {
    //Value *ElementSize = nullptr;
    LLVM_DEBUG(dbgs() << "Input set value: " << *V << "\n"); 
    Value *VPtr = B.CreateAlloca(V->getType()); 
    B.CreateStore(V, VPtr); 
    Value *VoidVPtr = B.CreateBitCast(VPtr, VoidPtrTy);
    Value *argPtr = B.CreateConstInBoundsGEP2_32(arrayType, argArray, 0, i++); 
    B.CreateStore(VoidVPtr, argPtr); 
  }

  Value *Grainsize = TL.getGrainsize() ?  
    ConstantInt::get(TripCount->getType(), TL.getGrainsize()) :
    OrderedInputs[2]; 

  //Type *Int64Ty = Type::getInt64Ty(LLVMM.getContext());
  Value *RunSizeQ = B.CreateUDiv(TripCount, Grainsize);
  Value *RunRem = B.CreateURem(TripCount, Grainsize);
  Value *IsRem = B.CreateICmp(ICmpInst::ICMP_UGT, RunRem, ConstantInt::get(RunRem->getType(), 0)); 
  Value *IsRemAdd = B.CreateZExt(IsRem, RunSizeQ->getType()); 
  Value *RunSize = B.CreateZExt(B.CreateAdd(RunSizeQ, IsRemAdd), Int64Ty);  

  Value* argsPtr = B.CreateConstInBoundsGEP2_32(arrayType, argArray, 0, 0); 
  Value* bcPtr = B.CreateConstInBoundsGEP2_32(LLVMGlobal->getValueType(), LLVMGlobal, 0, 0); 
  Value* stream = B.CreateCall(GPULaunchKernel, { bcPtr, kernelSize, argsPtr, RunSize });
  B.CreateCall(GPUWaitKernel, stream);

  LLVM_DEBUG(dbgs() << "Finished processOutlinedLoopCall: " << M);
}

