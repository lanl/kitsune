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

#pragma warning "GPUABI has been deprecated"
#if 0

#include "llvm/Transforms/Tapir/GPUABI.h"
#include "llvm/Transforms/Tapir/TapirToTarget.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include <sstream>
#include <fstream>

using namespace llvm;

#define DEBUG_TYPE "gpuabi"

// JIT compiler kernel at containing function entry, makes timing easier at the
// cost of less laziness
static cl::opt<bool>
    JIT("jit-callsite", cl::init(false), cl::NotHidden,
          cl::desc("Wait until parallel loop is called to jit kernel. " 
          "(default=false)"));

Value *GPUABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  //Value *Grainsize = ConstantInt::get(GrainsizeCall->getType(), 8);
  Module *M = GrainsizeCall->getModule(); 
  Type *Int64Ty = Type::getInt64Ty(M->getContext());
  Value* Grainsize = CallInst::Create(M->getOrInsertFunction("gpuGridSize", Int64Ty), {},  GrainsizeCall); 
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
  GetThreadIdx = LLVMM.getOrInsertFunction("gtid", LLVMInt32Ty);

  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *VoidPtrTy = Type::getInt8PtrTy(M.getContext());
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *Int64Ty = Type::getInt64Ty(M.getContext());
  GPUInit = M.getOrInsertFunction("initRuntime", VoidTy);
  GPULaunchKernel = M.getOrInsertFunction("launchBCKernel", VoidPtrTy, VoidPtrTy, Int64Ty, VoidPtrPtrTy, Int64Ty);
  GPUWaitKernel = M.getOrInsertFunction("waitKernel", VoidTy, VoidPtrTy);
  GPUManagedMalloc = M.getOrInsertFunction("gpuManagedMalloc", VoidPtrTy, Int64Ty);  
  GPUGridSize = M.getOrInsertFunction("gpuGridSize", Int64Ty);  
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
  Task *T = TL.getTask();
  Loop *L = TL.getLoop();

  Function *Helper = Out.Outline;

  DominatorTree DT;
  DT.recalculate(*Helper); 
  LoopInfo LI(DT); 

  // The loop we care about is the outermost loop in the 

  BasicBlock *Entry = cast<BasicBlock>(VMap[L->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[L->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TL.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TL.getPrimaryInduction().first]);

  Loop *NewLoop = LI.getLoopFor(Header); 

  InductionDescriptor ID = TL.getPrimaryInduction().second; 
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
  Value *Cond = B.CreateICmpUGE(ThreadID, End);

  ReplaceInstWithInst(Entry->getTerminator(), BranchInst::Create(Exit, Header,
                                                                 Cond));
  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TL.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  ClonedCond->setOperand(TripCountIdx, End);
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition");
  // we replace the step
  assert(TL.getPrimaryInduction().second.getInductionBinOp()->getOpcode() == Instruction::BinaryOps::Add &&
        "Only support gpu kernels with addition for induction step"); 
  ConstantInt *CStep = cast<ConstantInt>(VMap[ID.getConstIntStepValue()]);  
  BinaryOperator *BinOp = cast<BinaryOperator>(VMap[ID.getInductionBinOp()]); 
  assert(CStep && BinOp && "Couldn't infer step or operator from primary induction gpu variable"); 
  IRBuilder<> IncB(&BinOp->getParent()->front());  
  Value* NewStep = IncB.CreateMul(CStep, Grainsize); 
  BinOp->setOperand(1, NewStep); 
  ClonedCond->setPredicate(ICmpInst::Predicate::ICMP_UGE); 

  // Make each thread reduce into its local memory
  for(auto &BB : NewLoop->getBlocks()){
    for(auto &I : *BB){
      if(auto *CI = dyn_cast<CallInst>(&I)){
        auto *f = CI->getCalledFunction(); 
        if(f->getAttributes().hasAttrSomewhere(Attribute::KitsuneReduction)){
          IRBuilder<> PB(&I); 
          auto ptr = CI->getOperand(0); 
          auto ty = CI->getOperand(1)->getType();  
          auto lptr = PB.CreateGEP(ty, ptr, ThreadIdx); 
          CI->setOperand(0, lptr); 
          f->removeFnAttr(Attribute::NoInline); 
        }
      }
    }
  }

  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  TargetLibraryInfo TLI(TLII); 
  AssumptionCache AC(*Helper); 
  ScalarEvolution SE(*Helper, TLI, AC, DT, LI); 
  ValueToValueMapTy peelVMap; 
  simplifyLoop(NewLoop, &DT, &LI, &SE, &AC, nullptr, false); 
  peelLoop(NewLoop, 1, &LI, &SE, DT, &AC, false, peelVMap); 
  SmallVector<Instruction*, 4> cis;
  for(auto &BB : *Helper){
    if(!NewLoop->contains(&BB)){ // better way?
      for(auto &I : BB){
        if(auto *CI = dyn_cast<CallInst>(&I)){
          auto *f = CI->getCalledFunction(); 
          if(f->getAttributes().hasAttrSomewhere(Attribute::KitsuneReduction)){
            IRBuilder<> pb(&I); 
            pb.CreateStore(CI->getArgOperand(1), CI->getArgOperand(0)); 
            cis.push_back(&I); 
          }
        }
      }
    }
  }
  for(auto &I : cis){
    I->eraseFromParent(); 
  }
}

void LLVMLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {
  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  LLVM_DEBUG(dbgs() << "Running processOutlinedLoopCall: " << LLVMM);
  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock* RCBB = TOI.ReplCall->getParent();
  BasicBlock* NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());

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
                  if(!deviceF->hasFnAttribute(Attribute::NoInline))
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
  for(size_t i=0; i<mbuf.size(); i++) { heapbuf[i] = mbuf[i]; }

  std::string strbuf(mbuf.data(), mbuf.size());
  std::ofstream out("compile_time.bc");
  out << strbuf;
  out.close();

  Constant *LLVMBC = ConstantDataArray::getRaw(StringRef(heapbuf, mbuf.size()), mbuf.size(), Int8Ty);
  LLVMGlobal = new GlobalVariable(M, LLVMBC->getType(), true,
                                 GlobalValue::PrivateLinkage, LLVMBC,
                                 "gpu_" + Twine("kitsune_kernel"));

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

  Value* argsPtr = B.CreateConstInBoundsGEP2_32(arrayType, argArray, 0, 0); 
  Value* bcPtr = B.CreateConstInBoundsGEP2_32(LLVMGlobal->getValueType(), LLVMGlobal, 0, 0); 
  Value* stream = B.CreateCall(GPULaunchKernel, { bcPtr, kernelSize, argsPtr, TripCount });
  B.CreateCall(GPUWaitKernel, stream);

  LLVM_DEBUG(dbgs() << "Finished processOutlinedLoopCall: " << M);
}

void LLVMLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap, LoopInfo &LI) {
  Loop *L = TL.getLoop();  
  BasicBlock *PH = L->getLoopPreheader(); 
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  BranchInst *LatchBR = cast<BranchInst>(Latch->getTerminator());
  BasicBlock &Entry = Header->getParent()->getEntryBlock(); 
  unsigned ExitIndex = LatchBR->getSuccessor(0) == Header ? 1 : 0;
  BasicBlock *LatchExit = LatchBR->getSuccessor(ExitIndex);
  DetachInst *DI = cast<DetachInst>(Header->getTerminator());
  Value *SyncReg = DI->getSyncRegion();
  Value *GS; 

  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  // accumulate Reductions in main loop
  const std::vector<BasicBlock*>& Blocks = L->getBlocks(); 
  std::set<std::pair<CallInst*, Type*>> Reductions;
  for (BasicBlock *BB : Blocks){
    for (Instruction &I : *BB) {
      if(auto *CI = dyn_cast<CallInst>(&I)){
        auto *F = CI->getCalledFunction(); 
        if(F->getAttributes().hasAttrSomewhere(Attribute::KitsuneReduction)){
          LLVM_DEBUG(dbgs() << "Found reduction var: " << CI->getArgOperand(0)->getName() << 
                               "with reduction function: " << F->getName() << "\n"); 
          auto *Ty = CI->getArgOperand(1)->getType(); 
          Reductions.insert(std::make_pair(CI, Ty)); 
          //TODO: check the type to confirm valid reduction
        }
      }
    }
  }

  std::vector<std::tuple<CallInst*, Value* , Value*, Type*>> RedMap; 

  for(auto &Red : Reductions){
    // We assume there's a preheader, and use it to insert allocations for reductions:
    // preheader: 
    //   gs = gridSize()
    //   reds = gpuManagedMalloc(gs)
    //   br header
    IRBuilder<> EB(Entry.getFirstNonPHI()); 
    CallInst *CI = Red.first; 
    Value *Ptr = CI->getArgOperand(0); 
    Type *Ty = Red.second; 
    GS = EB.CreateCall(GPUGridSize); 
    Value *NBytes = EB.CreateMul(GS, ConstantInt::get(GS->getType(), DL.getTypeAllocSize(Ty))); 
    CallInst *Alloc = EB.CreateCall(GPUManagedMalloc, {NBytes}); 
    // We overwrite in the body the location to Alloc, which will be replaced with a GEP using the tid in postprocessing
    CI->setOperand(0, Alloc); 

    RedMap.push_back(std::make_tuple(CI, Ptr, Alloc, Ty)); 
  }

  if(!Reductions.empty()){
    Instruction* Term = LatchExit->getTerminator(); 
    BasicBlock *PostSync = Term->getSuccessor(0);
    BasicBlock* RedEpiHeader = BasicBlock::Create(LatchExit->getContext(), "reductionEpilogue", LatchExit->getParent(), LatchExit); 
    RedEpiHeader->moveAfter(LatchExit); 
    ReplaceInstWithInst(Term, SyncInst::Create(RedEpiHeader, SyncReg)); 
    BranchInst::Create(PostSync, RedEpiHeader);
    PHINode *Idx = PHINode::Create(GS->getType(), 2,
                                   "reductionepilogueidx",
                                   RedEpiHeader->getFirstNonPHI());
    IRBuilder<> BH(RedEpiHeader->getFirstNonPHI()); 
    Idx->addIncoming(ConstantInt::get(GS->getType(), 0), LatchExit);
    Instruction *BodyTerm, *ExitTerm;
    Value *Cmp = BH.CreateCmp(CmpInst::ICMP_NE, Idx, GS); 
    SplitBlockAndInsertIfThenElse(Cmp, RedEpiHeader->getTerminator(), &BodyTerm, &ExitTerm);

    IRBuilder<> BB(BodyTerm); 
    // For each reduction, get the allocated thread local reduced values and
    // reduce them. 
    for(auto& KV : RedMap){
      const auto [ CI, Ptr, Al, Ty ] = KV; 
      Value* Lptr = BB.CreateBitCast(
        BB.CreateGEP(Ty, Al, Idx), 
        Ptr->getType());                             
      Value *LR = BB.CreateLoad(Ty, Lptr); 
      BB.SetCurrentDebugLocation(CI->getDebugLoc()); 
      BB.CreateCall(CI->getCalledFunction(), { Ptr, LR }); 
    }
    Value *IdxAdd =
        BB.CreateAdd(Idx, ConstantInt::get(Idx->getType(), 1),
                          Idx->getName() + ".add");
    BasicBlock* Body = BodyTerm->getParent(); 
    Idx->addIncoming(IdxAdd, Body); 
    ReplaceInstWithInst(BodyTerm, BranchInst::Create(RedEpiHeader)); 
  }
  
  LLVM_DEBUG(dbgs() << "Finished preProcessTapirLoop: " << *PH->getParent());
}
