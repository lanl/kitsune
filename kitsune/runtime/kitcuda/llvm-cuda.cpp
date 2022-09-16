//
//===- llvm-cuda.cpp - Kitsune ABI runtime target CUDA support    ---------===//
//
// TODO: Need to update LANL/Triad Copyright notice.
//
// Copyright (c) 2021, Los Alamos National Security, LLC.
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "llvm-cuda.h"
#include "kitrt-debug.h"
#include "llvm/ADT/APInt.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsNVPTX.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/circular_raw_ostream.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <nvPTXCompiler.h>

#include <sstream>
#include <stdbool.h>


using namespace llvm;

static std::string cudaarch = "sm_80";
static std::string cudafeatures = "+ptx64";

#define NVPTXCOMPILER_SAFE_CALL(x)                                             \
  do {                                                                         \
    nvPTXCompileResult result = x;                                             \
    if (result != NVPTXCOMPILE_SUCCESS) {                                      \
      fprintf(stderr, "kitrt: %s failed, error code %d\n", #x, result);        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void *__kitrt_cuPTXtoELF(const char* ptx) {
  nvPTXCompilerHandle compiler = NULL;
  nvPTXCompileResult status;

  size_t elfSize, infoSize, errorSize;
  char *elf, *infoLog, *errorLog;
  unsigned int minorVer, majorVer;

  std::string gpuName = "--gpu-name=" + cudaarch;
  //std::string gpuFeatures = "--gpu-features=" + cudafeatures;

  const int MAX_ARGS = 32;
  const char* compile_options[MAX_ARGS];
  compile_options[0] = gpuName.c_str();
  int arg_count = 1;
  Optional<std::string> env_opts = sys::Process::GetEnv("KITSUNE_CUDA_JIT_ARGS");
  if (env_opts) {
    char * pch = std::strtok((char*)env_opts->c_str(),";");
    while (pch != NULL) {
      compile_options[arg_count] = pch;
      pch = strtok (NULL, ";");
      arg_count++;
      if (pch != NULL && (arg_count == (MAX_ARGS-1))) {
        printf("Warning: CUDA JIT environment argument count exceeds options array size!\n");
        pch = NULL;
      }
    }
  }


  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                              (size_t)strlen(ptx),  /* ptxCodeLen */
                                              ptx));                /* ptxCode */
  status = nvPTXCompilerCompile(compiler, arg_count, compile_options);

  if (status != NVPTXCOMPILE_SUCCESS) {
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

    if (errorSize != 0) {
      errorLog = (char*)malloc(errorSize+1);
      NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
      printf("Error log: %s\n", errorLog);
      free(errorLog);
    }
    exit(1);
  }

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elfSize));

  elf = (char*) malloc(elfSize);
  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

  if (infoSize != 0) {
    infoLog = (char*)malloc(infoSize+1);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
    printf("Info log: %s\n", infoLog);
    free(infoLog);
  }

  NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
  return elf;
}

std::string __kitrt_cuLLVMtoPTX(Module& m, CUdevice device) {
  //std::cout << "input module: " << std::endl;
  //m.print(llvm::errs(), nullptr);
  LLVMContext& ctx = m.getContext();
  int maj, min, warpsize;
  int sm_count, max_threads_per_blk, max_block_dim_x;
  cuDeviceGetAttribute_p(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute_p(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  cuDeviceGetAttribute_p(&warpsize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
  cuDeviceGetAttribute_p(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  cuDeviceGetAttribute_p(&max_threads_per_blk, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
  cuDeviceGetAttribute_p(&max_block_dim_x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);

  //fprintf(stderr, "number of SMs on device: %d\n", sm_count);
  //fprintf(stderr, "warp size: %d\n", warpsize);
  //fprintf(stderr, "maximum number of threads per block: %d\n", max_threads_per_blk);
  //fprintf(stderr, "maximum blocks in the x dimension: %d\n", max_block_dim_x);

  std::ostringstream arch;
  arch << "sm_" << maj << min;
  cudaarch = arch.str();
  fprintf(stderr, "cuda arch: %s\n", cudaarch.c_str());

  Triple TT("nvptx64", "nvidia", "cuda");
  m.setTargetTriple(TT.str());
  Function& F = *m.getFunction("kitsune_kernel");

  AttrBuilder Attrs;
  Attrs.addAttribute("target-cpu", cudaarch);
  Attrs.addAttribute("target-features", cudafeatures + ",+" + cudaarch);
  /*
  Attrs.addAttribute(Attribute::NoRecurse);
  Attrs.addAttribute(Attribute::Convergent);
  */
  F.removeFnAttr("target-cpu");
  F.removeFnAttr("target-features");
  /*
  F.removeFnAttr(Attribute::StackProtectStrong);
  F.removeFnAttr(Attribute::UWTable);
  */
  F.addAttributes(AttributeList::FunctionIndex, Attrs);
  NamedMDNode *Annotations =
    m.getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(&F));
  AV.push_back(MDString::get(ctx, "kernel"));
  AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(ctx),
                                                     1)));
  Annotations->addOperand(MDNode::get(ctx, AV));

  auto tid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_tid_x);
  auto ntid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  auto ctaid = Intrinsic::getDeclaration(&m, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);

  IRBuilder<> B(F.getEntryBlock().getFirstNonPHI());
  Value *tidv = B.CreateCall(tid, {});
  Value *ntidv = B.CreateCall(ntid, {});
  Value *ctaidv = B.CreateCall(ctaid, {});

  Value *tidoff = B.CreateMul(ctaidv, ntidv);
  Value *gtid = B.CreateAdd(tidoff, tidv);

  // PTXAS doesn't like .<n> global names
  for(GlobalVariable & g : m.globals()){
    auto name = g.getName().str();
    for(int i=0; i<name.size(); i++){
      if(name[i] == '.') name[i] = '_';
      //std::cout << name << std::endl;
      g.setName(name);
    }
  }

  // Check if there are unresolved sumbbols to see if we might need libdevice
  std::set<std::string> unresolved;
  for(auto &f : m) {
    if(f.hasExternalLinkage()){
      unresolved.insert(f.getName().str());
    }
  }

  if(!unresolved.empty()){
    // Load libdevice and check for provided functions
    llvm::SMDiagnostic SMD;
    Optional<std::string> path = sys::Process::FindInEnvPath("CUDA_PATH","nvvm/libdevice/libdevice.10.bc");
    if(!path){
      //path = "/opt/cuda/nvvm/libdevice/libdevice.10.bc";
      std::cerr << "Failed to find libdevice\n";
      exit(1);
    }
    std::unique_ptr<llvm::Module> libdevice =
        parseIRFile(*path, SMD, ctx);
    if(!libdevice){
      std::cerr << "Failed to parse libdevice\n";
      exit(1);
    }
    // We iterate through the provided functions of the moodule and if there are
    // remaining function calls we add them.
    std::set<std::string> provided;
    std::string nvpref = "__nv_";
    for(auto &f : *libdevice) {
      std::string name = f.getName().str();
      auto res = std::mismatch(nvpref.begin(), nvpref.end(), name.begin());
      auto oldName = name.substr(res.second - name.begin());
      if(res.first == nvpref.end() && unresolved.count(oldName) > 0) provided.insert(oldName);
    }

    for(auto &fn : provided){
      if(auto *f = m.getFunction(fn)) f->setName("__nv_" + fn); ;
    }
    /*
    for(auto & F : m){
      for(auto &BB : F){
        for(auto &I : BB){
          if(auto *CI = dyn_cast<CallInst>(&I)){
            if(Function *f = CI->getCalledFunction()){
              std::cout << f->getName().str() << "\n";
            }
          }
        }
      }
    }
    */
    auto l = Linker(m);
    l.linkInModule(std::move(libdevice), 2);
  }

  std::vector<Instruction*> tids;
  for(auto & F : m){
    for(auto &BB : F){
      for(auto &I : BB){
        if(auto *CI = dyn_cast<CallInst>(&I)){
          if(Function *f = CI->getCalledFunction()){
            if(f->getName() == "gtid"){
              tids.push_back(&I);
            }
          }
        }
      }
    }
  }

  for(auto p : tids){
    p->replaceAllUsesWith(gtid);
    p->eraseFromParent();
  }

  if(auto *f = m.getFunction("gtid")) f->eraseFromParent();

  //std::cout << "Module after llvm-gpu processing\n" << std::endl;
  //m.print(errs(), nullptr);
  //std::cout << std::endl;
  //
  // Create PTX
  auto ptxbuf = new SmallVector<char, 1<<20>();
  raw_svector_ostream ptx(*ptxbuf);

  legacy::PassManager PM;
  legacy::FunctionPassManager FPM(&m);
  PassManagerBuilder Builder;
  Builder.OptLevel = 3;
  Builder.VerifyInput = 1;
  Builder.Inliner = createFunctionInliningPass();
  Builder.populateLTOPassManager(PM);
  Builder.populateFunctionPassManager(FPM);
  Builder.populateModulePassManager(PM);

  // TODO: Hard coded machine configuration, use cuda to check
  std::string error;
  raw_os_ostream ostr(std::cout);

  //InitializeAllTargets();
  //InitializeAllTargetMCs();
  //InitializeAllAsmPrinters();

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, error);
  if(!PTXTarget){
    std::cerr << error << std:: endl;
    exit(1);
  }
  auto PTXTargetMachine =
      PTXTarget->createTargetMachine(TT.getTriple(), cudaarch,
                                     "+ptx64", TargetOptions(), Reloc::PIC_,
                                     CodeModel::Small, CodeGenOpt::Aggressive);
  m.setDataLayout(PTXTargetMachine->createDataLayout());

  bool Fail = PTXTargetMachine->addPassesToEmitFile(PM, ptx, nullptr, CodeGenFileType::CGFT_AssemblyFile, false);
  assert(!Fail && "Failed to emit PTX");

  FPM.doInitialization();
  if (PTXTargetMachine)
    PTXTargetMachine->adjustPassManager(Builder);
  for(Function &F : m)
    FPM.run(F);
  FPM.doFinalization();
  PM.run(m);
  //m.print(llvm::errs(), nullptr);
  //std::cout << ptx.str().str() << std::endl;
  return ptx.str().str();
}
