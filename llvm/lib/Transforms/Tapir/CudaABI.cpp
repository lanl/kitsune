//===- CudaABI.cpp - Lower Tapir to the Kitsune CUDA target ----*- C++ -*-===//
//
//
//                     The LLVM Compiler Infrastructure
//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//  All rights reserved.
//
// Copyright 2021, 2023. Los Alamos National Security, LLC. This
//  software was produced under U.S. Government contract
//  DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which
//  is operated by Los Alamos National Security, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT
//  NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS
//  OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.
//  If software is modified to produce derivative works, such modified
//  software should be clearly marked, so as not to confuse it with
//  the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
// Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
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
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
//  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===
//
// TODO: add printf() support.
// TODO: double precision device-side entry points.
// TODO: bring high-level design a bit closer to HipABI (as needed).
// TODO: expose enviornment variable target settings???
//

#include "llvm/Transforms/Tapir/CudaABI.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "cuabi" // support for -debug-only=cuabi

// NOTES: From the NVPTX target documentation.
//  (See: https://llvm.org/docs/NVPTXUsage.html)
//
//  The NVVMReflect pass should be executed early in the optimization
//  pipeline, immediately after the link stage. The internalize pass is also
//  recommended to remove unused math functions from the resulting PTX. For an
//  input IR module module.bc, the following compilation flow is recommended:
//
//  1. Save list of external functions in module.bc.
//  2. Link module.bc with libdevice.compute_XX.YY.bc.
//  3. Internalize all functions not in list from step #1.
//  4. Eliminate all unused internal functions.
//  5. Run NVVMReflect pass.
//  6. Run standard optimization pipeline.
//

/// ## CudaABI Transformation Command Line Options ##
///
/// All of the transformation's command line options must be
/// passed using `-mllvm` as the leading flag.  All
/// transform options have `-cuabi-` as the leading
/// string.  A summary of the options is provided below.
///
///   * `-cuabi-arch=target`: The target CUDA architecture
///     to generate code for.  This directly matches the
///     [NVPTX backend targets]
///     (https://llvm.org/docs/NVPTXUsage.html).
///
///   * `-cuabi-opt-level=[0,1,2,3]`: Set the optimization
///     level for transformation.  This corresponds directly
///     to standard optimization levels but will be applied
///     to the CUDA device-side code created by the
///     transformation.  Note that this optimization pass
///     occurs *after* an existing (in progress) optimization
///     pipeline has occurred on the original input code
///     module.  This currently defaults to level 3.  This
///     optimization level is also shared with the PTX
///     assembler (ptxas) that comes as part of the CUDA
///     distribution.
///
///   * `-cuabi-host-opt-level=[0,1,2,3]`: Set the optimization
///     level to use for the final host-side module after the CUDA
///     transformation has completed.  Even though the host code
///     has already been through a series of optimizations this
///     option enables a second series of passes over the code
///     after the transformation has completed.  At present there
///     are unlikely to be significant gains from this.  As a
///     result this defaults to level 0, which disables the
///     extra pass entirely.
///
///   * `-cuabi-prefetch`: Enable/Disable the generation of
///     data prefetch calls prior to the kernel launch. This
///     is enabled by default and typically will enable better
///     performance given the current use of managed memory
///     allocations.
///
///   * `-cuabi-max-threads-per-blk`: Set the maximum number
///     of threads that can run within a block.  This limit
///     is coordinated with the runtime's default settings
///     and will place an artificial limit on the thread count.
///     The default behavior is to match the hardware limits
///     within CUDA.
///
///   * `-cuabi-default-grainsize`: EXPERIMENTAL -- control the
///     transform's grain size.  By default this is set to 1 and
///     it is not recommended to change this unless you are
///     extremely familiar with the code generation details and
///     the implications for GPU code execution.
///
///   * `cuabi-keep-files`: The transform has the ability to
///     save the various stages of the IR during execution.
///     In addition, some files are created and removed during
///     execution.  This option will enable all these files to
///     remain (or be created) during execution.  This is
///     obviously helpful if you are debugging the transform.
///
namespace {

const std::string CUABI_PREFIX = "_cuabi";
const std::string CUABI_KERNEL_NAME_PREFIX = CUABI_PREFIX + "_kern_";

// NOTE: At this point in time we do not provide support for the older range
// of GPU architectures. We favor 64-bit and SM_60 or newer, which
// follows the trends of longer term CUDA support.  Although exposed here, we
// have not tested 32-bit host support.
#ifndef _CUDAABI_DEFAULT_TARGET_ARCH
#define _CUDAABI_DEFAULT_TARGET_ARCH "sm_80"
#endif
cl::opt<std::string>
    GPUArch("cuabi-arch", cl::init(_CUDAABI_DEFAULT_TARGET_ARCH), cl::NotHidden,
            cl::desc("Target GPU architecture for CUDA ABI transformation."
                     "(default: " _CUDAABI_DEFAULT_TARGET_ARCH));

cl::opt<unsigned>
    OptLevel("cuabi-opt-level", cl::init(3), cl::NotHidden,
             cl::desc("Specify the GPU kernel optimization level."));

cl::opt<unsigned> HostOptLevel(
    "cuabi-host-opt-level", cl::init(0), cl::NotHidden,
    cl::desc(
        "The optimization level for an experimental pass over the transformed "
        "host-side code."));

cl::opt<bool> CodeGenPrefetch("cuabi-prefetch", cl::init(true), cl::NotHidden,
                              cl::desc("Enable generation of calls to do data "
                                       "prefetching for managed memory."));

cl::opt<bool>
    UseOccupancyLaunches("cuabi-occupancy-launches", cl::init(true),
                         cl::NotHidden,
                         cl::desc("Enable generation of calls to calculate "
                                  "kernel launch parameters based "
                                  "on estimated occupancy calculations."));

const unsigned int CUDAABI_MAX_THREADS_PER_BLOCK = 1024;
const unsigned int CUDAABI_DEFAULT_MAX_THREADS_PER_BLOCK = 256;
cl::opt<unsigned int> MaxThreadsPerBlock(
    "cuabi-max-threads-per-blk",
    cl::init(CUDAABI_DEFAULT_MAX_THREADS_PER_BLOCK), cl::Hidden,
    cl::desc("Set the maximum number of threads per block generated code "
             "can support at execution.\n"));

cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"));

cl::opt<bool> KeepIntermediateFiles(
    "cuabi-keep-files", cl::init(false), cl::Hidden,
    cl::desc("Keep all the intermediate files on disk after"
             "successsful completion of the transforms. (default=false)"));

cl::opt<bool>
    Verbose("cuabi-verbose", cl::init(false), cl::NotHidden,
            cl::desc("Enable verbose mode for cuda toolchain components. "
                     "(default=false)"));

cl::opt<bool>
    EmbedPTXInFatbinaries("cuabi-embed-ptx", cl::init(false), cl::Hidden,
                          cl::desc("Embed intermediate PTX files in the "
                                   "fatbinaries used by the CUDA ABI "
                                   "transformation."));

cl::opt<unsigned>
    DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(0), cl::Hidden,
                           cl::desc("Set the runtime system's value for "
                                    "the default number of threads per block. "
                                    "(default: 0 = disabled)"));

cl::opt<unsigned>
    DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), cl::Hidden,
                         cl::desc("Hard-code the runtime system's value for "
                                  "the number of blocks per grid in kernel "
                                  "launches. (default: 0 = disabled)"));

// Take the NVIDIA CUDA 'sm_' architecture format and convert it into
// the 'compute_' form.
std::string virtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back some from the full suite of Nvidia targets
  // as we are going in assuming we will support only CUDA 11 or greater.
  // We should probably raise an error for sm_2x and sm_3x targets.
  LLVM_DEBUG(dbgs() << "cuabi: target architecture '" << Arch << "'.\n");
  std::string VirtArch = llvm::StringSwitch<std::string>(Arch)
                             .Case("sm_60", "compute_60") // Pascal
                             .Case("sm_61", "compute_61") //
                             .Case("sm_62", "compute_62") //
                             .Case("sm_70", "compute_70") // Volta
                             .Case("sm_72", "compute_72") //
                             .Case("sm_75", "compute_75") // Turing
                             .Case("sm_80", "compute_80") // Ampere
                             .Case("sm_86", "compute_86") //
                             .Case("sm_87", "compute_87") //
                             .Case("sm_90", "compute_90") // Hopper
                             .Default("unknown");
  LLVM_DEBUG(dbgs() << "cuabi: compute architecture '" << VirtArch << "'.\n");
  if (VirtArch == "unknown") {
    errs() << "cuabi: unsupported cuda architecture target '" << Arch << "'.\n";
    report_fatal_error("cuabi: fatal error -- unsupported target.");
  }
  return VirtArch;
}

std::string PTXVersionFromCudaVersion() {
#ifdef CUDATOOLKIT_VERSION
  std::string CudaVersion;
  raw_string_ostream ss(CudaVersion);
  ss << CUDATOOLKIT_VERSION_MAJOR << "." << CUDATOOLKIT_VERSION_MINOR;
#else
#pragma error("cuabi: CUDA Toolkit version info required for transform!")
#endif
  LLVM_DEBUG(dbgs() << "cuabi: cuda toolkit version: " << CudaVersion << "\n");

  std::string PTXVersionStr =
      llvm::StringSwitch<std::string>(CudaVersion)
          // TODO: These CUDA to PTX version translations will have
          // to be watched between CUDA and LLVM resources.  It is
          // not uncommon for LLVM to lag well behind CUDA PTX versions.
          // The details below are based on Cuda 12.3 and LLVM 16.
          .Case("10.0", "+ptx63")
          .Case("10.1", "+ptx64")
          .Case("10.2", "+ptx65")
          .Case("10.0", "+ptx63")
          .Case("11.0", "+ptx70")
          .Case("11.1", "+ptx71")
          .Case("11.2", "+ptx72")
          .Case("11.3", "+ptx72")
          .Case("11.4", "+ptx72")
          .Case("11.5", "+ptx72")
          .Case("11.6", "+ptx76")
          .Case("11.7", "+ptx77")
          .Case("11.8", "+ptx78")
          .Case("12.0", "+ptx83")
          .Case("12.1", "+ptx83")
          .Case("12.2", "+ptx83")
          .Case("12.3", "+ptx83")
          .Case("12.4", "+ptx83")
          .Case("12.5", "+ptx83")
          .Default("");

  if (PTXVersionStr == "") {
    errs() << "cuabi: cuda toolkit version: " << CudaVersion << "\n";
    report_fatal_error("cuabi: cuda --> ptx version mapping is out-of-date.");
  }

  LLVM_DEBUG(dbgs() << "cuabi: target ptx version: " << PTXVersionStr << "\n");
  return PTXVersionStr;
}

} // namespace

// Helper function to configure the details of our post-Tapir transformation
// passes.

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.  TODO: This is
/// a not so great naming mechanism and certainly not thread safe...
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, Module &KernelModule, const std::string &KN,
                   CudaABI *T, bool MakeUniqueName)
    : LoopOutlineProcessor(M, KernelModule), TTarget(T), KernelName(KN),
      KernelModule(KernelModule) {

  if (MakeUniqueName) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID++;
    KernelName = UN;
  }

  LLVM_DEBUG(dbgs() << "cuabi: creating loop outliner:\n"
                    << "\tkernel name: " << KernelName << "\n"
                    << "\tmodule     : " << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  // Thread index values -- equivalent to Cuda's builtins:  threadIdx.[x,y,z].
  CUThreadIdxX = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_x);
  CUThreadIdxY = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_y);
  CUThreadIdxZ = Intrinsic::getDeclaration(&KernelModule,
                                           Intrinsic::nvvm_read_ptx_sreg_tid_z);

  // Block index values -- equivalent to Cuda's builtins: blockIndx.[x,y,z].
  CUBlockIdxX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
  CUBlockIdxY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
  CUBlockIdxZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);

  // Block dimensions -- equivalent to Cuda's builtins: blockDim.[x,y,z].
  CUBlockDimX = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);
  CUBlockDimY = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_y);
  CUBlockDimZ = Intrinsic::getDeclaration(&KernelModule,
                                          Intrinsic::nvvm_read_ptx_sreg_ntid_x);

  // Grid dimensions -- equivalent to Cuda's builtins: gridDim.[x,y,z].
  CUGridDimX = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
  CUGridDimY = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
  CUGridDimZ = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);

  // NVVM-centric barrier -- equivalent to Cuda's __sync_threads().
  CUSyncThreads =
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::nvvm_barrier0);

  // Get entry points into the Cuda-centric portion of the Kitsune GPU runtime.
  KernelInstMixTy = StructType::get(Int64Ty,  // number of memory ops.
                                    Int64Ty,  // number of floating point ops.
                                    Int64Ty); // number of integer ops.
  KitCudaLaunchFn = M.getOrInsertFunction(
      "__kitcuda_launch_kernel",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrPtrTy,                    // arguments
      Int64Ty,                         // trip count
      Int32Ty,                         // threads-per-block
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque cuda stream

  KitCudaMemPrefetchFn =
      M.getOrInsertFunction("__kitcuda_mem_gpu_prefetch",
                            VoidPtrTy,  // return an opaque stream
                            VoidPtrTy,  // pointer to prefetch
                            VoidPtrTy); // opaque stream
  KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // return the device pointer for symbol.
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name
  KitCudaMemcpySymbolToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_symbol_to_device",
                            VoidTy,   // no return
                            Int32Ty,  // host pointer
                            Int64Ty,  // device pointer
                            Int64Ty); // number of bytes to copy
  LLVM_DEBUG(dbgs() << "\t\tdone.\n");
}

CudaLoop::~CudaLoop() {}

void CudaLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                    SmallVectorImpl<Value *> &HelperInputs,
                                    ValueSet &InputSet,
                                    const SmallVectorImpl<Value *> &LCArgs,
                                    const SmallVectorImpl<Value *> &LCInputs,
                                    const ValueSet &TLInputsFixed) {

  // Add the loop control inputs -- the first parameter defines
  // the extent of the index space (the number of threads to launch).
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName("runSize");
    HelperArgs.insert(EndArg);

    Value *InputVal = LCInputs[1];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The second parameter defines the start of the
  // index space.
  {
    Argument *StartArg = cast<Argument>(LCArgs[0]);
    StartArg->setName("runStart");
    HelperArgs.insert(StartArg);

    Value *InputVal = LCInputs[0];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // The third parameter defines the grain size, if it is
  // not constant.
  if (!isa<ConstantInt>(LCInputs[2])) {
    Argument *GrainsizeArg = cast<Argument>(LCArgs[2]);
    GrainsizeArg->setName("grainSize");
    HelperArgs.insert(GrainsizeArg);

    Value *InputVal = LCInputs[2];
    HelperInputs.push_back(InputVal);
    // Add loop-control input to the input set.
    InputSet.insert(InputVal);
  }

  // Add the loop-centric kernel parameters (i.e., variables/arrays
  // used in the loop body).
  for (Value *V : TLInputsFixed) {
    HelperArgs.insert(V);
    HelperInputs.push_back(V);
  }

  for (Value *V : HelperInputs) {
    OrderedInputs.push_back(V);
  }
}

unsigned CudaLoop::getIVArgIndex(const Function &F,
                                 const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned CudaLoop::getLimitArgIndex(const Function &F,
                                    const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

static std::set<GlobalValue *> &collect(Constant &c,
                                        std::set<GlobalValue *> &seen);

static std::set<GlobalValue *> &collect(BasicBlock &bb,
                                        std::set<GlobalValue *> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(Function &f,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&f);

  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalVariable &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalIFunc &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(GlobalAlias &g,
                                        std::set<GlobalValue *> &seen) {
  seen.insert(&g);
  llvm_unreachable("kitsune: GlobalAlias not yet supported");
  return seen;
}

static std::set<GlobalValue *> &collect(BlockAddress &blkaddr,
                                        std::set<GlobalValue *> &seen) {
  if (Function *f = blkaddr.getFunction())
    collect(*f, seen);
  if (BasicBlock *bb = blkaddr.getBasicBlock())
    collect(*bb, seen);
  return seen;
}

std::set<GlobalValue *> &collect(Constant &c, std::set<GlobalValue *> &seen) {
  if (GlobalValue *g = dyn_cast<GlobalValue>(&c))
    if (seen.find(g) != seen.end())
      return seen;

  if (auto *f = dyn_cast<Function>(&c))
    return collect(*f, seen);
  else if (auto *g = dyn_cast<GlobalVariable>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalAlias>(&c))
    return collect(*g, seen);
  else if (auto *g = dyn_cast<GlobalIFunc>(&c))
    return collect(*g, seen);
  else if (auto *blkaddr = dyn_cast<BlockAddress>(&c))
    return collect(*blkaddr, seen);
  else
    for (auto &op : c.operands())
      if (auto *cop = dyn_cast<Constant>(op))
        collect(*cop, seen);
  return seen;
}

void CudaLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {
  // TODO: process loop prior to outlining to do GPU/CUDA-specific things
  // like capturing global variables, etc.
  LLVM_DEBUG(dbgs() << "\tcuabi: preprocessing loop for kernel '" << KernelName
                    << "', in module '" << KernelModule.getName() << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValue's used in it will
  // need to be cloned into the KernelModule and then register with CUDA
  // in the CUDA-centric ctor.
  LLVM_DEBUG(dbgs() << "\t\t- gathering and analyzing global values...\n");
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();

  for (Loop *SL : L)
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  // Clone global variables (TODO: and aliases).
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      // TODO: Make sure this logic makes sense...
      // We don't necessarily need a GPU-side clone of a
      // global variable -- instead we need a location where
      // we can copy symbol information from the host.
      GlobalVariable *NewGV = nullptr;
      // If GV is a constant we can clone the entire
      // variable over, including the initializer
      // details, and deal with it as an internal
      // variable (i.e., no need to coordinate with
      // host).  TODO: make sure this is sound!
      if (GV->isConstant())
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ true,
            GlobalValue::InternalLinkage, GV->getInitializer(),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);
      else {
        // If GV is non-constant we will need to
        // create a device-side version that will
        // have the host-side value copied over
        // prior to launching the cooresponding
        // kernel...
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), /* isConstant*/ false,
            GlobalValue::ExternalWeakLinkage,
            (Constant *)Constant::getNullValue(GV->getValueType()),
            GV->getName() + "_devvar", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal);
        TTarget->pushGlobalVariable(GV);
      }
      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
      LLVM_DEBUG(dbgs() << "\t\t\tcreated kernel-side global variable '"
                        << NewGV->getName() << "'.\n");
    } else if (dyn_cast<GlobalAlias>(V))
      llvm_unreachable("cuabi: fatal error, GlobalAlias not implemented!");
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DeviceF = KernelModule.getFunction(F->getName());
      if (not DeviceF) {
        // LLVM_DEBUG(dbgs() << "\tanalyzing missing (device-side) function '"
        //                   << F->getName() << "'.\n");
        Function *LF = resolveLibDeviceFunction(F, false);
        if (LF && not KernelModule.getFunction(LF->getName())) {
          // LLVM_DEBUG(dbgs() << "\ttransformed to libdevice function '"
          //                   << LF->getName() << "'.\n");
          DeviceF = Function::Create(LF->getFunctionType(), F->getLinkage(),
                                     LF->getName(), KernelModule);
        } else {
          // LLVM_DEBUG(dbgs() << "\tcreated device function '" << F->getName()
          //                   << "'.\n");
          DeviceF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                     F->getName(), KernelModule);
        }
      }

      for (size_t i = 0; i < F->arg_size(); i++) {
        Argument *Arg = F->getArg(i);
        Argument *NewA = DeviceF->getArg(i);
        NewA->setName(Arg->getName());
        VMap[Arg] = NewA;
      }
      VMap[F] = DeviceF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues) {
    if (dyn_cast<GlobalIFunc>(V)) {
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size() && not F->isIntrinsic()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        CloneFunctionInto(DeviceF, F, VMap,
                          CloneFunctionChangeType::DifferentModule, Returns);

        LLVM_DEBUG(dbgs() << "cuabi: cloning device function '"
                          << DeviceF->getName() << "' into kernel module.\n");

        // GPU calls are slow, try to force inlining...
        if (OptLevel > 1 && not DeviceF->hasFnAttribute(Attribute::NoInline))
          DeviceF->addFnAttr(Attribute::AlwaysInline);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\tfinished preprocessing tapir loop.\n");
  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> PreLoopIRFile;
    SmallString<255> IRFileName("preprocess-loop.ll");
    PreLoopIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    M.print(PreLoopIRFile->os(), nullptr);
    PreLoopIRFile->keep();
  }
}

void CudaLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                  ValueToValueMapTy &VMap) {
  // addSyncToOutlineReturns(TLI, Out, VMap);

  LLVMContext &Ctx = M.getContext();
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

  TapirLoopHints Hints(TL);

  BasicBlock *Entry = cast<BasicBlock>(VMap[TL->getLoopPreheader()]);
  BasicBlock *Header = cast<BasicBlock>(VMap[TL->getHeader()]);
  BasicBlock *Exit = cast<BasicBlock>(VMap[TLI.getExitBlock()]);
  PHINode *PrimaryIV = cast<PHINode>(VMap[TLI.getPrimaryInduction().first]);
  Value *PrimaryIVInput = PrimaryIV->getIncomingValueForBlock(Entry);

  TTarget->pushSR(T->getDetach()->getSyncRegion());

  // We no longer need the cloned sync region.
  Instruction *ClonedSyncReg =
      cast<Instruction>(VMap[T->getDetach()->getSyncRegion()]);
  ClonedSyncReg->eraseFromParent();

  // Get the kernel function for this loop and clean up
  // any stray (target related) attributes that were
  // attached as part of the host-side target that
  // occurred before outlining.
  Function *KernelF = Out.Outline;
  KernelF->setName(KernelName);
  KernelF->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr("personality");
  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("target-features",
                     PTXVersionFromCudaVersion() + "," + GPUArch);
  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 6> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
  // AV.push_back(MDString::get(Ctx, "maxntidx"));
  // AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
  // 160)));
  // AV.push_back(MDString::get(Ctx, "maxnreg"));
  // AV.push_back(ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx),
  // 63)));
  Annotations->addOperand(MDNode::get(Ctx, AV));

  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grainsize arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We really only want a grainsize of 1 for now...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grainsize value, which is either constant or the third LC
    // arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> B(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic CUDA thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = B.CreateCall(CUThreadIdxX);
  Value *BlockIdx = B.CreateCall(CUBlockIdxX);
  Value *BlockDim = B.CreateCall(CUBlockDimX);
  Value *ThreadIV = B.CreateIntCast(
      B.CreateAdd(ThreadIdx, B.CreateMul(BlockIdx, BlockDim, "blk_offset"),
                  "cuthread_id"),
      PrimaryIV->getType(), false, "thread_iv");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the
  // current codegen...
  // ThreadID = B.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = B.CreateAdd(ThreadIV, Grainsize, "thread_end");
  Value *Cond = B.CreateICmpUGE(ThreadIV, End, "cond_thread_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Use the thread ID as the start iteration number for the primary IV.
  PrimaryIVInput->replaceAllUsesWith(ThreadIV);
  // TODO: ???? PrimaryIVInput->eraseFromParent();

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);

  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> PostLoopIRFile;
    SmallString<255> IRFileName("post-loop.ll");
    PostLoopIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    KernelModule.print(PostLoopIRFile->os(), nullptr);
    PostLoopIRFile->keep();
  }
}

Function *CudaLoop::resolveLibDeviceFunction(Function *Fn, bool enableFast) {
  std::unique_ptr<Module> &LDM = TTarget->getLibDeviceModule();
  const std::string NVPrefix = "__nv_";

  // Handle special cases where code generation can be a bit more
  // complex; e.g., printf().
  if (Fn->getName() == "printf" || Fn->getName() == "fprintf") {
    report_fatal_error("cuabi: printf is currently unsupported "
                       "in parallel loops... :-(\n");
  }

  std::string FnName = "";
  if (Fn->isIntrinsic()) {

    if (enableFast)
      FnName = "fast_";

    if (Fn->getName().str().compare(0, 9, "llvm.nvvm") == 0)
      return nullptr; // backend can handle these...
    else if (Fn->getName() == "llvm.cos.f32")
      FnName += "cosf";
    else if (Fn->getName() == "llvm.cos.f64")
      FnName += "cos";
    else if (Fn->getName() == "llvm.sin.f32")
      FnName += "sinf";
    else if (Fn->getName() == "llvm.sin.f64")
      FnName += "sin";
    else if (Fn->getName() == "llvm.tan.f32")
      FnName += "tanf";
    else if (Fn->getName() == "llvm.tan.f64")
      FnName += "tan";
    else if (Fn->getName() == "llvm.exp.f64")
      FnName += "exp";
    else if (Fn->getName() == "llvm.expf.f32")
      FnName += "expf";
    else {
      // errs() << "cuabi: transforming intrinsic call " << Fn->getName() <<
      // "()\n"; report_fatal_error("cuabi: no transform for llvm intrinsic!");
      return nullptr;
    }
  } else {
    if (Fn->getName() == "__sqrtf_finite") {
      FnName = "llvm.nvvm.sqrt.approx.ftz.f";
      // errs() << "\t mapping to " << FnName << "\n";
    } else if (Fn->getName() == "__powf_finite")
      FnName = "fast_powf";
    else if (Fn->getName() == "__fmodf_finite")
      FnName = "modff";
    else if (Fn->getName() == "expf") {
      if (enableFast)
        FnName = "__nv_fast_expf";
      else
        FnName = "__nv_expf";
      errs() << "call for exp: " << FnName << "().\n";
    }
  }

  FnName = Fn->getName().str();
  if (Function *KF = KernelModule.getFunction(NVPrefix + FnName)) {
    LLVM_DEBUG(dbgs() << "\t\tfound existing device function '" << KF->getName()
                      << "'.\n");
    return KF;
  }

  for (auto &DF : *LDM) {
    std::string DFName = DF.getName().str();
    auto Match =
        std::mismatch(NVPrefix.begin(), NVPrefix.end(), DFName.begin());
    auto BaseName = DFName.substr(Match.second - DFName.begin());
    if (BaseName == FnName) {
      LLVM_DEBUG(dbgs() << "Found libdevice function: '" << DF.getName()
                        << "' to resolve function '" << FnName << "'.\n");
      return &DF;
    }
  }
  return nullptr;
}

void CudaLoop::transformForPTX(Function &F) {

  // LLVM_DEBUG(dbgs() << "Transforming function '" << F.getName() << "' "
  //                   << "in preparation for PTX generation.\n");

  // PTX doesn't like .<n> global names, rename them to
  // replace the '.' with an underscore, '_'.
  for (GlobalVariable &G : KernelModule.globals()) {
    auto name = G.getName().str();
    std::replace(name.begin(), name.end(), '.', '_');
    G.setName(name);
  }

  // We now need to walk the kernel (outlined loop) and look for
  // unresolved function calls.  In particular we need to check
  // to see if they can be resolved via CUDA's libdevice.
  // LLVM_DEBUG(
  //    dbgs() << "cuabi: search for unresolved calls in outlined kernel...\n");
  std::list<CallInst *> Replaced;
  bool enableFast;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      if (FPMathOperator *FPO = dyn_cast<FPMathOperator>(CI)) {
        if (FPO->isFast()) {
          enableFast = true;
        } else {
          // LLVM_DEBUG(dbgs() << " [std/full precision]\n");
          enableFast = false;
        }
      }

      Function *CF = CI->getCalledFunction();
      if (CF->size() == 0) {
        Function *DF = resolveLibDeviceFunction(CF, enableFast);
        if (DF != nullptr) {
          CallInst *NCI = dyn_cast<CallInst>(CI->clone());
          NCI->insertBefore(CI);
          NCI->setCalledFunction(DF);
          CI->replaceAllUsesWith(NCI);
          Replaced.push_back(CI);
        }
      }
    }
  }

  for (auto CI : Replaced)
    CI->eraseFromParent();

  if (KeepIntermediateFiles) {
    std::error_code EC;
    std::unique_ptr<ToolOutputFile> KernelIRFile;
    SmallString<255> IRFileName(Twine(F.getName()).str() + "-ptx");
    sys::path::replace_extension(IRFileName, ".ll");
    KernelIRFile = std::make_unique<ToolOutputFile>(
        IRFileName, EC, sys::fs::OpenFlags::OF_None);
    KernelModule.print(KernelIRFile->os(), nullptr);
    KernelIRFile->keep();
  }
}

void CudaLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                       DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "cudaloop: processing outlined loop call...\n"
                    << "\tkernel name: " << KernelName << "\n");

  LLVMContext &Ctx = M.getContext();

  // NOTE: If we are dealing with loop nests with multiple targets
  // (in this case only a CPU-target w/ a nested GPU target is
  // supported) we can end up with multiple calls to the outlined
  // loop (which has been setup for dead code elimination) but can
  // cause invalid IR that trips us up when handling the GPU module
  // code generation. This is a challenge in the Tapir design that
  // was not geared to handle some of the nuances of GPU target
  // transformations (and code gen).  To address this, we need to
  // do some clean up to keep the IR correct (or the verifier will
  // fail on us...).  Specifically, we can no longer depend upon
  // DCE as it runs too late in the GPU transformation process...
  //
  // TODO: This code can be shared between the cuda and hip targets...
  //
  Function *TargetKF = KernelModule.getFunction(KernelName);
  std::list<Instruction *> RemoveList;
  if (TargetKF) {
    LLVM_DEBUG(dbgs() << "\t*- searching for 'dangling' outline calls...\n");
    for (Use &U : TargetKF->uses()) {
      if (auto *Inst = dyn_cast<Instruction>(U.getUser())) {
        LLVM_DEBUG(dbgs() << "\t\t- marking use for removal.\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }

  for (auto I : RemoveList)
    I->eraseFromParent();

  // Make a pass to prep for PTX code generation...
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForPTX(F);
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for PTX code gen.\n");

  // Create two builders -- one inserts code into the entry block
  // (e.g. new "up-front" allocas) and the other is for generating
  // new code into a split BB.
  Function *Parent = TOI.ReplCall->getFunction();
  BasicBlock &EntryBB = Parent->getEntryBlock();
  IRBuilder<> EntryBuilder(&EntryBB.front());

  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NewBB = RCBB->splitBasicBlock(TOI.ReplCall);
  IRBuilder<> NewBuilder(&NewBB->front());

  // TODO: There is some potential here to share this code across both
  // the hip and cuda transforms...
  LLVM_DEBUG(dbgs() << "\t*- code gen packing of " << OrderedInputs.size()
                    << " kernel args.\n");
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EntryBuilder.CreateAlloca(ArrayTy);
  AllocaInst *CudaStream = EntryBuilder.CreateAlloca(VoidPtrTy);
  EntryBuilder.CreateStore(ConstantPointerNull::get(VoidPtrTy), CudaStream);
  bool StreamAssigned = false;
  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Value *VP = EntryBuilder.CreateAlloca(V->getType());
    NewBuilder.CreateStore(V, VP);
    Value *VoidVPtr = NewBuilder.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr =
        NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    NewBuilder.CreateStore(VoidVPtr, ArgPtr);
    i++;

    if (CodeGenPrefetch && V->getType()->isPointerTy()) {
      LLVM_DEBUG(dbgs() << "\t\t- code gen prefetch for kernel arg #" << i
                        << "\n");
      Value *VoidPP = NewBuilder.CreateBitCast(V, VoidPtrTy);
      Value *SPtr = NewBuilder.CreateLoad(VoidPtrTy, CudaStream);
      Value *NewSPtr =
          NewBuilder.CreateCall(KitCudaMemPrefetchFn, {VoidPP, SPtr});
      if (not StreamAssigned) {
        NewBuilder.CreateStore(NewSPtr, CudaStream);
        StreamAssigned = true;
      }
    }
  }

  // The next step is prep for the actual kernel launch call via
  // the kitsune runtime.  We have to add some extra levels of
  // pointers to match API details, deal with some potential
  // type mismatches, build a dummy pointer for the yet-to-be-created
  // fat binary, etc...
  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr =
      NewBuilder.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);

  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, "kern.name");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We place *all* transformed tapir loops from the input module into a
  // single GPU target module.  At this point we can not create a complete
  // fat binary image.  However, we have all the important info for the
  // current loop so we use a 'dummy' (null) fat binary for code gen at
  // this point -- we'll post-process the module to clean this up after
  // we've processed all tapir loops.
  Constant *DummyFBGV =
      tapir::getOrInsertFBGlobal(M, "_cuabi.dummy_fatbin", VoidPtrTy);
  Value *DummyFBPtr = NewBuilder.CreateLoad(VoidPtrTy, DummyFBGV);

  // Deal with type mismatches for the trip count. A difference
  // introduced via the input source details and the runtime's
  // API type signature for the launch.
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Value *TripCount = OrderedInputs[0];
  Value *CastTripCount = nullptr;
  if (TripCount->getType() != Int64Ty) {
    CastTripCount = CastInst::CreateIntegerCast(TripCount, Int64Ty, false);
    NewBuilder.Insert(CastTripCount, "cast.tc");
  } else
    CastTripCount = TripCount;

  // At this point we need a threads-per-block value for the launch
  // call.  The runtime will determine this value if ThreadsPerBlock
  // is zero but it can also be overridden via kitsune's forall launch
  // attribute.  The catch here is the launch attribute's value for
  // this is flexible and be a computed expression vs. a compile-time
  // constant.  For this first step of creating the kernel launch, we
  // take the path of a runtime configuration vs. an attributed
  // launch.  This will get patched up as needed when we post-process
  // the module and replace the DummyFBPtr (as we will also need to
  // replace the kernel launch call parameter for threads-per-block if
  // an attributed expression is present).  See postProcessModule()'s
  // stage of finalizing the launch calls for details.
  TapirLoopHints Hints(TL.getLoop());
  unsigned ThreadsPerBlock = 0;
  Constant *TPBlockValue =
      ConstantInt::get(Type::getInt32Ty(Ctx), ThreadsPerBlock);

  LLVM_DEBUG(dbgs() << "\tgathering kernel instruction mix....\n");
  tapir::KernelInstMixData InstMix;
  tapir::getKernelInstructionMix(&F, InstMix);
  LLVM_DEBUG(
      dbgs() << "\tinstruction mix:\n"
             << "      memory ops      : " << InstMix.num_memory_ops << "\n"
             << "      flop count      : " << InstMix.num_flops << "\n"
             << "      integer op count: " << InstMix.num_iops << "\n\n");

  Constant *InstructionMix = ConstantStruct::get(
      KernelInstMixTy, ConstantInt::get(Int64Ty, InstMix.num_memory_ops),
      ConstantInt::get(Int64Ty, InstMix.num_flops),
      ConstantInt::get(Int64Ty, InstMix.num_iops));

  AllocaInst *AI = NewBuilder.CreateAlloca(KernelInstMixTy);
  NewBuilder.CreateStore(InstructionMix, AI);

  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch....\n");
  Value *KSPtr = NewBuilder.CreateLoad(VoidPtrTy, CudaStream);
  CallInst *LaunchStream = NewBuilder.CreateCall(
      KitCudaLaunchFn, {DummyFBPtr, KNameParam, argsPtr, CastTripCount,
                        TPBlockValue, AI, KSPtr});
  // if (not StreamAssigned)
  NewBuilder.CreateStore(LaunchStream, CudaStream);
  LLVM_DEBUG(dbgs() << "\t\t+- registering launch stream:\n"
                    << "\t\t\tcall: " << *LaunchStream << "\n"
                    << "\t\t\tstream: " << *CudaStream << "\n");
  TTarget->registerLaunchStream(LaunchStream, CudaStream);

  TOI.ReplCall->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

CudaABI::CudaABI(Module &M)
    : TapirTarget(M),
      KernelModule(Twine(CUABI_PREFIX + sys::path::filename(M.getName())).str(),
                   M.getContext()) {

  // A helping hand for invocation of the transform from a JIT-driven
  // environment where it can be a bit painful to get to
  std::optional<std::string> envTarget = sys::Process::GetEnv("CUDAABI_TARGET");
  if (envTarget) {
    LLVM_DEBUG(dbgs() << "cuabi: target set via environment '"
                      << envTarget.value() << "'.\n");
    GPUArch.setInitialValue(envTarget.value());
  }

  std::optional<std::string> ThreadsPBVar =
      sys::Process::GetEnv("CUDABI_DEFAULT_THREADS_PER_BLOCK");
  if (ThreadsPBVar) {
    MaxThreadsPerBlock = std::stoi(ThreadsPBVar.value());
    if (MaxThreadsPerBlock <= 0 ||
        MaxThreadsPerBlock > CUDAABI_MAX_THREADS_PER_BLOCK)
      report_fatal_error(
          "CUDAABI_DEFAULT_THREADS_PER_BLOCK must be > 0 && <= 1024!");
  }
  LLVM_DEBUG(dbgs() << "cuabi: setting max threads per block: "
                    << MaxThreadsPerBlock << "\n");

  LLVM_DEBUG(dbgs() << "cuabi: creating tapir target for module '"
                    << M.getName() << "' (w/ kernel module: '"
                    << KernelModule.getName() << "')\n");

  // Create a module (KernelModule) to hold all device side functions for all
  // parallel constructs in the module we are processing (M). At present a loop
  // processor will be created for each construct and is then responsible for
  // the steps required to prepare the "kernel" module (KernelModule) for code
  // generation to PTX.

  // Build the details required to have a PTX code generation path ready to go
  // at completion of the module processing; see postProcessModule() for when
  // that stage is kicked off via the Tapir layer.
  std::string ArchString = "nvptx64";
  Triple TT(ArchString, "nvidia", "cuda");

  std::string PTXVersionStr = PTXVersionFromCudaVersion();
  std::string Error;
  const Target *PTXTarget = TargetRegistry::lookupTarget("", TT, Error);
  if (!PTXTarget) {
    errs() << "Target lookup failed: " << Error << "\n";
    report_fatal_error("Unable to find registered PTX target. "
                       "Was LLVM built with the NVPTX target enabled?");
  }

  PTXTargetMachine = PTXTarget->createTargetMachine(
      TT.getTriple(), GPUArch, PTXVersionStr.c_str(), TargetOptions(),
      Reloc::PIC_, CodeModel::Large, CodeGenOptLevel::Aggressive);

  KernelModule.setTargetTriple(TT.str());
  KernelModule.setDataLayout(PTXTargetMachine->createDataLayout());

  LLVM_DEBUG(dbgs() << "\ttarget triple: " << TT.getTriple() << "\n");
}

CudaABI::~CudaABI() { LLVM_DEBUG(dbgs() << "cuabi: destroy tapir target.\n"); }

void CudaABI::pushPTXFilename(const std::string &FN) {
  ModulePTXFileList.push_back(FN);
}

void CudaABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}

std::unique_ptr<Module> &CudaABI::getLibDeviceModule() {
  if (not LibDeviceModule) {
    LLVMContext &Ctx = KernelModule.getContext();
    llvm::SMDiagnostic SMD;
    llvm::errs() << "libdevice: " << KITSUNE_CUDA_LIBDEVICE_BC << "\n";
    // KITSUNE FIXME: It might be useful during development to override the
    // libdevice.10.bc function. We could do this with a command-line argument
    // that gets passed to this transform.
    // std::optional<std::string> CudaPath = sys::Process::FindInEnvPath(
    //     "CUDA_HOME", "nvvm/libdevice/libdevice.10.bc");
    // if (!CudaPath) {
    //   CudaPath = sys::Process::FindInEnvPath("CUDA_PATH",
    //                                          "nvvm/libdevice/libdevice.10.bc");
    //   if (!CudaPath)
    //     report_fatal_error("Unable to load cuda libdevice.10.bc!");
    // }

    llvm::StringRef LibDeviceBCFile = KITSUNE_CUDA_LIBDEVICE_BC;
    LibDeviceModule = parseIRFile(LibDeviceBCFile, SMD, Ctx);
    if (not LibDeviceModule)
      report_fatal_error(llvm::StringRef("Failed to parse: ") +
                         LibDeviceBCFile);
  }

  return LibDeviceModule;
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grainsize at 1 has almost always shown to yield the
  // best results in terms of performance.  We have yet to really do
  // a detailed study of the aspects here so consider anything done
  // here as a lot of remaining work and exploration.
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize.getValue());
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // no-op...
  // The CUDA transformations split the code into device and host
  // side modules.
}

void CudaABI::addHelperAttributes(Function &F) { /* no-op */ }

bool CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) {
  return false;
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops) {
    LLVMContext &Ctx = M.getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
    Value *CudaStream = ConstantPointerNull::get(VoidPtrTy);
    FunctionCallee KitCudaSyncFn = M.getOrInsertFunction(
        "__kitcuda_sync_thread_stream", VoidTy, VoidPtrTy);

    for (Value *SR : SyncRegList) {
      for (Use &U : SR->uses()) {
        if (auto *SyncI = dyn_cast<SyncInst>(U.getUser()))
          CallInst::Create(KitCudaSyncFn, {CudaStream}, "",
                           &*SyncI->getSuccessor(0)->begin());
      }
    }
    SyncRegList.clear();
  }
}

void CudaABI::postProcessHelper(Function &F) { /* no-op */ }

void CudaABI::preProcessOutlinedTask(llvm::Function &, llvm::Instruction *,
                                     llvm::Instruction *, bool, BasicBlock *) {
  /* no-op */
}

void CudaABI::postProcessOutlinedTask(Function &F, Instruction *DetachPt,
                                      Instruction *TaskFrameCreate,
                                      bool IsSpawner, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::postProcessRootSpawner(Function &F, BasicBlock *TFEntry) {
  /* no-op */
}

void CudaABI::processSubTaskCall(TaskOutlineInfo &TOI, DominatorTree &DT) {
  /* no-op */
}

void CudaABI::preProcessRootSpawner(llvm::Function &, BasicBlock *TFEntry) {
  /* no-op */
}

CudaABIOutputFile CudaABI::assemblePTXFile(CudaABIOutputFile &PTXFile) {

  LLVM_DEBUG(dbgs() << "\t- assembling PTX file '" << PTXFile->getFilename()
                    << "'.\n");

  std::error_code EC;
  llvm::StringRef PTXASExe = KITSUNE_CUDA_PTXAS;
  SmallString<255> AsmFileName(PTXFile->getFilename());
  sys::path::replace_extension(AsmFileName, ".s");
  std::unique_ptr<ToolOutputFile> AsmFile;
  AsmFile = std::make_unique<ToolOutputFile>(AsmFileName, EC,
                                             sys::fs::OpenFlags::OF_None);

  // Build the command line for ptxas...  There are some target specific options
  // that we support to configure some specifics here.  See the 'opt' entries
  // near the top of this file.
  // These can be passed to the transform via '-mllvm <cuabi-option>'.
  opt::ArgStringList PTXASArgList;
  PTXASArgList.push_back(PTXASExe.data());

  // TODO: Do we need/want to add support for generating relocatable code?

  // --gpu-name <gpu name>: Specify name of GPU to generate code for.
  // (e.g., 'sm_70','sm_72','sm_75','sm_80','sm_86', 'sm_87')
  PTXASArgList.push_back("--gpu-name"); // target gpu architecture.
  PTXASArgList.push_back(GPUArch.c_str());

  // For now let's always warn if we spill registers...
  PTXASArgList.push_back("--warn-on-spills");
  PTXASArgList.push_back("--verbose");

  if (OptLevel > 3) {
    errs() << "cuabi: warning -- unknown optimization level.  Using level-3.\n";
    OptLevel = 3;
  }

  PTXASArgList.push_back("--opt-level");
  switch (OptLevel) {
  case 0:
    PTXASArgList.push_back("0");
    break;
  case 1:
    PTXASArgList.push_back("1");
    break;
  case 2:
    PTXASArgList.push_back("2");
    break;
  case 3:
    PTXASArgList.push_back("3");
    // PTXASArgList.push_back("--extensible-whole-program");
    break;
  default:
    llvm_unreachable_internal("unhandled/unexpected optimization level",
                              __FILE__, __LINE__);
    break;
  }

  PTXASArgList.push_back("--output-file");
  std::string SCodeFilename = AsmFile->getFilename().str();
  PTXASArgList.push_back(SCodeFilename.c_str());
  std::string ptxfile(PTXFile->getFilename().str());
  PTXASArgList.push_back(ptxfile.c_str());

  // Build argv for exec'ing ptxas...
  SmallVector<const char *, 128> PTXASArgv;
  PTXASArgv.append(PTXASArgList.begin(), PTXASArgList.end());
  PTXASArgv.push_back(nullptr);

  auto PTXASArgs = toStringRefArray(PTXASArgv.data());
  LLVM_DEBUG(dbgs() << "\t- ptxas command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  // Finally we are ready to run ptxas...
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(PTXASExe, PTXASArgs, std::nullopt, {},
                                     0, /* secs to wait -- 0 --> unlimited */
                                     0, /* memory limit -- 0 --> unlimited */
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("fatal error: 'ptxas' execution failed!");

  if (ExecStat != 0)
    // 'ptxas' ran but returned an error state.
    report_fatal_error("fatal error: 'ptxas' failure: " + StringRef(ErrMsg));

  // TODO: Not sure we need to force 'keep' here as we return
  // the output file but will keep it here for now just to play it
  // safe.
  AsmFile->keep();
  return AsmFile;
}

// We can't create a correct launch sequence until all the kernels within a
// (LLVM) module are generated.  When post-processing the module we create the
// fat binary and then to revisit the kernel launch calls we created at the loop
// level and replace the fat binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the host to the
// device prior to kernel launches.  This requires digging some additonal
// details out of the fat binary (CUDA module).
void CudaABI::finalizeLaunchCalls(Module &M, GlobalVariable *Fatbin) {

  LLVM_DEBUG(dbgs() << "\t- finalizing kernel launch calls...\n");

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  // TODO: Move these callees to the constructor -- no need
  // to build and destory each time... Perhaps speed us up a
  // tidbit...
  FunctionCallee KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitcuda_get_global_symbol",
                            Int64Ty,    // device pointer
                            VoidPtrTy,  // fat binary
                            CharPtrTy); // symbol name

  FunctionCallee KitCudaMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitcuda_memcpy_sym_to_device",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            Int64Ty,   // device ptr
                            Int64Ty);  // num bytes

  // Search for kernel launch calls that we built prior to the creation
  // of the fat binary -- which we now have.  Replace the first parameter
  // in each call (which is currently null) with the fat binary pointer.
  LLVM_DEBUG(dbgs() << "\t\tsearching...\n");
  auto &FnList = M.getFunctionList();

  // If we have encountered any attributed launches (forall loops) we
  // will have a call to a dummy runtime entry point
  // (_kitrt_dummy_threads_per_blk) that we use to keep the
  // threads-per-block expression from getting DCE'ed.  When lowered
  // from clang, this call was inserted and we need to use its sole
  // parameter in place of the threads-per-block parameter in the
  // kernel launch call.  These dummy calls should be paired with
  // launch calls, so as we look for a launch we *should* first
  // encounter the threads-per-block call to pair with it.  We keep
  // track of the dummy calls so they can removed at the end of the
  // launch finalization.
  CallInst *ThreadsPerBlockCI = nullptr;
  std::list<CallInst *> DummyCIList;
  CallInst *SavedLaunchCI = nullptr;

  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {

            if (CFn->getName().starts_with("__kitrt_dummy_threads_per_blk")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* discovered a threads-per-block "
                                   "placeholder call.\n");
              assert(ThreadsPerBlockCI == nullptr && "expected null pointer!");
              ThreadsPerBlockCI = CI;
            } else if (CFn->getName().starts_with(
                           "__kitcuda_sync_thread_stream")) {
              LLVM_DEBUG(dbgs()
                         << "\t\t\t* patching sync call: " << *CI << "\n");
              if (SavedLaunchCI != nullptr) {
                AllocaInst *StreamAI = getLaunchStream(SavedLaunchCI);
                assert(StreamAI != nullptr && "unexpected null launch stream!");
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t* stream alloca: " << *StreamAI << "\n");
                IRBuilder<> SyncBuilder(CI);
                Value *CudaStream =
                    SyncBuilder.CreateLoad(VoidPtrTy, StreamAI, "custreamh");
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t* cuda stream: " << *CudaStream << "\n");
                CI->setArgOperand(0, CudaStream);
                SavedLaunchCI = nullptr;
                LLVM_DEBUG(dbgs() << "\t\t\t* patched call: " << *CI << "\n");
              } else {
                LLVM_DEBUG(dbgs()
                           << "\t\t\t- note, matching launch not found!\n");
              }
            } else if (CFn->getName().starts_with("__kitcuda_launch_kernel")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* patching launch: " << *CI << "\n");
              SavedLaunchCI = CI;
              Value *CFatbin;
              CFatbin = CastInst::CreateBitOrPointerCast(Fatbin, VoidPtrTy,
                                                         "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);

              if (ThreadsPerBlockCI) {
                LLVM_DEBUG(dbgs()
                           << "\t\t\t\t replacing thread-per-block arg.\n");
                Value *TBPParam = ThreadsPerBlockCI->getArgOperand(0);
                assert(TBPParam && "unexpected null arg operand!");
                CI->setArgOperand(4, TBPParam);
                DummyCIList.push_back(ThreadsPerBlockCI);
                ThreadsPerBlockCI = nullptr;
              }

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              assert(NI && "unexpected null instruction!");

              // We need to explicitly add code to sync up host- and
              // device-side global values prior to launching kernels.
              // We only have a complete awareness of this now so insert
              // the supporting runtime calls.
              //
              // TODO: This is overdone -- we copy *all* globals and not
              // just those that the kernel we're launching is using.
              //
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + "_devvar";
                LLVM_DEBUG(dbgs() << "\t\t\t  processing global: '"
                                  << HostGV->getName() << "'\n");
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitCudaGetGlobalSymbolFn,
                                     {CFatbin, SymName}, ".cuabi_devptr", NI);
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitCudaMemcpyToDeviceFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    NI);
              }
            }
          }
        }
      }
    }
  }

  for (auto I : DummyCIList) {
    LLVM_DEBUG(dbgs() << "\t\t\t erasing dummy threads-per-block call.\n");
    I->eraseFromParent();
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable("_cuabi.dummy_fatbin", true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(Fatbin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\tcleaning up dummy fatbin global.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    // FIXME: If we haven't found the proxy for a fat binary the odds are we
    // have not found any parallel loops in the code...  Technically, this
    // should not be seen as a compiler error...
    report_fatal_error("internal error! unable to find proxy fatbin pointer!");
  }
}

CudaABIOutputFile CudaABI::createFatbinaryFile(CudaABIOutputFile &AsmFile) {
  std::error_code EC;
  SmallString<255> FatbinFilename(AsmFile->getFilename());
  sys::path::replace_extension(FatbinFilename, ".cufatbin");
  CudaABIOutputFile FatbinFile;
  FatbinFile = std::make_unique<ToolOutputFile>(FatbinFilename, EC,
                                                sys::fs::OpenFlags::OF_None);

  LLVM_DEBUG(dbgs() << "\t- generatng fatbinary image file '"
                    << FatbinFile->getFilename() << "'.\n");

  // // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  // auto FatbinaryExe = sys::findProgramByName("fatbinary");
  // if ((EC = FatbinaryExe.getError()))
  //   report_fatal_error("'fatbinary' not found. "
  //                      "Is a CUDA installation in your path?");

  llvm::StringRef FatbinaryExe = KITSUNE_CUDA_FATBINARY;
  opt::ArgStringList FatbinaryArgList;
  FatbinaryArgList.push_back(FatbinaryExe.data());
  FatbinaryArgList.push_back("--64");
  FatbinaryArgList.push_back("--create");
  FatbinaryArgList.push_back(FatbinFilename.c_str());

  std::string FatbinaryImgArgs =
      "--image=profile=" + GPUArch + ",file=" + AsmFile->getFilename().str();
  FatbinaryArgList.push_back(FatbinaryImgArgs.c_str());

  std::list<std::string> PTXFilesArgList;
  if (EmbedPTXInFatbinaries) {
    std::string VArchStr = virtualArchForCudaArch(GPUArch);
    if (VArchStr == "unknown")
      report_fatal_error("cuabi: no virtual target for given gpuarch '" +
                         StringRef(GPUArch) + "'!");

    std::string PTXFixedArgStr = "--image=profile=" + VArchStr + ",file=";
    for (auto &PTXFile : ModulePTXFileList) {
      std::string arg = PTXFixedArgStr + PTXFile;
      std::list<std::string>::const_iterator it;
      it = PTXFilesArgList.emplace(PTXFilesArgList.end(), std::move(arg));
      FatbinaryArgList.push_back(it->c_str());
    }
  }

  FatbinaryArgList.push_back(nullptr);

  SmallVector<const char *, 128> FatbinaryArgv;
  FatbinaryArgv.append(FatbinaryArgList.begin(), FatbinaryArgList.end());
  auto FatbinaryArgs = toStringRefArray(FatbinaryArgv.data());

  LLVM_DEBUG(dbgs() << "\tfatbinary command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : FatbinaryArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat =
      sys::ExecuteAndWait(FatbinaryExe, FatbinaryArgs, std::nullopt, {},
                          0, /* secs to wait -- 0 --> unlimited */
                          0, /* memory limit -- 0 --> unlimited */
                          &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("unable to execute 'fatbinary'.");

  if (ExecStat != 0)
    // 'fatbinary' ran but returned an error state.
    // TODO: Need to check what sort of actual state 'fatbinary' returns to the
    // environment -- currently assuming it matches standard practices...
    report_fatal_error("'fatbinary' error:" + StringRef(ErrMsg));

  if (EmbedPTXInFatbinaries) {
    std::list<std::string>::iterator it = PTXFilesArgList.begin();
    while (it != PTXFilesArgList.end()) {
      PTXFilesArgList.erase(it++);
    }
  }

  // TODO: Not sure we need to force 'keep' here as we return the output file
  // but will keep it here for now just to play it safe.
  FatbinFile->keep();
  return FatbinFile;
}

GlobalVariable *CudaABI::embedFatbinary(CudaABIOutputFile &FatbinaryFile) {

  LLVM_DEBUG(dbgs() << "\t- code gen for embedded fat binary image...\n");

  // Allocate a buffer to store the fat binary image in.  We
  // will then codegen it into the host-side module.
  std::unique_ptr<llvm::MemoryBuffer> Fatbinary = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> FBBufferOrErr =
      MemoryBuffer::getFile(FatbinaryFile->getFilename());
  if (std::error_code EC = FBBufferOrErr.getError()) {
    report_fatal_error("cuabi: failed to load fat binary image: " +
                       StringRef(EC.message()));
  }
  Fatbinary = std::move(FBBufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\t\treading fat binary.  size = "
                    << Fatbinary->getBufferSize() << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FatbinArray = ConstantDataArray::getRaw(
      StringRef(Fatbinary->getBufferStart(), Fatbinary->getBufferSize()),
      Fatbinary->getBufferSize(), Int8Ty);

  LLVM_DEBUG(dbgs() << "\t\tcreating associated global 'fatbin' variable.\n");

  // Create a global variable to hold the fatbinary image.
  GlobalVariable *FatbinaryGV;
  FatbinaryGV = new GlobalVariable(M, FatbinArray->getType(), true,
                                   GlobalValue::PrivateLinkage, FatbinArray,
                                   "_cuabi_fatbin_ptr");
  return FatbinaryGV;
}

void CudaABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = Int64Ty;
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  FunctionCallee RegisterVarFn = M.getOrInsertFunction(
      "__cudaRegisterVar", VoidTy, VoidPtrPtrTy, CharPtrTy, CharPtrTy,
      CharPtrTy, IntTy, VarSizeTy, IntTy, IntTy);
  for (auto &HostGV : GlobalVars) {
    uint64_t VarSize = DL.getTypeAllocSize(HostGV->getType());
    Value *VarName = tapir::createConstantStr(HostGV->getName().str(), M);
    std::string DevVarName = HostGV->getName().str() + "_devvar";
    Value *DevName = tapir::createConstantStr(DevVarName, M, DevVarName);
    llvm::Value *Args[] = {
        Handle,
        B.CreateBitCast(HostGV, VoidPtrTy),
        VarName,
        DevName,
        ConstantInt::get(IntTy, 0), // HostGV->isExternalLinkage()),
        ConstantInt::get(VarSizeTy, VarSize),
        ConstantInt::get(IntTy, HostGV->isConstant()),
        ConstantInt::get(IntTy, 0)};

    LLVM_DEBUG(dbgs() << "\t\t\thost global '" << HostGV->getName().str()
                      << "' to device '" << DevVarName << "'.\n");
    B.CreateCall(RegisterVarFn, Args);
  }
}

Function *CudaABI::createCtor(GlobalVariable *Fatbinary,
                              GlobalVariable *Wrapper) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *BoolTy = Type::getInt8Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".ctor." + KernelModule.getName(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  FunctionCallee KitRTSetDefaultMaxTheadsPerBlockFn = M.getOrInsertFunction(
      "__kitcuda_set_default_threads_per_blk", VoidTy, IntTy);
  CtorBuilder.CreateCall(KitRTSetDefaultMaxTheadsPerBlockFn,
                         {ConstantInt::get(IntTy, MaxThreadsPerBlock)});

  FunctionCallee KitCudaInitFn =
      M.getOrInsertFunction("__kitcuda_initialize", VoidTy);
  CtorBuilder.CreateCall(KitCudaInitFn, {});

  FunctionCallee KitCudaOccLaunchFn =
      M.getOrInsertFunction("__kitcuda_use_occupancy_launch", VoidTy, BoolTy);
  Value *EnableOccLaunches;
  if (UseOccupancyLaunches)
    EnableOccLaunches = ConstantInt::get(BoolTy, 1);
  else
    EnableOccLaunches = ConstantInt::get(BoolTy, 0);
  CtorBuilder.CreateCall(KitCudaOccLaunchFn, {EnableOccLaunches});

  // TODO: The parameters to the CUDA registration calls can be opaque about
  // specifics (e.g., types).  Once we sort out some details we should clean
  // this up.

  // The general layout of the calls for fat binary registration
  // look something like this:
  //
  // void** __cudaRegisterFatBinary(void *fatCubin);
  //
  // void __cudaRegisterVar(void **fatCubinHandle,
  //                        char  *hostVar,
  //                        char  *deviceAddress,
  //                        const char  *deviceName,
  //                        int    ext,
  //                        size_t size,
  //                        int    constant,
  //                        int    global);
  //
  // void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
  //
  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__cudaRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, // cubin handle.
                                              VoidPtrTy, // fat bin device txt.
                                              false));
  CallInst *RegFatbin = CtorBuilder.CreateCall(
      RegisterFatbinaryFn, CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));

  GlobalVariable *Handle = new GlobalVariable(
      M, VoidPtrPtrTy, false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), CUABI_PREFIX + ".fbhand");
  Handle->setAlignment(Align(DL.getPointerABIAlignment(0)));
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerABIAlignment(0));
  Handle->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  Value *HandlePtr = CtorBuilder.CreateLoad(VoidPtrPtrTy, Handle,
                                            CUABI_PREFIX + ".fbhand.ptr");

  // TODO: It is not 100% clear what calls we actually need to make here for
  // kernel, variable, etc. registration with CUDA.  Clang makes these calls but
  // we are targeting CUDA driver API entry points via the Kitsune runtime
  // library so these calls are potentially unneeded...
  if (!GlobalVars.empty()) {
    LLVM_DEBUG(dbgs() << "\t\tbinding host and device global variables...\n");
    bindGlobalVariables(HandlePtr, CtorBuilder);
  }

  // Wrap up fatbinary registration steps...
  FunctionCallee EndFBRegistrationFn =
      M.getOrInsertFunction("__cudaRegisterFatBinaryEnd",
                            FunctionType::get(VoidTy,
                                              VoidPtrPtrTy, // cubin handle.
                                              false));
  CtorBuilder.CreateCall(EndFBRegistrationFn, RegFatbin);

  // Now add a Dtor to help us clean up at program exit...
  if (Function *CleanupFn = createDtor(Handle)) {
    // Hook into 'atexit()'...
    FunctionType *AtExitFnTy =
        FunctionType::get(IntTy, CleanupFn->getType(), false);
    FunctionCallee AtExitFn =
        M.getOrInsertFunction("atexit", AtExitFnTy, AttributeList());
    CtorBuilder.CreateCall(AtExitFn, CleanupFn);
  }

  CtorBuilder.CreateRetVoid();
  return CtorFn;
}

Function *CudaABI::createDtor(GlobalVariable *FBHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__cudaUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".dtor", &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, FBHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kitcuda_destroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});

  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void CudaABI::registerFatbinary(GlobalVariable *Fatbinary) {

  LLVM_DEBUG(dbgs() << "\t- registering fat binary...\n");

  // Registering the fat binary image (and all the associated components) is an
  // undocumented portion of the CUDA API. One place to peek for some details
  // hides in the cuda header files; specifially fatbinary_section.h. This
  // shows the following struct that we need to have in the host side code.
  //
  //    struct fatbinC_Wrapper_t {
  //      int magic;
  //      int version;
  //      const unsigned long long *data;
  //      void *filename_or_fatbins;
  //    };
  //
  // * Per the header, the magic number is 0x466243B1.
  // * FATBINC_VERSION is 1 and FATBINC_LINK_VERSION is 2.
  //   (more below)
  // * Then section and segments are needed that contains
  //   the "fatbin control structure".  This loosely looks
  //   like:
  //
  //        Control section name: ".nvFatBinSegment"
  //        Fatbinary section name: ".nv_fatbin"
  //        Pre-linked relocatable section: "__nv_relfatbin"
  //
  // * The last struct member varies between versions.  In the
  //   case of version 1 it can be a offline filename and for
  //   version 2 it is an array of pre-linked fatbins.
  //
  const int FATBINARY_MAGIC_ID = 0x466243B1;
  const int FATBINARY_VERSION = 1;
  const char *FATBIN_CONTROL_SECTION_NAME = ".nvFatBinSegment";
  const char *FATBIN_DATA_SECTION_NAME = ".nv_fatbin";

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  Type *FatbinStrTy = Fatbinary->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(FatbinStrTy), 0),
                       ConstantInt::get(DL.getIndexType(FatbinStrTy), 0)};

  Fatbinary->setSection(FATBIN_DATA_SECTION_NAME);
  Constant *FatbinaryPtr = ConstantExpr::getGetElementPtr(
      Fatbinary->getValueType(), Fatbinary, Zeros);

  // Wrap the fatbinary in struct that the CUDA runtime and tools expect
  // to exist in final objects/executables.
  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // data
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS = ConstantStruct::get(
      WrapperTy, ConstantInt::get(IntTy, FATBINARY_MAGIC_ID),
      ConstantInt::get(IntTy, FATBINARY_VERSION), FatbinaryPtr,
      ConstantPointerNull::get(VoidPtrTy));

  GlobalVariable *Wrapper =
      new GlobalVariable(M, WrapperTy, true, GlobalValue::InternalLinkage,
                         WrapperS, "_cuabi_wrapper");
  Wrapper->setSection(FATBIN_CONTROL_SECTION_NAME);
  Wrapper->setAlignment(Align(DL.getPrefTypeAlign(Wrapper->getType())));

  // The rest of the registration details are tucked into a constructor
  // entry...
  LLVM_DEBUG(dbgs() << "\t\tcode gen new global ctor entry...\n");
  Function *CtorFn = createCtor(Fatbinary, Wrapper);
  if (CtorFn) {
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M, ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  }
}

CudaABIOutputFile CudaABI::generatePTX() {

  LLVM_DEBUG(dbgs() << "\t- generating PTX...\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                 ".post.preopt.ll"));

  // Take the intermediate form code in the kernel module and
  // generate a PTX file.  The PTX file will be named the same as
  // the original input source module (M) with the extension changed
  // to PTX.
  std::string ModelPTXFileName =
      std::string(CUABI_PREFIX) + "%%-%%-%%_" + KernelModule.getName().str();
  SmallString<1024> PTXFileName;
  sys::fs::createUniquePath(ModelPTXFileName.c_str(), PTXFileName, true);
  sys::path::replace_extension(PTXFileName, ".ptx");

  std::error_code EC;
  std::unique_ptr<ToolOutputFile> PTXFile;
  PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                             sys::fs::OpenFlags::OF_None);
  PTXFile->keep();

  KernelModule.addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", true);

  if (OptLevel > 0) {
    if (OptLevel > 3)
      OptLevel = 3;
    LLVM_DEBUG(dbgs() << "\t- running kernel module optimization passes...\n");
    PipelineTuningOptions pto;
    pto.LoopVectorization = OptLevel > 2;
    pto.SLPVectorization = OptLevel > 2;
    pto.LoopUnrolling = OptLevel > 2;
    pto.LoopInterleaving = OptLevel > 2;
    pto.LoopStripmine = OptLevel > 2;
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[OptLevel];

    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassBuilder pb(PTXTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    PTXTargetMachine->registerPassBuilderCallbacks(pb, false);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    LLVM_DEBUG(dbgs() << "\t\t* module: " << KernelModule.getName() << "\n");
    mpm.run(KernelModule, mam);
    LLVM_DEBUG(dbgs() << "\t\tpasses complete.\n");
    LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                   ".postopt.LTO.ll"));
  }

  // Setup the passes and request that the output goes to the
  // specified PTX file.
  LLVM_DEBUG(dbgs() << "\t- PTX file: '" << PTXFileName << "'.\n");
  legacy::PassManager PassMgr;
  if (PTXTargetMachine->addPassesToEmitFile(PassMgr, PTXFile->os(), nullptr,
                                            CodeGenFileType::AssemblyFile,
                                            false))
    report_fatal_error("Cuda ABI transform -- PTX generation failed!");
  PassMgr.run(KernelModule);
  LLVM_DEBUG(dbgs() << "\tkernel optimizations and code gen complete.\n\n");
  LLVM_DEBUG(dbgs() << "\t\tPTX file: " << PTXFile->getFilename() << "\n");
  return PTXFile;
}

void CudaABI::postProcessModule() {
  // At this point, all tapir constructs in the input module (M) have been
  // transformed (i.e., outlined) into the kernel module. We can now wrap up
  // module-wide changes for both modules and generate a GPU binary.
  // NOTE: postProcessModule() will not be called in cases where parallelism
  // was not discovered during loop spawning.
  LLVM_DEBUG(dbgs() << "\n\n"
                    << "cuabi: postprocessing the kernel '"
                    << KernelModule.getName() << "' and input '" << M.getName()
                    << "' modules.\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                 ".post.unoptimized"));
  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".outline-debug"));

  auto L = Linker(KernelModule);
  if (LibDeviceModule) {
    LLVM_DEBUG(dbgs() << "\t- linking in cuda libdevice into kernel module.\n");
    L.linkInModule(std::move(LibDeviceModule), Linker::LinkOnlyNeeded);
  }

  CudaABIOutputFile PTXFile = generatePTX();
  CudaABIOutputFile AsmFile = assemblePTXFile(PTXFile);
  CudaABIOutputFile FatbinFile = createFatbinaryFile(AsmFile);
  GlobalVariable *Fatbinary = embedFatbinary(FatbinFile);

  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".post-fatbin"));

  finalizeLaunchCalls(M, Fatbinary);

  LLVM_DEBUG(saveModuleToFile(&M, M.getName().str() + ".post-finalize-launch"));

  registerFatbinary(Fatbinary);
  if (HostOptLevel > 0) {
    if (HostOptLevel > 3)
      HostOptLevel = 3;
    PipelineTuningOptions pto;
    pto.LoopVectorization = HostOptLevel > 2;
    pto.SLPVectorization = HostOptLevel > 2;
    pto.LoopUnrolling = true;
    pto.LoopInterleaving = true;
    pto.LoopStripmine = false;

    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[HostOptLevel];
    PassBuilder pb(PTXTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    PTXTargetMachine->registerPassBuilderCallbacks(pb, false);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    mpm.run(M, mam);
    LLVM_DEBUG(dbgs() << "\tpasses complete.\n");
  }

  if (not KeepIntermediateFiles) {
    sys::fs::remove(PTXFile->getFilename());
    sys::fs::remove(AsmFile->getFilename());
    sys::fs::remove(FatbinFile->getFilename());
  }
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  // Create a CUDA loop outline processor for transforming parallel tapir loop
  // constructs into suitable GPU device code.  We hand the outliner the kernel
  // module (KernelModule) as the destination for all generated (device-side)
  // code.

  std::string ModuleName = sys::path::filename(M.getName()).str();
  Loop *TheLoop = TL->getLoop();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KernelName = Fn->getName().str();

  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module use a line number
    // based naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KernelName =
        CUABI_KERNEL_NAME_PREFIX + ModuleName + "_" + Twine(LineNumber).str();
  } else {
    // SmallString<255> ModName(Twine(ModuleName).str());
    // sys::path::replace_extension(ModName, "");
    // KernelName = CUABI_PREFIX + ModName.c_str();
    //  In the non-debug mode we use a consecutive numbering scheme for our
    //  kernel names (this is currently handled via the 'make unique'
    //  parameter).
    KernelName = CUABI_KERNEL_NAME_PREFIX + KernelName;
  }

  CudaLoop *Outliner = new CudaLoop(M, KernelModule, KernelName, this);
  return Outliner;
}
