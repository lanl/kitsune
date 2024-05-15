//===- HipABI.cpp - Tapir to Kitsune runtime HIP target ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
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
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for HIP to produce a fully compiled fat binary inserted into
// the input LLVM Module.
//
// TODO: device-side calls to cover feature set and double-precision support
// TODO: add printf() support.
// TODO: revisit/refactor 'mutate' type uses.
// TODO: -- math options for:
//             - DAZ [on|off],
//             - unsafe math [on|off],
//             - sqrt rounding [on|off],
//             - etc.
// TODO: more robust target architecture processing
// TODO: better optimization and code gen.
//
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/HipABI.h"
#include "kitsune/Config/config.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Tapir/TapirLoopInfo.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi" // support for -debug-only=hipabi

static const std::string HIPABI_PREFIX = "__hipabi";
static const std::string HIPABI_KERNEL_NAME_PREFIX = HIPABI_PREFIX + ".kern.";

// Transformation-specific command line arguments.
//
//  Usage: -mllvm -hipabi-[option...]
//
#ifndef HIPABI_DEFAULT_ARCH
#define HIPABI_DEFAULT_ARCH "gfx90a"
#endif

/// ## HIPABI Transformation Command Line Options ##
///
/// All of the transformation's command line options must be
/// passed using the the `-mllvm` as the leading flag.  All
/// transform options should have `-hipabi-` as the leading
/// string.  A summary of these options is provided below.
///
///   * `-hipabi-arch=target`: The target AMDGPU architecture
///     to generate code for.  This directly matches the
///     [AMDGPU processor
///     targets](https://llvm.org/docs/AMDGPUUsage.html#id112).
///
///   * `-hipabi-opt-level=[0,1,2,3]`: Set the optimization
///     level for transformation.  This corresponds directly
///     to standard optimization levels but will be applied
///     to the HIP-/GPU-centric code created by the
///     transformation.  Note that this transformation
///     occurs *after* an existing (in progress) optimization
///     pipeline has occurred on the original input code
///     module.  This currently defaults to level 2.
///
///   * `-hipabi-host-opt-level=[0,1,2,3]`: Set the optimization
///     level to use for the final host-side module after the HIP
///     transformation has completed.  Even though the host code
///     has already been through a series of optimizations this
///     option enables a second series of passes over the code
///     after the transformation has completed.  At present there
///     are unlikely to be significant gains from this.  As a
///     result this defaults to level 0, which disables the
///     extra pass entirely.
///
///   * `-hipabi-prefetch`: Enable/Disable the generation of
///     data prefetch calls prior to the kernel launch. This
///     is enabled by default and typically will enable better
///     performance given the current use of managed memory
///     allocations (although HIP currently has some poor
///     performance with managed memory in general).
///
///   * `-hipabi-max-threads-per-blk`: Set the maximum number
///     of threads that can run within a block (a la CUDA).
///     Note that this value has to be coordinated with the
///     runtime's default settings as a mismatch can result
///     in a kernel that fails to launch (currently a very
///     opaque error message will be reported by HIP if this
///     occurs).  This is just the maximum allowed value, not
///
///   * `-hipabi-xnack`: Enable XNACK code generation. This
///     is off by default.  XNACK is tricky and unclear in
///     terms of advantages it can (might?) provide without
///     digging into low-level system configuration details.
///     At present we've found little advantage to enabling
///     it (but that might change as things mature w/ HIP and
///     ROCm). Default value is disabled/false.
///
///   * `-hipabi-use-sramecc`: Enable SRAMECC support in the
///     generated code.  Default value is disabled/false.
///
///   * `-hipabi-wavefront64`: Enable/Disable the use of 64
///     wavefronts.  Default is enabled.
///
///   * `-hipabi-default-grainsize`: EXPERIMENTAL -- control the
///     transform's grainsize.  By default this is set to 1 and
///     it is not recommended to change this unless you are
///     extremely familiar with the code generation details and
///     the implications for GPU code execution.
///
///   * `-hipabi-rocm-abi`: The ROCm ABI version to target.  This
///     defaults to version 4 and it not suggested that it be
///     changed unless you are experimenting with details of
///     ROCm and HIP.
///
///   * `hipabi-keep-files`: The transform has the ability to
///     save the various stages of the IR during execution.
///     In addition, some files are created and removed during
///     execution.  This option will enable all these files to
///     remain (or be created) during execution.  This is
///     obviously helpful if you are debugging the transform.
///
namespace {

cl::opt<std::string> GPUArch(
    "hipabi-arch", cl::init(HIPABI_DEFAULT_ARCH), cl::NotHidden,
    cl::desc("Target AMD GPU architecture. (default: #HIPABI_DEFAULT_ARCH)"));

cl::opt<unsigned>
    OptLevel("hipabi-opt-level", cl::init(2), cl::NotHidden,
             cl::desc("The Tapir HIP target transform optimization level"));

cl::opt<unsigned> HostOptLevel( // EXPERIMENTAL
    "hipabi-host-opt-level", cl::init(0), cl::NotHidden,
    cl::desc("The optimization level for a final pass over the transformed "
             "host-side code."));

cl::opt<bool> CodeGenPrefetch("hipabi-prefetch", cl::init(true), cl::Hidden,
                              cl::desc("Enable generation of calls to do data "
                                       "prefetching for managed memory."));

const unsigned int AMDGPU_MAX_THREADS_PER_BLOCK = 1024;
const unsigned int HIPABI_DEFAULT_MAX_THREADS_PER_BLOCK =
    AMDGPU_MAX_THREADS_PER_BLOCK;
cl::opt<unsigned int> MaxThreadsPerBlock(
    "hipabi-max-threads-per-blk",
    cl::init(HIPABI_DEFAULT_MAX_THREADS_PER_BLOCK), cl::Hidden,
    cl::desc("Set the maximum number of threads per block generated code "
             "can support at execution.\n"));

enum ROCmABIVersion {
  ROCm_ABI_V4, // default
  ROCm_ABI_V5, // new
};

cl::opt<ROCmABIVersion> ROCmABITarget(
    "hipabi-rocm-abi", cl::init(ROCm_ABI_V4), cl::Hidden,
    cl::desc("Select the targeted ROCm ABI version."),
    cl::values(clEnumValN(ROCm_ABI_V4, "v4", "Target ROCm version 4 ABI."),
               clEnumValN(ROCm_ABI_V5, "v5", "Target ROCm v. 5 ABI.")));

cl::opt<bool> Use64ElementWavefront(
    "hipabi-wavefront64", cl::init(true), cl::Hidden,
    cl::desc("Use 64 element wavefronts. (default: enabled)"));

cl::opt<bool> EnableXnack("hipabi-xnack", cl::init(false), cl::NotHidden,
                          cl::desc("Enable/disable xnack. (default: false)"));

cl::opt<bool>
    EnableSRAMECC("hipabi-sramecc", cl::init(true), cl::NotHidden,
                  cl::desc("Enable/disable sramecc.(default: false)"));

cl::opt<unsigned> DefaultGrainSize(
    "hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grain size used by the transform "
             "when analysis fails to determine one. (default=1)"));

cl::opt<bool>
    KeepIntermediateFiles("hipabi-keep-files", cl::init(false), cl::Hidden,
                          cl::desc("Keep/create intermediate files during the "
                                   "various stages of the transform."));

// LLVM variable name for the embedded fat binary image.
const char *HIPAPI_DUMMY_FATBIN_NAME = "_hipabi.dummy_fatbin";

// --- Address spaces.
//
// TODO: Need to work on making sure we understand the nuances
// here for address space selection.  In some cases, wrong address
// spaces seem to cause crashes, in others they are performance
// optimizations, and sometimes they almost seem to be no-ops...
// Some of the AMD documentation details seem incomplete.
//
//   See: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces-table.
//
const unsigned HIPABI_GLOBAL_ADDR_SPACE = 1; // global virtual addresses.
const unsigned HIPABI_CONST_ADDR_SPACE = 4;  // indicates that the data will not
                                             // change during the execution of
                                             // the kernel.
const unsigned HIPABI_ALLOCA_ADDR_SPACE = 5; // "private" (scratch, 32-bit)

// --- Some utility functions for helping during the transformation.

/// @brief Is the given function an AMD GPU kernel.
/// @param F -- the Function to inspect.
/// @return true if the function is a kernel, false otherwise.
bool isAMDKernelFunction(Function *Fn) {
  return Fn->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

/// @brief Make calls within a function match the function's calling conv.
/// @param F -- The function to walk looking for calls.
/// @return void (calls within F will be modified)
void transformCallingConv(Function &F) {
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CI->getCallingConv() != CF->getCallingConv()) {
        LLVM_DEBUG(dbgs() << "\t\t\t-* updated calling convention to "
                          << "match '" << CF->getName() << "()'.\n");
        CI->setCallingConv(CF->getCallingConv());
      }
    }
  }
}

/// @brief Look for the given function in the device-side modules.
/// @param Fn - the function to resolve.
/// @param DevMod - Module containing the device-side routines (e.g. math).
/// @param KernelModule - Module containing the transformed device-side code.
/// @return The resolved function -- nullptr if not unresolved.
Function *resolveDeviceFunction(Function *Fn, Module &DevMod,
                                Module &KernelModule) {

  // Check for known device-side replacement of frequently used calls
  // (e.g., libmath) or return null to signal that a declaration should
  // be created for what will become a full device-side definition or
  // resolved at link-time.

  // Note that hip provides this functionality via header file
  // mechanisms but since kitsune+tapir are agnostic of host
  // vs. device calls we have some additional steps to take here.
  // However, the entry points below come specifically from the hip/amd
  // headers.
  std::string DevFnName;
  if (Fn->getName() == "sqrtf")
    DevFnName = "__ocml_sqrt_f32";
  else if (Fn->getName() == "acosf")
    DevFnName = "__ocml_acos_f32";
  else if (Fn->getName() == "acoshf")
    DevFnName = "__ocml_acosh_f32";
  else if (Fn->getName() == "asinf")
    DevFnName = "__ocml_asin_f32";
  else if (Fn->getName() == "asinhf")
    DevFnName = "__ocml_asinh_f32";
  else if (Fn->getName() == "atan2f")
    DevFnName = "__ocml_atan2_f32";
  else if (Fn->getName() == "atanf")
    DevFnName = "__ocml_atan_f32";
  else if (Fn->getName() == "atanhf")
    DevFnName = "__ocml_atanh_f32";
  else if (Fn->getName() == "cbrtf")
    DevFnName = "__ocml_cbrt_f32";
  else if (Fn->getName() == "ceilf")
    DevFnName = "__ocml_ceil_f32";
  else if (Fn->getName() == "copysignf")
    DevFnName = "__ocml_copysign_f32";
  else if (Fn->getName() == "cosf")
    DevFnName = "__ocml_cos_f32";
  else if (Fn->getName() == "coshf")
    DevFnName = "__ocml_cosh_f32";
  else if (Fn->getName() == "cospif")
    DevFnName = "__ocml_cospi_f32";
  else if (Fn->getName() == "cyl_bessel_i0f")
    DevFnName = "__ocml_i0_f32";
  else if (Fn->getName() == "cyl_bessel_i1f")
    DevFnName = "__ocml_i1_f32";
  else if (Fn->getName() == "erfcf")
    DevFnName = "__ocml_erfc_f32";
  else if (Fn->getName() == "erfcinvf")
    DevFnName = "__ocml_erfcinv_f32";
  else if (Fn->getName() == "erfcxf")
    DevFnName = "__ocml_erfcx_f32";
  else if (Fn->getName() == "erff")
    DevFnName = "__ocml_erf_f32";
  else if (Fn->getName() == "erfinvf")
    DevFnName = "__ocml_erfinv_f32";
  else if (Fn->getName() == "exp10f")
    DevFnName = "__ocml_exp10_f32";
  else if (Fn->getName() == "exp2f")
    DevFnName = "__ocml_exp2_f32";
  else if (Fn->getName() == "expf")
    DevFnName = "__ocml_exp_f32";
  else if (Fn->getName() == "expm1f")
    DevFnName = "__ocml_expm1_f32";
  else if (Fn->getName() == "fabsf")
    DevFnName = "__ocml_fabs_f32";
  else if (Fn->getName() == "fdimf")
    DevFnName = "__ocml_fdim_f32";
  else if (Fn->getName() == "fdividef")
    llvm_unreachable("fdividef() needs transformation -- unsupported.");
  else if (Fn->getName() == "floorf")
    DevFnName = "__ocml_floor_f32";
  else if (Fn->getName() == "fmaf")
    DevFnName = "__ocml_fma_f32";
  else if (Fn->getName() == "fmaxf")
    DevFnName = "__ocml_fmax_f32";
  else if (Fn->getName() == "fminf")
    DevFnName = "__ocml_fmin_f32";
  else if (Fn->getName() == "fmodf")
    DevFnName = "__ocml_fmod_f32";
  else if (Fn->getName() == "powf")
    DevFnName = "__ocml_pow_f32";
  else if (Fn->getName() == "sinf")
    DevFnName = "__ocml_native_sin_f32";
  else if (Fn->getName() == "sincosf")
    llvm_unreachable("sincosf() needs transformation -- unsupported.");
  else
    DevFnName = Fn->getName().str();

  if (Function *DevFn = KernelModule.getFunction(DevFnName)) {
    // Fn is present in the kernel module, use it as-is.
    LLVM_DEBUG(dbgs() << "\tresolved function '" << DevFn->getName()
                      << "()' in kernel module.\n");
    return DevFn;
  } else if (Function *DevFn = DevMod.getFunction(DevFnName)) {
    // The function is present in the device library we've loaded.
    // This means we will be able to resolve it when we link in the
    // library module prior to constructing the fat binary image.
    // For our codegen to validate add a declaration.
    LLVM_DEBUG(dbgs() << "\tinserting device function decl '"
                      << DevFn->getName() << "()' into kernel module.\n");
    FunctionCallee FCE = KernelModule.getOrInsertFunction(
        DevFnName, DevFn->getFunctionType(), DevFn->getAttributes());
    return cast<Function>(FCE.getCallee());
  } else {
    // The function was not found in the kernel module nor the device
    // library.  In this case we can not resolve it so we return null
    // to the caller.  The next steps are left up to the caller.
    LLVM_DEBUG(dbgs() << "\t\tunable to resolve device function '"
                      << Fn->getName() << "()'.\n");
    return nullptr;
  }
}

/// @brief Transform the given function so it is ready for the final AMDGPU code
/// generation steps.
/// @param F - the function to transform.
/// @return
void transformForGCN(Function &F, Module &DevMod, Module &KernelModule) {
  // There are two main tasks (1) resolve call instructions and (2)
  // transform allocas.  Calls have to resolved via device-side
  // entries (loaded/linked into the DevMod module via bitcode files
  // provide as part of the ROCm distribution).  Secondly, allocas
  // must be transformed to use the appropriate device-side address
  // space...
  LLVM_DEBUG(dbgs() << "\t\ttransforming instructions in function: "
                    << F.getName() << "\n");
  std::map<CallInst *, CallInst *> Replaced;
  std::map<AllocaInst *, AddrSpaceCastInst *> AllocaReplaced;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      Function *Fn = KernelModule.getFunction(CF->getName());
      if (not Fn) {
        LLVM_DEBUG(dbgs() << "\t\t\tcall: " << CF->getName() << "() ");
        Function *DF = resolveDeviceFunction(CF, DevMod, KernelModule);
        if (DF) {
          LLVM_DEBUG(dbgs() << "resolved as: " << DF->getName() << "()\n");
          CallInst *NCI = dyn_cast<CallInst>(CI->clone());
          NCI->setCalledFunction(DF);
          Replaced[CI] = NCI;
        } else
          LLVM_DEBUG(dbgs() << "is unresolved.\n");
      }
    } else if (auto AI = dyn_cast<AllocaInst>(&*I)) {
      if (AI->getAddressSpace() != HIPABI_ALLOCA_ADDR_SPACE) {
        LLVM_DEBUG(dbgs() << "\t\t\ttransforming alloca address space from "
                          << AI->getAddressSpace() << " to "
                          << HIPABI_ALLOCA_ADDR_SPACE << ".\n");
        AllocaInst *NewAI =
            new AllocaInst(AI->getType(), HIPABI_ALLOCA_ADDR_SPACE,
                           AI->getArraySize(), AI->getAlign(), AI->getName());
        NewAI->insertBefore(AI);
        AddrSpaceCastInst *CastAI = new AddrSpaceCastInst(NewAI, AI->getType());
        AllocaReplaced[AI] = CastAI;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\t\t\treplacing identified call instructions...\n");
  for (auto I : Replaced) {
    CallInst *CI = I.first;
    CallInst *NCI = I.second;
    NCI->insertAfter(CI);
    CI->replaceAllUsesWith(NCI);
    CI->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "\t\t\treplacing identified alloca instructions...\n");
  for (auto I : AllocaReplaced) {
    AllocaInst *AI = I.first;
    AddrSpaceCastInst *AC = I.second;
    AC->insertAfter(AI);
    AI->replaceAllUsesWith(AC);
    AI->eraseFromParent();
  }
  LLVM_DEBUG(saveFunctionToFile(&F, F.getName().str(), ".hipabi.ll"));
}

std::set<GlobalValue *> &collect(Constant &c, std::set<GlobalValue *> &seen);

std::set<GlobalValue *> &collect(BasicBlock &bb,
                                 std::set<GlobalValue *> &seen) {
  for (auto &inst : bb)
    for (auto &op : inst.operands())
      if (auto *c = dyn_cast<Constant>(&op))
        collect(*c, seen);
  return seen;
}

std::set<GlobalValue *> &collect(Function &f, std::set<GlobalValue *> &seen) {
  seen.insert(&f);

  for (auto &bb : f)
    collect(bb, seen);
  return seen;
}

std::set<GlobalValue *> &collect(GlobalVariable &g,
                                 std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  if (g.hasInitializer())
    collect(*g.getInitializer(), seen);
  return seen;
}

std::set<GlobalValue *> &collect(GlobalIFunc &g,
                                 std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  llvm_unreachable("kitsune: GNU IFUNC not yet supported");
  return seen;
}

std::set<GlobalValue *> &collect(GlobalAlias &g,
                                 std::set<GlobalValue *> &seen) {
  seen.insert(&g);

  llvm_unreachable("kitsune: GlobalAlias not yet supported");
  return seen;
}

std::set<GlobalValue *> &collect(BlockAddress &blkaddr,
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

} // namespace

void HipABI::transformConstants(Function *Fn) {

  std::map<GetElementPtrInst *, GetElementPtrInst *> GEPMap;

  for (BasicBlock &BB : *Fn) {
    for (Instruction &I : BB) {
      if (auto GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (auto PTy = dyn_cast<PointerType>(GEP->getType())) {
          auto AddrSpace = GEP->getAddressSpace();
          auto PtrAddrSpace = PTy->getAddressSpace();
          if (AddrSpace != PtrAddrSpace) {
            std::vector<Value *> opt_vec;
            for (Use &idx : GEP->indices())
              opt_vec.push_back(idx.get());
            ArrayRef<Value *> IdxList(opt_vec);
            Type *DestTy = GetElementPtrInst::getIndexedType(
                GEP->getSourceElementType(), IdxList);
            assert(DestTy && "GEP indices invalid!");
            GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
                GEP->getSourceElementType(), GEP->getPointerOperand(), IdxList,
                GEP->getName() + ".asp", GEP);
            GEPMap[GEP] = NewGEP;
          }
        }
      }
    }
  }

  for (auto &iGEP : GEPMap) {
    GetElementPtrInst *OldGEP = iGEP.first;
    GetElementPtrInst *NewGEP = iGEP.second;
    std::vector<Use *> uses;
    for (Use &U : OldGEP->uses())
      uses.push_back(&U);
    for (Use *U : uses) {
      if (auto LI = dyn_cast<LoadInst>(U->getUser())) {
        LLVM_DEBUG(dbgs() << "\t\tpatching load instruction: " << *LI << "\n");
        LI->setOperand(LI->getPointerOperandIndex(), NewGEP);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew load: " << *LI << "\n");
      } else if (auto SI = dyn_cast<StoreInst>(U->getUser())) {
        LLVM_DEBUG(dbgs() << "\t\tpatching store instruction: " << *SI << "\n");
        SI->setOperand(SI->getPointerOperandIndex(), NewGEP);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew store: " << *SI << "\n");
      } else if (auto *Call = dyn_cast<CallBase>(U->getUser())) {
        unsigned argNo = Call->getArgOperandNo(U);
        LLVM_DEBUG(dbgs() << "\t\tpatching callable instruction: " << *Call
                          << "\n");
        // FIXME: This is not correct! The function operand should be
        // checked to see what address space it expects.
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", Call);
        Call->setArgOperand(argNo, asCast);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew call: " << *Call << "\n");
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U->getUser())) {
        LLVM_DEBUG(dbgs() << "\t\tpatching gep instruction:\n\t\t\t" << *GEP
                          << "\n");
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", GEP);
        GEP->setOperand(GEP->getPointerOperandIndex(), asCast);
        LLVM_DEBUG(dbgs() << "\t\t\t\tnew gep:\n\t\t\t\t  " << *GEP << "\n");
      } else {
        U->get()->dump();
        U->getUser()->dump();
        assert(false && "unexpected use/user of gep.");
      }
    }
    OldGEP->eraseFromParent();
  }
}

void HipABI::transformArguments(Function *Fn) {
  std::list<Value *> TransformedArgs;
  std::vector<Type *> FnArgTypes(Fn->arg_size());
  for (auto &A : Fn->args()) {
    FnArgTypes[A.getArgNo()] = A.getType();
    if (isa<PointerType>(A.getType())) {
      LLVM_DEBUG(dbgs() << "\t\ttransforming argument: " << A << "\n");
      PointerType *OldPtrTy = cast<PointerType>(A.getType());
      PointerType *NewPtrTy =
          PointerType::get(OldPtrTy->getContext(), HIPABI_GLOBAL_ADDR_SPACE);
      // TODO: Better path here than mutate?
      A.mutateType(NewPtrTy);
      FnArgTypes[A.getArgNo()] = NewPtrTy;
      LLVM_DEBUG(dbgs() << "\t\t\tto: " << A << "\n");
      TransformedArgs.push_back(&A);
    }
  }

  FunctionType *NewFTy = FunctionType::get(Fn->getReturnType(),
                                           ArrayRef<Type *>(FnArgTypes), false);
  Fn->mutateType(NewFTy->getPointerTo());
  // TODO: Need a better path here than mutate... We added this call to LLVM
  // to serve our testing and prototyping purposes.  Not sure there is a clean
  // (and easy to implement) way to accompish the same functionality...
  Fn->mutateValueType(NewFTy);
}

// --- Loop Outliner

/// @brief Return the work item ID for the calling thread. (thread index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
/// @param Low - Low-end of value range if known.
/// @param High -- High-end of value range if known.
Value *HipLoop::emitWorkItemId(IRBuilder<> &Builder, int ItemIndex, int Low,
                               int High) {
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  llvm::MDBuilder MDHelper(Ctx);
  Constant *IndexVal = ConstantInt::get(Int32Ty, ItemIndex);

  std::string WIName = "threadIdx.";
  switch (ItemIndex) {
  case 0:
    WIName.append("x");
    break;
  case 1:
    WIName.append("y");
    break;
  case 2:
    WIName.append("z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
  }
  llvm::Instruction *WorkItemCall =
      Builder.CreateCall(KitHipWorkItemIdFn, {IndexVal}, WIName.c_str());
  // WorkItemCall->setMetadata(llvm::LLVMContext::MD_range, RangeMD);
  return WorkItemCall;
}

/// @brief Return the work group ID for the calling thread. (block index)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupId(IRBuilder<> &Builder, int ItemIndex) {
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *IndexVal = ConstantInt::get(Int32Ty, ItemIndex);
  std::string WGName = "blockIdx.";
  switch (ItemIndex) {
  case 0:
    WGName.append("x");
    break;
  case 1:
    WGName.append("y");
    break;
  case 2:
    WGName.append("z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
  }
  llvm::Instruction *WorkGroupCall =
      Builder.CreateCall(KitHipWorkGroupIdFn, {IndexVal}, WGName);
  return WorkGroupCall;
}

/// @brief Return the work group size for the calling thread. (block size)
/// @param Builder - IR builder for code gen assistance.
/// @param ItemIndex - which work item dimension (x=0,y=1,z=2)
Value *HipLoop::emitWorkGroupSize(IRBuilder<> &Builder, int ItemIndex) {
  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Constant *IndexVal = ConstantInt::get(Int32Ty, ItemIndex);
  std::string WGName = "blockDim.";
  switch (ItemIndex) {
  case 0:
    WGName.append("x");
    break;
  case 1:
    WGName.append("y");
    break;
  case 2:
    WGName.append("z");
    break;
  default:
    llvm_unreachable("unexpected item index!");
  }
  llvm::Instruction *WorkGroupSizeCall =
      Builder.CreateCall(KitHipBlockDimFn, {IndexVal}, WGName);
  return WorkGroupSizeCall;
}

// Static ID for kernel naming -- each encountered kernel (loop)
// during compilation will receive a unique ID.
unsigned HipLoop::NextKernelID = 0;

HipLoop::HipLoop(Module &M, Module &KModule, const std::string &Name,
                 HipABI *LoopTarget)
    : LoopOutlineProcessor(M, KModule), TTarget(LoopTarget), KernelName(Name),
      KernelModule(KModule) {

  std::string UN = KernelName + "." + Twine(NextKernelID).str();
  NextKernelID++;
  KernelName = UN;

  LLVM_DEBUG(dbgs() << "hipabi: hip loop outliner creation:\n"
                    << "\ttransforming loop to kernel: " << KernelName
                    << "(...)\n"
                    << "\tdevice-side module name    : "
                    << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);

  // We use ROCm/HSA/HIP entry points for various runtime calls.  These calls
  // are often at a lower level vs. user-facing entry points.  This follows
  // lower-level code generation details for HIP (that also include details
  // tucked into the HIP-centric header files as well a Clang lowering).

  // Get the local workitem ID for the calling thread.
  KitHipWorkItemIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the work group ID for the calling thread.
  KitHipWorkGroupIdFn = KernelModule.getOrInsertFunction(
      "__ockl_get_group_id",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // Get the block size for the calling thread.
  KitHipBlockDimFn = KernelModule.getOrInsertFunction(
      "__ockl_get_local_size",
      Int64Ty,  // return local thread id.
      Int32Ty); // axis/index select (x=0, y=1, z=2).

  // The runtime entry points above also appear to have some similar intrinsic
  // entry points.
  //
  // TODO: We've experienced some issues related to using these intrinsic
  // calls vs. the runtime calls above.  Need to sort out if there is a
  // reason for this or if one path should be preferred over another.
  KitHipWorkItemIdXFn = /* threadIdx.x */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_x);
  KitHipWorkItemIdYFn = /* threadIdx.y */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_y);
  KitHipWorkItemIdZFn = /* threadIdx. z */
      Intrinsic::getDeclaration(&KernelModule, Intrinsic::amdgcn_workitem_id_z);
  KitHipWorkGroupIdXFn = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_x);
  KitHipWorkGroupIdYFn = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_y);
  KitHipWorkGroupIdZFn = Intrinsic::getDeclaration(
      &KernelModule, Intrinsic::amdgcn_workgroup_id_z);

  // Get entry points into the Hip-centric portion of the Kitsune runtime.
  KernelInstMixTy = StructType::get(Int64Ty,  // number of memory ops.
                                    Int64Ty,  // number of floating point ops.
                                    Int64Ty); // number of integer ops.

  KitHipLaunchFn = M.getOrInsertFunction(
      "__kithip_launch_kernel",
      VoidPtrTy,                       // return an opaque stream
      VoidPtrTy,                       // fat-binary
      VoidPtrTy,                       // kernel name
      VoidPtrTy,                       // arguments
      Int64Ty,                         // trip count
      Int32Ty,                         // threads-per-block
      KernelInstMixTy->getPointerTo(), // instruction mix info
      VoidPtrTy);                      // opaque cuda stream

  KitHipMemPrefetchFn =
      M.getOrInsertFunction("__kithip_mem_gpu_prefetch",
                            VoidPtrTy,  // return an opaque stream
                            VoidPtrTy,  // pointer to prefetch
                            VoidPtrTy); // use opaque stream.
}

HipLoop::~HipLoop() { /* no-op */ }

// TODO: Can we also transform the arguments into a different address space here
// and avoid our use of 'mutate' elsewhere in the code?
void HipLoop::setupLoopOutlineArgs(Function &F, ValueSet &HelperArgs,
                                   SmallVectorImpl<Value *> &HelperInputs,
                                   ValueSet &InputSet,
                                   const SmallVectorImpl<Value *> &LCArgs,
                                   const SmallVectorImpl<Value *> &LCInputs,
                                   const ValueSet &TLInputsFixed) {

  LLVM_DEBUG(dbgs() << "\n\n"
                    << "hipabi: SETTING UP LOOP OUTLINE ARGUMENTS FOR '"
                    << F.getName() << "()'.\n");

  // Add the loop control inputs -- the first parameter defines
  // the extent of the index space.
  {
    Argument *EndArg = cast<Argument>(LCArgs[1]);
    EndArg->setName(".kern.input_size"); // nice for debugging...
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
    StartArg->setName(".kern.start_idx");
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
    GrainsizeArg->setName(".kern.grain_size");
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

unsigned HipLoop::getIVArgIndex(const Function &F, const ValueSet &Args) const {
  // The argument for the primary induction variable is the second input.
  return 1;
}

unsigned HipLoop::getLimitArgIndex(const Function &F,
                                   const ValueSet &Args) const {
  // The argument for the loop limit is the first input.
  return 0;
}

void HipLoop::preProcessTapirLoop(TapirLoopInfo &TL, ValueToValueMapTy &VMap) {

  LLVM_DEBUG(dbgs() << "hiploop: pre-processing tapir loop...\n");

  // TODO: process loop prior to outlining to do GPU/HIP-specific things
  // like capturing global variables, etc.

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValues will need to be
  // cloned into the KernelModule (with different details for the specific
  // type of value).
  LLVM_DEBUG(dbgs() << "\t*- collecting and analyzing global values...\n");
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();
  for (Loop *SL : L) {
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);
  }

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  const DataLayout &DL = KernelModule.getDataLayout();
  unsigned HIPABI_GLOBAL_ADDR_SPACE = DL.getDefaultGlobalsAddressSpace();
  LLVM_DEBUG(dbgs() << "\t*- note: AMDGPU default global addr space: "
                    << HIPABI_GLOBAL_ADDR_SPACE << ".\n");

  // Clone global variables (TODO: and aliases).
  LLVM_DEBUG(dbgs() << "\t*- cloning global variables into kernel module...\n");
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {

      GlobalVariable *NewGV = nullptr;

      LLVM_DEBUG(dbgs() << "\t\t\t* '" << GV->getName() << "' ");

      if (GV->isConstant()) {
        LLVM_DEBUG(dbgs() << "cloned as constant value.\n");
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), true /*isConstant*/,
            GlobalValue::InternalLinkage, GV->getInitializer(),
            GV->getName() + ".dev_gv", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal,
            std::optional<unsigned>(HIPABI_CONST_ADDR_SPACE));
      } else {
        // If GV is non-constant it will need a device-side version whose
        // runtime value must be copied from the host to device prior to
        // the loop's (outlined) kernel function.
        LLVM_DEBUG(
            dbgs() << "non-constant, requires host-to-device copy codegen.\n");
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), false /*isConstant*/,
            GlobalValue::LinkageTypes::ExternalLinkage, GV->getInitializer(),
            GV->getName() + ".dev_gv", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal,
            std::optional<unsigned>(HIPABI_CONST_ADDR_SPACE));
        NewGV->setExternallyInitialized(true);
        NewGV->setVisibility(GlobalValue::ProtectedVisibility);
        // Flag the GV for post-processing (e.g., insert copy calls).
        TTarget->pushGlobalVariable(GV);
      }

      // HIP (appears) to require protected visibility!  Without
      // this the runtime won't be able to find GV for
      // host <-> device transfers.
      NewGV->setDSOLocal(GV->isDSOLocal());
      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
    } else if (dyn_cast<GlobalAlias>(V))
      llvm_unreachable("hipabi: GlobalAlias support not implemented!");
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  LLVM_DEBUG(dbgs() << "\t*- resolving functions for kernel module...\n");
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DF = resolveDeviceFunction(F, *TTarget->getLibDeviceModule(),
                                           KernelModule);
      if (not DF) {
        LLVM_DEBUG(dbgs() << "\t\t\t* adding declaration for function: '"
                          << demangle(F->getName().str()) << "'.\n");
        IRBuilder<> B(F->getContext());
        DF = Function::Create(F->getFunctionType(),
                              GlobalValue::LinkageTypes::ExternalLinkage, 0,
                              F->getName(), &KernelModule);
        auto NewFArgIt = DF->arg_begin();
        for (auto &Arg : F->args()) {
          StringRef ArgName = Arg.getName();
          NewFArgIt->setName(ArgName);
          VMap[&Arg] = &(*NewFArgIt++);
        }
      }
      VMap[F] = DF;
    }
  }

  // FIXME: Support GlobalIFunc at some point. This is a GNU extension, so we
  // may not want to support it at all, but just in case, this is here.
  for (GlobalValue *V : UsedGlobalValues) {
    if (dyn_cast<GlobalIFunc>(V)) {
      llvm_unreachable("hipabi: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  LLVM_DEBUG(dbgs() << "\t*- cloning/creating device-side functions...\n");
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        if (DeviceF) {
          LLVM_DEBUG(dbgs() << "\t\t* clone '" << DeviceF->getName() << "'.\n");
          CloneFunctionInto(DeviceF, F, VMap,
                            CloneFunctionChangeType::DifferentModule, Returns,
                            "");
          LLVM_DEBUG(dbgs() << "\t\t\t(remove target attributes from clone)\n");
          DeviceF->removeFnAttr("target-cpu");
          DeviceF->removeFnAttr("target-features");

          // Exceptions are not supported on the device side, so remove any
          // related attributes...
          LLVM_DEBUG(dbgs()
                     << "\t\t\t(remove exception attributes from clone)\n");
          DeviceF->removeFnAttr(Attribute::UWTable);
          DeviceF->addFnAttr(Attribute::NoUnwind);

          if (OptLevel > 1 &&
              not DeviceF->hasFnAttribute(Attribute::NoInline)) {
            // Try to encourage inlining at high optimization levels.
            DeviceF->addFnAttr(Attribute::AlwaysInline);
            LLVM_DEBUG(dbgs()
                       << "\t\t\t(optimization: mark as always-inline)\n");
          }

          LLVM_DEBUG(dbgs() << "\t\t\t(target for '" << GPUArch << "')\n");
          DeviceF->addFnAttr("target-cpu", GPUArch);
          const std::string target_feature_str =
              "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-"
              "insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,"
              "flat-address-space,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-"
              "insts,+s-memrealtime,+s-memtime-inst";
          DeviceF->addFnAttr("target-features", target_feature_str.c_str());
          LLVM_DEBUG(dbgs() << "\t\t\t(add target features: '"
                            << target_feature_str << "')\n");
          DeviceF->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
          LLVM_DEBUG(dbgs() << "\t\t\t(target for fast calling convention\n");
          DeviceF->setCallingConv(CallingConv::Fast);
        }
      }
      //} else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(v)) {
      //  GlobalVariable *NewGV = cast<GlobalVariable>(VMap[GV]);
      // TODO: Should this be unreachable???  Looks like we stopped short
      //}
    }
  }

  LLVM_DEBUG(dbgs() << "\tfinished preprocessing tapir loop.\n\n");
}

void HipLoop::postProcessOutline(TapirLoopInfo &TLI, TaskOutlineInfo &Out,
                                 ValueToValueMapTy &VMap) {
  // addSyncToOutlineReturns(TLI, Out, VMap);
  Task *T = TLI.getTask();
  Loop *TL = TLI.getLoop();

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

  KernelF->removeFnAttr("target-cpu");
  KernelF->removeFnAttr("target-features");
  KernelF->removeFnAttr(Attribute::UWTable);
  KernelF->addFnAttr(Attribute::NoUnwind);

  // Look for environment variables to help guide some of our kernel
  // attributes...
  //
  // These parameters can be tricky...  Given that the GPU shares
  // resources (e.g., registers and shared memory) across warps
  // fine-tuning things can get quite challenging: using more
  // resources can improve the performance of a single warp but reduce
  // the number of warps that can be simultaneously running...  There's
  // currently not a great approach to optimize for these aspects so
  // searching for a relationship between resource usage and performance is
  // important for tuning (i.e., we don't expect to do this automatically...).
  std::optional<std::string> ThreadsPBVar =
      sys::Process::GetEnv("KITHIP_THREADS_PER_BLOCK");
  if (ThreadsPBVar) {
    MaxThreadsPerBlock = std::stoi(ThreadsPBVar.value());
    if (MaxThreadsPerBlock > HIPABI_DEFAULT_MAX_THREADS_PER_BLOCK)
      report_fatal_error(
          "KITRT_THREADS_PER_BLOCK must be less than 1024 and greater than 0!");
  }
  LLVM_DEBUG(dbgs() << "hipabi: setting kernel's max threads per block: "
                    << MaxThreadsPerBlock << "\n");

  unsigned MinWarpsPerExecUnit = 1;
  std::optional<std::string> MinWarpsPerExecUnitVar =
      sys::Process::GetEnv("KITRT_MIN_WARPS_PER_EXEC_UNIT");
  if (MinWarpsPerExecUnitVar) {
    MinWarpsPerExecUnit = std::stoi(MinWarpsPerExecUnitVar.value());
    if (MinWarpsPerExecUnit < 1 || MinWarpsPerExecUnit >= MaxThreadsPerBlock)
      report_fatal_error(
          "KITRT_MIN_WARPS_PER_EXEC_UNIT must be greater than and "
          "less than the maximum number of threads-per-block!");
  }
  LLVM_DEBUG(
      dbgs() << "hipabi: setting kernel's minimum warps per execution unit to: "
             << MinWarpsPerExecUnit << "\n");
  using namespace llvm::AMDGPU;
  std::string target_feature_str = "";
  switch (llvm::AMDGPU::parseArchAMDGCN(GPUArch)) {
  case GK_GFX90A:
    target_feature_str = "+gfx90a-insts,";
    [[fallthrough]];
  case GK_GFX908:
    target_feature_str += "+dot3-insts,+dot4-insts,+dot5-insts,"
                          "+dot6-insts,+mai-insts,";
    [[fallthrough]];
  case GK_GFX906:
    target_feature_str += "+dl-insts,+dot1-insts,+dot2-insts,+dot7-insts,";
    [[fallthrough]];
  case GK_GFX90C:
  case GK_GFX909:
  case GK_GFX904:
  case GK_GFX902:
  case GK_GFX900:
    target_feature_str += "+gfx9-insts,";
    [[fallthrough]];
  case GK_GFX810:
  case GK_GFX805:
  case GK_GFX803:
  case GK_GFX802:
  case GK_GFX801:
    target_feature_str += "+gfx8-insts,+16-bit-insts,+dpp,"
                          "+s-memrealtime,";
    [[fallthrough]];
  case GK_GFX705:
  case GK_GFX704:
  case GK_GFX703:
  case GK_GFX702:
  case GK_GFX701:
  case GK_GFX700:
    target_feature_str += "+ci-insts,";
    [[fallthrough]];
  case GK_GFX602:
  case GK_GFX601:
  case GK_GFX600:
    target_feature_str += "+s-memtime-inst";
    break;
  case GK_NONE:
    break;
  default:
    llvm_unreachable("Unhandled GPU!");
  }
  // TODO: Need to build target-specific string... and decide if we
  // really need this...
  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("uniform-work-group-size", "true");
  std::string AttrVal = llvm::utostr(MinWarpsPerExecUnit) + std::string(",") +
                        llvm::utostr(MaxThreadsPerBlock);
  KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);
  KernelF->addFnAttr("amdgpu-waves-per-eu", AttrVal);
  KernelF->addFnAttr("target-features", target_feature_str.c_str());
  KernelF->addFnAttr("no-trapping-math", "true");
  KernelF->setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);
  KernelF->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  // Verify that the Thread ID corresponds to a valid iteration.  Because
  // Tapir loops use canonical induction variables, valid iterations range
  // from 0 to the loop limit with stride 1.  The End argument encodes the
  // loop limit. Get end and grain size arguments
  Argument *End;
  Value *Grainsize;
  {
    // TODO: We really only want a grain size of 1 for now...
    auto OutlineArgsIter = KernelF->arg_begin();
    // End argument is the first LC arg.
    End = &*OutlineArgsIter++;

    // Get the grain size value, which is either constant or the third LC
    // arg.
    // if (unsigned ConstGrainsize = TLI.getGrainsize())
    //  Grainsize = ConstantInt::get(PrimaryIV->getType(), ConstGrainsize);
    // else
    Grainsize =
        ConstantInt::get(PrimaryIV->getType(), DefaultGrainSize.getValue());
  }

  IRBuilder<> Builder(Entry->getTerminator());

  // Get the thread ID for this invocation of Helper.
  //
  // This is the classic thread ID calculation:
  //      i = blockDim.x * blockIdx.x + threadIdx.x;
  // For now we only generate 1-D thread IDs.
  Value *ThreadIdx = emitWorkItemId(Builder, /* X */ 0, 0, 1024);
  Value *BlockIdx = emitWorkGroupId(Builder, /* X */ 0);
  Value *BlockDim = emitWorkGroupSize(Builder, /* X */ 0);

  Value *ThreadID = Builder.CreateIntCast(
      Builder.CreateAdd(
          ThreadIdx,
          Builder.CreateMul(BlockIdx, BlockDim, ".kern.blk_offset.x"),
          ".kern.tid.x"),
      PrimaryIV->getType(), false, ".kern.thread_id.x");

  // NOTE/TODO: Assuming that the grainsize is fixed at 1 for the
  // current codegen...
  // ThreadID = Builder.CreateMul(ThreadID, Grainsize);
  Value *ThreadEnd = Builder.CreateAdd(ThreadID, Grainsize, ".kern.last_idx.x");
  Value *Cond = Builder.CreateICmpUGE(ThreadID, End, ".kern.at_end");
  ReplaceInstWithInst(Entry->getTerminator(),
                      BranchInst::Create(Exit, Header, Cond));

  // Replace the loop's induction variable with the GPU thread id.
  PrimaryIVInput->replaceAllUsesWith(ThreadID);

  // Update cloned loop condition to use the thread-end value.
  unsigned TripCountIdx = 0;
  ICmpInst *ClonedCond = cast<ICmpInst>(VMap[TLI.getCondition()]);
  if (ClonedCond->getOperand(0) != End)
    ++TripCountIdx;
  assert(ClonedCond->getOperand(TripCountIdx) == End &&
         "End argument not used in condition!");
  ClonedCond->setOperand(TripCountIdx, ThreadEnd);
  TTarget->saveKernel(KernelF);
}

std::unique_ptr<Module> HipABI::loadBCFile(const std::string &BCFile) {
  LLVMContext &Ctx = KernelModule.getContext();
  llvm::SMDiagnostic SMD;
  LLVM_DEBUG(dbgs() << "\tloading bitcode file: " << BCFile << "...\n");
  std::unique_ptr<Module> BCM = parseIRFile(BCFile, SMD, Ctx);
  if (not BCM)
    report_fatal_error("Failed to parse bitcode file!");
  return BCM;
}

bool HipABI::linkInModule(std::unique_ptr<Module> &Mod) {

  assert(Mod != nullptr && "unexpected null module ptr!");
  // At this point we are ready to link in the device-side module
  // for the final steps of the target transformation.  This
  // basically completes resolution for device-side calls that
  // typically come from the GPU software stack (e.g., the GPU
  // math calls).
  auto L = Linker(KernelModule);

  if (L.linkInModule(std::move(Mod), Linker::LinkOnlyNeeded))
    // TODO: Is there a way to get details here about why the
    // link failed?  For now just use a fatal error until more
    // details can be provided.
    report_fatal_error("Failed to link in HipABI module!");
  else
    return true;
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "hiploop: processing outlined loop call...\n"
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
        LLVM_DEBUG(dbgs() << "\t\t- remove use: " << *Inst << "\n");
        if (Inst != TOI.ReplCall)
          RemoveList.push_back(Inst);
      }
    }
  }

  for (auto I : RemoveList)
    I->eraseFromParent();

  // Make a pass to prep for GCN code generation...
  LLVM_DEBUG(dbgs() << "\t*- transform kernel for GCN code gen.\n");
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForGCN(F, *TTarget->getLibDeviceModule(), KernelModule);

  // Create two builders -- one inserts code into the entry block
  // (e.g., new "up-front" allocas) and the other is for generating
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
  AllocaInst *HipStream = EntryBuilder.CreateAlloca(VoidPtrTy);
  EntryBuilder.CreateStore(ConstantPointerNull::get(VoidPtrTy), HipStream);
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
      Value *SPtr = NewBuilder.CreateLoad(VoidPtrTy, HipStream);
      Value *NewSPtr =
          NewBuilder.CreateCall(KitHipMemPrefetchFn, {VoidPP, SPtr});
      if (not StreamAssigned) {
        NewBuilder.CreateStore(NewSPtr, HipStream);
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
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, ".kern.name");
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
      tapir::getOrInsertFBGlobal(M, HIPAPI_DUMMY_FATBIN_NAME, VoidPtrTy);
  Value *DummyFBPtr = NewBuilder.CreateLoad(VoidPtrTy, DummyFBGV);

  // Deal with type mismatches for the trip count.  A difference
  // introduced via the input source details and the runtime's
  // API type signature for the lanuch.
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Value *TripCount = OrderedInputs[0];
  Value *CastTripCount;
  if (TripCount->getType() != Int64Ty)
    // It is not clear that this is actually signed.
    CastTripCount = NewBuilder.CreateIntCast(TripCount, Int64Ty, true);
  else
    CastTripCount = TripCount; // Simplify cases in launch code gen below...

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

  LLVM_DEBUG(dbgs() << "\t*- code gen kernel launch...\n");
  Value *KSPtr = NewBuilder.CreateLoad(VoidPtrTy, HipStream);
  CallInst *LaunchStream = NewBuilder.CreateCall(
      KitHipLaunchFn, {DummyFBPtr, KNameParam, argsPtr, CastTripCount,
                       TPBlockValue, AI, KSPtr});
  NewBuilder.CreateStore(LaunchStream, HipStream);
  LLVM_DEBUG(dbgs() << "\t\t+- registering launch stream:\n"
                    << "\t\t\tcall: " << *LaunchStream << "\n"
                    << "\t\t\tstream: " << *HipStream << "\n");
  TTarget->registerLaunchStream(LaunchStream, HipStream);

  TOI.ReplCall->eraseFromParent();
  LLVM_DEBUG(dbgs() << "*** finished processing outlined call.\n");
}

// ----- Hip Target

// As is the pattern with the GPU targets, the HipABI is setup to process
// all Tapir constructs within a given input Module (M).  It then creates
// a corresponding module that contains the transformed device-side code.
// This is the KernelModule that is created below in the target
// constructor.
HipABI::HipABI(Module &InputModule)
    : TapirTarget(InputModule),
      KernelModule(HIPABI_KERNEL_NAME_PREFIX + InputModule.getName().str(),
                   InputModule.getContext()) {

  LLVM_DEBUG(saveModuleToFile(&InputModule, InputModule.getName().str(),
                              ".hipabi.ll"));
  LLVM_DEBUG(dbgs() << "hipabi: creating target for module: '" << M.getName()
                    << "'\n");

  LLVMContext &Ctx = InputModule.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  KitHipGetGlobalSymbolFn =
      InputModule.getOrInsertFunction("__kithip_get_global_symbol",
                                      VoidPtrTy,  // return the device pointer
                                      VoidPtrTy,  // fat binary
                                      CharPtrTy); // symbol name
  KitHipMemcpySymbolToDevFn =
      InputModule.getOrInsertFunction("__kithip_memcpy_symbol_to_device",
                                      VoidTy,   // no return
                                      Int32Ty,  // host pointer
                                      Int64Ty,  // device pointer
                                      Int64Ty); // number of bytes to copy
  KitHipSyncFn = M.getOrInsertFunction("__kithip_sync_thread_stream", VoidTy,
                                       VoidPtrTy); // no return, nor parameters
  // Build the details we need for the AMDGPU/HIP target.
  std::string ArchString = "amdgcn";
  Triple TargetTriple(ArchString, "amd", "amdhsa");
  std::string Error;
  const Target *AMDGPUTarget =
      TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (not AMDGPUTarget) {
    errs() << "hipabi: target lookup failed! '" << Error << "'\n";
    report_fatal_error("hipabi: unable to find registered HIP target. "
                       "Was LLVM built with the AMDGPU target enabled?");
  }
  LLVM_DEBUG(dbgs() << "Created target: " << AMDGPUTarget->getShortDescription()
                    << "\n");

  SmallString<255> NewModuleName(ArchString + KernelModule.getName().str());
  sys::path::replace_extension(NewModuleName, ".amdgcn");
  KernelModule.setSourceFileName(NewModuleName.c_str());
  llvm::CodeGenOptLevel TMOptLevel = CodeGenOptLevel::None;
  llvm::CodeModel::Model TMCodeModel = CodeModel::Model::Large;

  if (OptLevel == 0)
    TMOptLevel = CodeGenOptLevel::None;
  else if (OptLevel == 1)
    TMOptLevel = CodeGenOptLevel::Less;
  else if (OptLevel == 2)
    TMOptLevel = CodeGenOptLevel::Default;
  else if (OptLevel >= 3)
    TMOptLevel = CodeGenOptLevel::Aggressive;

  std::string Features = "";
  // TODO: feature is arch specific. need to cross-check.
  // NOTE: If the HSA_XNACK enviornment variable is not set this feature
  // can result in a crash that would appear to be an incorrect/corrupt
  // fatbinary.   Calling the runtime _kitrt_hipEnableXnack() will
  // auto-set the environment variable (now done via the global ctor).
  if (EnableXnack)
    Features += "+xnack,+xnack-support,";

  if (EnableSRAMECC) // TODO: feature is arch specific. need to cross-check.
    Features += ",+sramecc,";

  if (Use64ElementWavefront) // TODO: feature is arch specific. need to cross
                             // check.
    Features += "-wavefrontsize16,-wavefrontsize32,+wavefrontsize64";
  else
    Features += "-wavefrontsize16,+wavefrontsize32,-wavefrontsize64";

  AMDTargetMachine = AMDGPUTarget->createTargetMachine(
      TargetTriple.getTriple(), GPUArch, Features.c_str(), TargetOptions(),
      Reloc::PIC_, TMCodeModel, TMOptLevel);

  LLVM_DEBUG(dbgs() << "\ttarget feature string:\n\t\t"
                    << AMDTargetMachine->getTargetFeatureString() << "\n\n");
  KernelModule.setTargetTriple(TargetTriple.str());
  KernelModule.setDataLayout(AMDTargetMachine->createDataLayout());
  ROCmModulesLoaded = false;
}

HipABI::~HipABI() { /* no-op */ }

std::unique_ptr<Module> &HipABI::getLibDeviceModule() {

  if (not LibDeviceModule) {
    LLVMContext &Ctx = KernelModule.getContext();
    llvm::SMDiagnostic SMD;

    std::initializer_list<std::string> BaseBCFiles = {
        "hip.bc",    // hip built-ins
        "ocml.bc",   // open compute math library
        "ockl.bc",   // open compute kernel library
        "opencl.bc", // printf lives here...
        "oclc_daz_opt_off.bc",
        "oclc_unsafe_math_off.bc",
        "oclc_finite_only_off.bc",
        "oclc_correctly_rounded_sqrt_on.bc",
    };

    std::list<std::string> ROCmBCFiles;
    for (std::string BCFile : BaseBCFiles)
      ROCmBCFiles.push_back(BCFile);

    // Pick the corresponding bitcode file for the
    // target architecture.
    //
    // TODO: Add support for multiple architectures
    // in a single transform.
    if (GPUArch == "gfx900")
      ROCmBCFiles.push_back("oclc_isa_version_900.bc");
    else if (GPUArch == "gfx902")
      ROCmBCFiles.push_back("oclc_isa_version_902.bc");
    else if (GPUArch == "gfx904")
      ROCmBCFiles.push_back("oclc_isa_version_904.bc");
    else if (GPUArch == "gfx906")
      ROCmBCFiles.push_back("oclc_isa_version_906.bc");
    else if (GPUArch == "gfx908")
      ROCmBCFiles.push_back("oclc_isa_version_908.bc");
    else if (GPUArch == "gfx90a")
      ROCmBCFiles.push_back("oclc_isa_version_90a.bc");
    else if (GPUArch == "gfx90c")
      ROCmBCFiles.push_back("oclc_isa_version_90c.bc");
    else {
      errs() << "unsupported amdgpu archicture target: " << GPUArch << ".\n";
      report_fatal_error("fatal error!");
    }

    if (ROCmABITarget == ROCm_ABI_V4)
      ROCmBCFiles.push_back("oclc_abi_version_400.bc");
    else if (ROCmABITarget == ROCm_ABI_V5)
      ROCmBCFiles.push_back("oclc_abi_version_500.bc");
    else
      llvm_unreachable("unhandled ROCm ABI version!");

    if (Use64ElementWavefront)
      ROCmBCFiles.push_back("oclc_wavefrontsize64_on.bc");
    else
      ROCmBCFiles.push_back("oclc_wavefrontsize64_off.bc");

    LLVM_DEBUG(dbgs() << "\tpre-loading AMDGCN device bitcode files.\n");
    for (std::string BCFile : ROCmBCFiles) {
      const std::string GCNFile = std::string(KITSUNE_HIP_BITCODE_DIR) + BCFile;
      LLVM_DEBUG(dbgs() << "\t\t* " << GCNFile << "\n");
      if (LibDeviceModule == nullptr) {
        LibDeviceModule = parseIRFile(GCNFile, SMD, Ctx);
        if (LibDeviceModule == nullptr) {
          SMD.print(GCNFile.c_str(), llvm::errs());
          report_fatal_error("Failed to parse bitcode file!");
        }
      } else {
        std::unique_ptr<Module> BCModule;
        BCModule = parseIRFile(GCNFile, SMD, Ctx);
        if (BCModule == nullptr) {
          SMD.print(GCNFile.c_str(), llvm::errs());
          report_fatal_error("Failed to parse bitcode file!");
        }
        LLVM_DEBUG(dbgs() << "\t\t\tlinking into device module...\n");
        if (Linker::linkModules(*LibDeviceModule, std::move(BCModule),
                                Linker::OverrideFromSrc)) {
          errs() << "hipabi transform: device module preloading failed...\n";
          report_fatal_error("hipabi: failed to link device bitcode module!");
        }
      }
    }
    LLVM_DEBUG(dbgs() << "\tfinished rocm bitcode loading+linking.\n");
  }

  return LibDeviceModule;
}

Value *HipABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grain size on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grain size at 1 has almost always shown to yield the
  // best results in terms of performance but we should take a closer
  // look...  We have some tweaks for experimenting with this via the
  // command line but it remains unexplored.
  Value *Grainsize;
  Grainsize = ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grain size intrinsic call with a computed
  // grain size value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void HipABI::lowerSync(SyncInst &SI) {
  // no-op
}

void HipABI::addHelperAttributes(Function &F) {
  // no-op
}

bool HipABI::preProcessFunction(Function &F, TaskInfo &TI,
                                bool OutliningTapirLoops) {
  return false;
}

void HipABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops) {
    LLVMContext &Ctx = M.getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
    Value *HipStream = ConstantPointerNull::get(VoidPtrTy);
    for (Value *SR : SyncRegList) {
      for (Use &U : SR->uses()) {
        if (auto *SyncI = dyn_cast<SyncInst>(U.getUser()))
          CallInst::Create(KitHipSyncFn, {HipStream}, "",
                           &*SyncI->getSuccessor(0)->begin());
      }
    }
    SyncRegList.clear();
  }
}

// We can't create a correct launch sequence until all the kernels
// within a (LLVM) module are generated.  When post-processing the
// module we create the fatbinary and then to revisit the kernel
// launch calls we created at the loop level and replace the fat
// binary pointer/handle with the completed version.
//
// In addition, we must copy data for global variables from the
// host to the device prior to kernel launches.  This requires
// digging some additional details out of the fat binary.
void HipABI::finalizeLaunchCalls(Module &M, GlobalVariable *BundleBin) {

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  auto &FnList = M.getFunctionList();
  CallInst *SavedLaunchCI = nullptr;

  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {
        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {
            if (CFn->getName().starts_with("__kithip_sync_thread_stream")) {
              if (SavedLaunchCI != nullptr) {
                AllocaInst *StreamAI = getLaunchStream(SavedLaunchCI);
                assert(StreamAI != nullptr && "unexpected null launch stream!");
                IRBuilder<> SyncBuilder(CI);
                Value *HipStream =
                    SyncBuilder.CreateLoad(VoidPtrTy, StreamAI, "hipstreamh");
                CI->setArgOperand(0, HipStream);
                SavedLaunchCI = nullptr;
              }
            } else if (CFn->getName().starts_with("__kithip_launch_kernel")) {
              LLVM_DEBUG(dbgs() << "\t\t\t* patching launch: " << *CI << "\n");
              SavedLaunchCI = CI;
              Value *HipFatbin;
              HipFatbin = CastInst::CreateBitOrPointerCast(
                  BundleBin, VoidPtrTy, "_hipbin.fatbin", CI);
              CI->setArgOperand(0, HipFatbin);

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              assert(NI && "unexpected null instruction!");

              // TODO: Do we want to sync naming conventions up between the
              // CUDA and HIP ABIs?  Might make the world a better place???
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + ".dev_gv";
                LLVM_DEBUG(dbgs() << "\t\t* processing global: "
                                  << HostGV->getName() << "\n");
                // Get the global's name, look it up on the device side,
                // and then issue the copy-to-device call (with appropriate
                // casts).
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitHipGetGlobalSymbolFn, {SymName, CI},
                                     ".hipabi_devptr", NI);
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitHipMemcpySymbolToDevFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    NI);
              }
            }
          }
        }
      }
    }
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable(HIPAPI_DUMMY_FATBIN_NAME, true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(BundleBin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\t\treplacing and removing proxy fatbin ptr.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else
    report_fatal_error("unable to find the proxy fatbin pointer! "
                       "something has gone horribly wrong!");
}

HipABIOutputFile HipABI::createTargetObj(const StringRef &ObjFileName) {

  LLVM_DEBUG(dbgs() << "\tgenerating amdgpu object file.\n");

  std::error_code EC;
  HipABIOutputFile ObjFile = std::make_unique<ToolOutputFile>(
      ObjFileName, EC, sys::fs::OpenFlags::OF_None);
  if (EC) {
    errs() << "hipabi: could not open object file '" << ObjFileName
           << "':" << EC.message();
    report_fatal_error("code transformation failed!");
  }
  ObjFile->keep();

  if (OptLevel > 0) {
    if (OptLevel > 3)
      OptLevel = 3;
    LLVM_DEBUG(dbgs() << "\t- running kernel module optimization passes...\n");
    PipelineTuningOptions pto;
    pto.LoopVectorization = OptLevel > 2;
    pto.SLPVectorization = OptLevel > 2;
    pto.LoopUnrolling = OptLevel >= 2;
    pto.LoopInterleaving = OptLevel > 2;
    pto.LoopStripmine = OptLevel > 2;
    pto.ForgetAllSCEVInLoopUnroll = OptLevel > 2;
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[OptLevel];

    // From the LLVM docs: Create the analysis managers.
    // These must be declared in this order so that they are destroyed in the
    // correct order due to inter-analysis-manager
    // references.
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassBuilder pb(AMDTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    AMDTargetMachine->registerPassBuilderCallbacks(pb, false);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    LLVM_DEBUG(dbgs() << "\t\t* optimize module: " << KernelModule.getName()
                      << "\n");
    mpm.run(KernelModule, mam);
    LLVM_DEBUG(dbgs() << "\t\tpasses complete.\n");
  }

  legacy::PassManager PassMgr;
  if (AMDTargetMachine->addPassesToEmitFile(PassMgr, ObjFile->os(), nullptr,
                                            CodeGenFileType::ObjectFile, false))
    report_fatal_error("hipabi: AMDGPU target failed!");

  PassMgr.run(KernelModule);
  LLVM_DEBUG(dbgs() << "\tkernel optimizations and code gen complete.\n\n");
  LLVM_DEBUG(dbgs() << "\t\tobject file: " << ObjFile->getFilename() << "\n");
  return ObjFile;
}

HipABIOutputFile HipABI::linkTargetObj(const HipABIOutputFile &ObjFile,
                                       const StringRef &LinkedObjFileName) {
  assert(ObjFile != nullptr && "null object file!");
  LLVM_DEBUG(dbgs() << "\tlinking amdgpu object file.\n");
  std::error_code EC;

  HipABIOutputFile LinkedObjFile = std::make_unique<ToolOutputFile>(
      LinkedObjFileName, EC, sys::fs::OpenFlags::OF_None);
  if (EC) {
    errs() << "hipabi: failed to open file '" << LinkedObjFileName
           << "':" << EC.message();
    report_fatal_error("hip code transformation failed!");
  }
  LinkedObjFile->keep();

  // TODO: The lld invocation below is unix-specific...
  auto LLD = sys::findProgramByName("/projects/kitsune/x86_64/18.x/bin/lld");
  if ((EC = LLD.getError()))
    report_fatal_error("executable 'lld' not found! "
                       "check your path?");
  opt::ArgStringList LDDArgList;
  LDDArgList.push_back(LLD->c_str());
  LDDArgList.push_back("-flavor");
  LDDArgList.push_back("gnu");
  LDDArgList.push_back("-m");
  LDDArgList.push_back("elf64_amdgpu");
  LDDArgList.push_back("--no-undefined");
  LDDArgList.push_back("-shared");
  // LDDArgList.push_back("--eh-frame-hdr");
  LDDArgList.push_back("--plugin-opt=-amdgpu-internalize-symbols");
  LDDArgList.push_back("--plugin-opt=-amdgpu-early-inline-all=true");
  LDDArgList.push_back("--plugin-opt=-amdgpu-function-calls=false");
  std::string mcpu_arg = "-plugin-opt=mcpu=" + GPUArch;
  LDDArgList.push_back(mcpu_arg.c_str());
  std::string optlevel_arg = "--plugin-opt=O" + std::to_string(OptLevel);
  LDDArgList.push_back(optlevel_arg.c_str());
  LDDArgList.push_back("-o");
  std::string outfile = LinkedObjFile->getFilename().str();
  LDDArgList.push_back(outfile.c_str());
  std::string infile = ObjFile->getFilename().str();
  LDDArgList.push_back(infile.c_str());
  LDDArgList.push_back(nullptr);

  auto LDDArgs = toStringRefArray(LDDArgList.data());
  LLVM_DEBUG(dbgs() << "hipabi: ld.lld command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : LDDArgs) {
               dbgs() << "\t" << c << ". " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n";);
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*LLD, LDDArgs, std::nullopt, {},
                                     0, // secs to wait -- 0 --> unlimited.
                                     0, // memory limit -- 0 --> unlimited.
                                     &ErrMsg, &ExecFailed);
  if (ExecFailed)
    report_fatal_error("hipabi: 'ldd' execution failed!");

  if (ExecStat != 0)
    report_fatal_error("hipabi: 'ldd' failure - " + StringRef(ErrMsg));

  return LinkedObjFile;
}

HipABIOutputFile HipABI::createBundleFile() {
  // At this point the kernel module should have all the necessary
  // pieces from the input module. Convert the kernel module into
  // a fat binary that can be embedded into the host-side module.
  //
  // We attempt to mimic portions of the steps that the hip/clang
  // frontend uses but given we are "mid-stage" there are some
  // differences.
  //
  // TODO: At present this produces working code but the vast majority
  // of tools (e.g., rocm-obj) don't appear to work correctly.

  LLVM_DEBUG(dbgs() << "hip-abi: creating binary bundle (fat binary).\n");

  std::error_code EC;

  // Run the AMDGPU target to create the associated object file for the
  // kernel module.
  std::string ModelBundleFileName =
      HIPABI_PREFIX + "%%-%%-%%_" + KernelModule.getName().str();
  SmallString<1024> BundleFileName;
  sys::fs::createUniquePath(ModelBundleFileName.c_str(), BundleFileName, true);
  sys::path::replace_extension(BundleFileName, ".amdgpu.o");
  HipABIOutputFile ObjFile = createTargetObj(BundleFileName.str());
  assert(ObjFile != nullptr && "bad unique ptr!");

  // Link the target object file to create a shared object.
  SmallString<255> LinkedObjFileName(BundleFileName);
  sys::path::replace_extension(LinkedObjFileName, ".amdgpu.so");
  HipABIOutputFile LinkedObjFile = linkTargetObj(ObjFile, LinkedObjFileName);

  if (not KeepIntermediateFiles)
    sys::fs::remove(ObjFile->getFilename());

  LLVM_DEBUG(dbgs() << "\tfat binary files:\n"
                    << "\t\tobject file: " << ObjFile->getFilename() << "\n"
                    << "\t\tlinked obj file: " << LinkedObjFile->getFilename()
                    << "\n");

  LinkedObjFile->keep();
  return LinkedObjFile;
}

GlobalVariable *HipABI::embedBundle(HipABIOutputFile &BundleFile) {
  std::unique_ptr<llvm::MemoryBuffer> Bundle = nullptr;
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(BundleFile->getFilename());

  if (std::error_code EC = BufferOrErr.getError()) {
    report_fatal_error("hipabi: failed to load bundle file: " +
                       StringRef(EC.message()));
  }

  Bundle = std::move(BufferOrErr.get());
  LLVM_DEBUG(dbgs() << "\treading binary bundle file, "
                    << Bundle->getBufferSize() << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *BundleArray = ConstantDataArray::getRaw(
      StringRef(Bundle->getBufferStart(), Bundle->getBufferSize()),
      Bundle->getBufferSize(), Int8Ty);
  GlobalVariable *BundleGV;
  BundleGV = new GlobalVariable(
      M, BundleArray->getType(), true, GlobalValue::PrivateLinkage, BundleArray,
      "__hip_fatbin", nullptr, GlobalVariable::NotThreadLocal);

  const char *BundleSectionName = ".hip_fatbin";
  BundleGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
  BundleGV->setSection(BundleSectionName);
  const unsigned HIPCodeObjectAlign = 4096;
  BundleGV->setAlignment(llvm::Align(HIPCodeObjectAlign));
  return BundleGV;
}

void HipABI::registerKernels(Value *HandlePtr, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  llvm::Constant *NullPtr = llvm::ConstantPointerNull::get(VoidPtrTy);

  llvm::Type *RegisterFuncParams[] = {
      VoidPtrPtrTy, CharPtrTy, CharPtrTy, CharPtrTy, Int32Ty,
      VoidPtrTy,    VoidPtrTy, VoidPtrTy, VoidPtrTy, Int32Ty->getPointerTo()};

  FunctionCallee RegisterFn = M.getOrInsertFunction(
      "__hipRegisterFunction",
      FunctionType::get(VoidTy, RegisterFuncParams, false));
  LLVM_DEBUG(dbgs() << "\tregistering kernel functions:\n");
  for (auto KF : KernelFunctions) {
    LLVM_DEBUG(dbgs() << "\t\t* " << KF->getName() << "\n");
    llvm::Constant *KernelName =
        tapir::createConstantStr(KF->getName().str(), M);

    Function *HostSideFunc = M.getFunction(KF->getName());
    if (not HostSideFunc) {
      LLVM_DEBUG(dbgs() << "\t\t\t* not found in host-side module...\n");
      FunctionCallee HSFn =
          M.getOrInsertFunction(KF->getName(), KF->getFunctionType());
      HostSideFunc = cast<Function>(HSFn.getCallee());
      HostSideFunc->setLinkage(KF->getLinkage());
    }

    llvm::Value *Args[] = {
        HandlePtr,
        B.CreateBitCast(HostSideFunc, VoidPtrTy),
        KernelName,
        KernelName,
        llvm::ConstantInt::get(Int32Ty, -1),
        NullPtr,
        NullPtr,
        NullPtr,
        NullPtr,
        llvm::ConstantPointerNull::get(Int32Ty->getPointerTo())};
    B.CreateCall(RegisterFn, Args);
  }
}

void HipABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = IntTy;
  PointerType *CharPtrTy = PointerType::getUnqual(Ctx);

  FunctionCallee RegisterVarFn = M.getOrInsertFunction(
      "__hipRegisterManagedVar",
      VoidTy,       // returns nothing...
      VoidPtrPtrTy, // fatbin handle
      VoidPtrTy,    // Device side (managed) variable (cast).
      VoidPtrTy,    // Global (host side) variable (cast).
      CharPtrTy,    // variable name (same on both device and host side?)
      VarSizeTy,    // variable size (bytes?)
      IntTy);       // alignment

  for (GlobalVariable *HostGV : GlobalVars) {
    std::string DevVarName = HostGV->getName().str() + ".dev_gv";
    GlobalVariable *DevGV = KernelModule.getGlobalVariable(DevVarName);
    assert(DevGV && "unable to find global variable!");
    Value *VarName = tapir::createConstantStr(DevVarName, M);
    uint64_t VarSize = DL.getTypeAllocSize(HostGV->getValueType());
    LLVM_DEBUG(dbgs() << "\t\thost global '" << HostGV->getName().str()
                      << "' to device '" << DevVarName << "'.\n");
    llvm::Value *Args[] = {Handle, // fat binary handle
                           B.CreateBitOrPointerCast(HostGV, VoidPtrTy),
                           B.CreateBitOrPointerCast(HostGV, VoidPtrTy),
                           VarName,
                           ConstantInt::get(VarSizeTy, VarSize),
                           ConstantInt::get(IntTy, DevGV->getAlignment())};
    B.CreateCall(RegisterVarFn, Args);
  }
}

Function *HipABI::createCtor(GlobalVariable *Bundle, GlobalVariable *Wrapper) {

  LLVM_DEBUG(dbgs() << "\tcreating global ctor entries...\n");

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      HIPABI_PREFIX + ".ctor." + sys::path::filename(M.getName()).str(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  LLVM_DEBUG(dbgs() << "\tadd runtime initialization...\n");
  if (EnableXnack) {
    FunctionCallee KitRTEnableXnackFn =
        M.getOrInsertFunction("__kithip_enable_xnack", VoidTy);
    CtorBuilder.CreateCall(KitRTEnableXnackFn, {});
  }

  FunctionCallee KitRTSetDefaultMaxTheadsPerBlockFn = M.getOrInsertFunction(
      "__kithip_set_default_max_threads_per_blk", VoidTy, IntTy);
  CtorBuilder.CreateCall(KitRTSetDefaultMaxTheadsPerBlockFn,
                         {ConstantInt::get(IntTy, MaxThreadsPerBlock)});

  FunctionCallee KitRTInitFn =
      M.getOrInsertFunction("__kithip_initialize", VoidTy);
  CtorBuilder.CreateCall(KitRTInitFn, {});

  // TODO: It is still somewhat unclear if we actually need to register fat
  // binaries given we take a different path with codegen here than the more
  // commmon approach done via the frontend (e.g., we have no stub functions).
  // We should dig more into the details to find out if this is actually
  // needed/helpful/etc.  This might mean digging into the ROCm source...
  FunctionCallee RegisterFatbinaryFn =
      M.getOrInsertFunction("__hipRegisterFatBinary",
                            FunctionType::get(VoidPtrPtrTy, VoidPtrTy, false));
  LLVM_DEBUG(dbgs() << "\tregister fat binary.\n");
  CallInst *RegFatbin = CtorBuilder.CreateCall(
      RegisterFatbinaryFn, CtorBuilder.CreateBitCast(Wrapper, VoidPtrTy));
  GlobalVariable *Handle = new GlobalVariable(
      M, VoidPtrPtrTy,
      /*isConstant=*/false, GlobalValue::InternalLinkage,
      ConstantPointerNull::get(VoidPtrPtrTy), "__hip_gpubin_handle");
  Handle->setAlignment(DL.getPointerPrefAlignment());
  CtorBuilder.CreateAlignedStore(RegFatbin, Handle,
                                 DL.getPointerPrefAlignment());

  LoadInst *HandlePtr = CtorBuilder.CreateLoad(VoidPtrPtrTy, Handle,
                                               HIPABI_PREFIX + "__hip_fatbin");
  HandlePtr->setAlignment(DL.getPointerPrefAlignment());

  // TODO: It is not 100% clear what calls we actually need to make
  // here for kernel, variable, etc. registration with HIP/ROCm.  Clang
  // makes these calls but it is unclear when this is actually
  // necessary...
  //
  // *** CURRENTLY DISABLED W/OUT ISSUES...
  //
  // if (not KernelFunctions.empty()) {
  //   LLVM_DEBUG(dbgs() << "\t\tregistering kernels...\n");
  //   registerKernels(HandlePtr, CtorBuilder);
  // }
  // if (not GlobalVars.empty()) {
  //   LLVM_DEBUG(dbgs() << "\t\tbinding host and device global
  //   variables...\n"); bindGlobalVariables(HandlePtr, CtorBuilder);
  // }

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

Function *HipABI::createDtor(GlobalVariable *BundleHandle) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *VoidPtrPtrTy = VoidPtrTy->getPointerTo();

  FunctionCallee UnregisterFatbinFn =
      M.getOrInsertFunction("__hipUnregisterFatBinary",
                            FunctionType::get(VoidTy, VoidPtrPtrTy, false));

  Function *DtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      HIPABI_PREFIX + ".dtor", &M);

  // TODO: Do we call into this too many times???
  BasicBlock *DtorEntryBB = BasicBlock::Create(Ctx, "entry", DtorFn);
  IRBuilder<> DtorBuilder(DtorEntryBB);
  Value *HandleValue = DtorBuilder.CreateAlignedLoad(
      VoidPtrPtrTy, BundleHandle, DL.getPointerABIAlignment(0));
  DtorBuilder.CreateCall(UnregisterFatbinFn, HandleValue);

  FunctionCallee KitRTDestroyFn =
      M.getOrInsertFunction("__kithip_destroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});
  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void HipABI::registerBundle(GlobalVariable *Bundle) {
  const int BundleMagicID = 0x48495046;
  const DataLayout &Layout = M.getDataLayout();
  Type *BundleStrTy = Bundle->getType();
  Constant *Zeros[] = {ConstantInt::get(Layout.getIndexType(BundleStrTy), 0),
                       ConstantInt::get(Layout.getIndexType(BundleStrTy), 0)};
  Constant *BundlePtr =
      ConstantExpr::getGetElementPtr(Bundle->getValueType(), Bundle, Zeros);
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = PointerType::getUnqual(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  StructType *WrapperTy = StructType::get(IntTy,      // magic #
                                          IntTy,      // version
                                          VoidPtrTy,  // binary (gpu executable)
                                          VoidPtrTy); // unused for now.
  Constant *WrapperS =
      ConstantStruct::get(WrapperTy, ConstantInt::get(IntTy, BundleMagicID),
                          ConstantInt::get(IntTy, 1), BundlePtr,
                          ConstantPointerNull::get(VoidPtrTy));
  GlobalVariable *Wrapper =
      new GlobalVariable(M, WrapperTy, true, GlobalValue::InternalLinkage,
                         WrapperS, "__hip_fatbin_wrapper");
  const char *BundleSectionName = ".hipFatBinSegment";
  Wrapper->setSection(BundleSectionName);
  Wrapper->setAlignment(Align(DL.getPrefTypeAlign(Wrapper->getType())));

  Function *CtorFn = createCtor(Bundle, Wrapper);
  if (CtorFn) {
    LLVM_DEBUG(
        dbgs()
        << "\tadding global ctor for runtime and module initialization.\n");
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    tapir::appendToGlobalCtors(M, ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy),
                               65536, nullptr);
  } else
    LLVM_DEBUG(
        dbgs() << "WARNING: received null ctor -- initialization skipped?\n");
}

void HipABI::postProcessModule() {
  // At this point, all tapir constructs in the input module (M) have been
  // transformed (i.e., outlined) into the kernel module. We can now wrap up
  // module-wide changes for both modules and generate a GPU binary.
  // NOTE: postProcessModule() will not be called in cases where parallelism
  // was not discovered during loop spawning.
  LLVM_DEBUG(dbgs() << "\n\n"
                    << "hipabi: postprocessing the kernel '"
                    << KernelModule.getName() << "' and input '" << M.getName()
                    << "' modules.\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str(),
                              ".hipabi.preopt.ll"));

  if (Function *puts = KernelModule.getFunction("puts")) {
    Value *printf = KernelModule.getFunction("printf");
    if (not printf) {
      LLVMContext &context = KernelModule.getContext();
      Type *paramTys[] = {PointerType::getUnqual(context)};
      Type *retTy = Type::getInt32Ty(context);
      FunctionType *funcTy = FunctionType::get(retTy, paramTys, false);
      FunctionCallee fce = KernelModule.getOrInsertFunction("printf", funcTy);
      printf = fce.getCallee();
    }
    puts->replaceAllUsesWith(printf);
  }

  // Do the final transformation step for the *device* functions
  // in the kernel module.  Note that we have already completed
  // the transformations for the outlined loops (the kernel functions)
  // so we skip them here...
  // Function *KFunc;
  for (Function &F : KernelModule) {
    LLVM_DEBUG(dbgs() << "\tdevice function: " << F.getName() << "() ");
    if (F.isDeclaration())
      LLVM_DEBUG(dbgs() << "(skipping declaration)\n");
    else if (isAMDKernelFunction(&F)) {
      LLVM_DEBUG(dbgs() << "(kernel -- checking calling conventions...)\n");
      transformCallingConv(F);
      transformArguments(&F);
      transformConstants(&F);
      // KFunc = &F;
    } else {
      LLVM_DEBUG(dbgs() << "(transforming)\n");
      transformForGCN(F, *LibDeviceModule, KernelModule);
      transformCallingConv(F);
      transformConstants(&F);
    }
  }

  LLVM_DEBUG(dbgs() << "\n"
                    << "hipabi: LINKING DEVICE LIBRARY...\n");
  linkInModule(LibDeviceModule);

  // At this point we know all tapir loop constructs in the input
  // module (M) have been processed and the kernel module is populated
  // with the corresponding transformed code and is ready to be
  // converted into a fat binary and then embedded into the host-side
  // module.
  LLVM_DEBUG(dbgs() << "\n"
                    << "hipabi: CREATING MODULE FATBINARY...\n");
  HipABIOutputFile BundleFile = createBundleFile();
  LLVM_DEBUG(dbgs() << "\n"
                    << "hipabi: EMBEDDING AND REGISTERING FATBINARY...\n");
  GlobalVariable *Bundle = embedBundle(BundleFile);
  registerBundle(Bundle);

  // Before we finish we now need to patch the launch calls that were
  // initially created before the fat binary was complete.
  LLVM_DEBUG(dbgs() << "\n"
                    << "hipabi: FINALIZE KERNEL LAUNCH CALLS...\n");
  finalizeLaunchCalls(M, Bundle);

  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str(),
                              ".hipabi.final.ll"));

  // EXPERIMENTAL: We have removed code from the host side and
  // inserted some additional code.  Re-run a series of optimization
  // passes -- in general the return on investment here is probably
  // pretty low but we have yet to dig into any details.  For now
  // we will only run this at the highest optimization levels.
  if (HostOptLevel > 0) {
    LLVM_DEBUG(dbgs() << "hipabi: Running experimental post-transform "
                      << "host-side (re)optimization passes.\n");

    PipelineTuningOptions pto;
    pto.LoopVectorization = HostOptLevel > 2;
    pto.SLPVectorization = HostOptLevel > 2;
    pto.LoopUnrolling = HostOptLevel > 1;
    pto.LoopInterleaving = HostOptLevel > 1;
    pto.LoopStripmine = false;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;
    PassBuilder pb(AMDTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    AMDTargetMachine->registerPassBuilderCallbacks(pb, false);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };

    if (HostOptLevel > 3)
      HostOptLevel = 3;
    OptimizationLevel optLevel = optLevels[HostOptLevel];
    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    mpm.run(M, mam);
  }

  if (not KeepIntermediateFiles)
    sys::fs::remove(BundleFile->getFilename());
}

LoopOutlineProcessor *HipABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  // Create a HIP loop outline processor for transforming parallel tapir loop
  // constructs into suitable GPU device code.  We hand the outliner the kernel
  // module (KM) as the destination for all generated (device-side) code.
  std::string ModuleName = sys::path::filename(M.getName()).str();
  Loop *TheLoop = TL->getLoop();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KernelName = Fn->getName().str();

  if (M.getNamedMetadata("llvm.dbg")) {
    // TODO: Is there any hip specific debug naming?
    // If we have debug info in the module use a line number-based
    // naming scheme for kernels.
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KernelName =
        HIPABI_KERNEL_NAME_PREFIX + ModuleName + "_" + Twine(LineNumber).str();
  } else {
    SmallString<255> ModName(Twine(ModuleName).str());
    sys::path::replace_extension(ModName, "");
    KernelName = HIPABI_KERNEL_NAME_PREFIX + ModName.c_str();
  }

  HipLoop *Outliner = new HipLoop(M, KernelModule, KernelName, this);
  return Outliner;
}

void HipABI::pushGlobalVariable(GlobalVariable *GV) {
  GlobalVars.push_back(GV);
}
