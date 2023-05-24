//===- HipABI.cpp - Lower Tapir to the Kitsune HIP GPU back end -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// TODO: Triad/LANL copyright here.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir HIP ABI to convert Tapir
// instructions to calls into the HIP-centric portions of the Kitsune
// runtime for GPUs to produce a fully compiled (non-JIT) executable
// that is suitable for a given GCN architecture target.
//
// TODO: add printf() support.
// TODO: expand device-side calls to cover double precision entry points.
// TODO: revisit/refactor 'mutate' type uses.
// TODO: -- math options for DAZ [on|off], unsafe math [on|off], sqrt rounding
// [on|off]
// TODO: more robust target archicture processing (for correctness).
// TODO: rocm/hsa v5 ABI version testing.
// TODO: host-to-device global value copies need to be implemented.
// TODO: address space selection details (e.g., correctness, performance, etc.)
// TODO: requirements for target feature strings are not clear, need to explore
//       this further.
// TODO: move some of the common helpful functions to a shared location of all
//       target transforms to use.  (e.g. save module/function).
// TODO: poke at host-side post transform passes (likely not all that helpful
//       and could be removed).
//===----------------------------------------------------------------------===//
#include "llvm/Transforms/Tapir/HipABI.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/Inliner.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Utils/AMDGPUEmitPrintf.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

#define DEBUG_TYPE "hipabi" // support for -debug-only=hipabi

static const std::string HIPABI_PREFIX = "__hipabi";
static const std::string HIPABI_KERNEL_NAME_PREFIX = HIPABI_PREFIX + ".kern.";

// ---- HIP transformation-specific command line arguments.
//
//  Usage: -mllvm -hipabi-[option...]
//

// Select the target GPU/GCN architecture.
// TODO: Checking here for a valid target string -- we will fail at some
// point but not gracefully with malformed/unknown architectures.

#ifndef _HIPABI_DEFAULT_ARCH
#define _HIPABI_DEFAULT_ARCH "gfx90a"
#endif

/// Target GPU architecture.
static cl::opt<std::string> GPUArch(
    "hipabi-arch", cl::init(_HIPABI_DEFAULT_ARCH), cl::NotHidden,
    cl::desc("Target AMDGCN architecture. (default: #_HIPABI_DEFAULT_ARCH)"));

// Select the ROCm ABI version to target.
enum ROCmABIVersion {
  ROCm_ABI_V4,
  ROCm_ABI_V5, // Treat this as experimental.
};

static cl::opt<ROCmABIVersion> ROCmABITarget(
    "hipabi-rocm-abi", cl::init(ROCm_ABI_V4), cl::Hidden,
    cl::desc("Select the targeted ROCm ABI version."),
    cl::values(clEnumValN(ROCm_ABI_V4, "v4", "Target ROCm version 4 ABI."),
               clEnumValN(ROCm_ABI_V5, "v5",
                          "Target ROCm v. 5 ABI. (experimental)")));

// Enable/disable 64 element wavefronts.
static cl::opt<bool> Use64ElementWavefront(
    "wavefront64", cl::init(true), cl::Hidden,
    cl::desc("Use 64 element wavefronts. (default: enabled)"));

// Enable/disable xnack.
static cl::opt<bool>
    EnableXnack("hipabi-xnack", cl::init(false), cl::NotHidden,
                cl::desc("Enable/disable xnack. (default: false)"));

// Enable/disable sramecc.
static cl::opt<bool>
    EnableSRAMECC("sramecc", cl::init(true), cl::NotHidden,
                  cl::desc("Enable/disable sramecc.(default: true)"));

// Set the optimization level for use within the transformation.  This
// level is used internally within the transform IR as well handed off
// to any external toolchain elements (e.g., the clang offload bundler).
//
// NOTE: We stick with level 1 as the default here because that level is
// currently required to enabled tapir support via the frontend.
static cl::opt<unsigned>
    OptLevel("hipabi-opt-level", cl::init(1), cl::NotHidden,
             cl::desc("The Tapir HIP target transform optimization level."));

// Enable an extra set of passes over the host-side code after the
// code has been transformed (e.g., loops replaced with kernel launch
// calls).
static cl::opt<bool> RunPostOpts(
    "hipabi-run-post-opts", cl::init(false), cl::NotHidden,
    cl::desc("Run post-transform optimizations on host-side code."));

// Keep intermediate files around after compilation.  This will include
// various stages of the LLVM IR.
static cl::opt<bool>
    KeepIntermediateFiles("hipabi-keep-files", cl::init(false), cl::Hidden,
                          cl::desc("Keep/create intermediate files during the "
                                   "various stages of the transform."));

// Disable generation of prefetch calls prior to kernel launches.
static cl::opt<bool> DisablePrefetch(
    "hipabi-no-prefetch", cl::init(false), cl::Hidden,
    cl::desc("Enable/disable generation of calls to do data "
             "prefetching for memory managed kernel parameters."));

// Enable generation of stream-based prefetching and kernel launches.
static cl::opt<bool>
    CodeGenStreams("hipabi-streams", cl::init(false), cl::Hidden,
                   cl::desc("Generate prefetch and kernel launches "
                            "as a combined set of stream operations."));

/// Set the HIP ABI's default grain size value.  This is used internally
/// by the transform.
static cl::opt<unsigned> DefaultGrainSize(
    "hipabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grainsize used by the transform "
             "when analysis fails to determine one. (default=1)"));

// --- Address spaces.
//
// TODO: Need to work on making sure we understand the nuances
// here for address space selection.  In some cases, wrong address
// spaces seem to cause crashes, in others they are performance
// optimizaitons, and sometimes they almost seem to be no-ops...
// Some of the AMD documentation details seem incomplete.
//
//   See: https://llvm.org/docs/AMDGPUUsage.html#amdgpu-address-spaces-table.
//
static const unsigned GlobalAddrSpace = 1; // global virtual addresses.
static const unsigned ConstAddrSpace = 4;  // indicates that the data will not
                                           // change during the execution of the
                                           // kernel.
static const unsigned AllocaAddrSpace = 5; // "private" (scratch, 32-bit)

// --- Some utility functions for helping during the transformation.

/// @brief Is the given function an AMD GPU kernel.
/// @param F -- the Function to inspect.
/// @return true if the function is a kernel, false otherwise.
static bool isAMDKernelFunction(Function *Fn) {
  return Fn->getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL;
}

/// @brief Wite the given module to a file as readable IR.
/// @param M - the module to save.
/// @param Filename - optional file name (empty string uses module name).
/// TODO: Move this to a common location for utilities.
static void saveModuleToFile(const Module *M,
                             const std::string &FileName = "") {
  std::error_code EC;
  SmallString<256> IRFileName;
  if (FileName.empty())
    IRFileName = Twine(sys::path::filename(M->getName())).str() + ".hipabi.ll";
  else
    IRFileName = Twine(FileName).str() + ".hipabi.ll";

  std::unique_ptr<ToolOutputFile> IRFile = std::make_unique<ToolOutputFile>(
      IRFileName, EC, sys::fs::OpenFlags::OF_None);
  if (not EC) {
    M->print(IRFile->os(), nullptr);
    IRFile->keep();
  } else
    errs() << "warning: unable to save module '" << IRFileName.c_str() << "'\n";
}

/// @brief Write the given function to a file as readable IR.
/// @param Fn - the function to save.
/// @param Filename - optional file name (empty string uses function name).
static void saveFunctionToFile(const Function *Fn,
                               const std::string &FileName = "") {
  std::error_code EC;
  SmallString<256> IRFileName;
  if (FileName.empty()) {
    std::string DName = demangle(Fn->getName().str());
    auto ParenLoc = DName.find("(");
    std::string ShortName = DName.substr(0, ParenLoc);
    IRFileName = ShortName + ".hipabi.ll";
  } else
    IRFileName = Twine(FileName).str() + ".hipabi.ll";

  std::unique_ptr<ToolOutputFile> IRFile = std::make_unique<ToolOutputFile>(
      IRFileName, EC, sys::fs::OpenFlags::OF_None);
  if (not EC) {
    Fn->print(IRFile->os(), nullptr);
    IRFile->keep();
  } else
    errs() << "warning: unable to save function '" << IRFileName.c_str()
           << "'\n";
}

/// @brief Look for the given function in the device-side modules.
/// @param Fn - the function to resolve.
/// @param DevMod - Module containing the device-side routines (e.g. math).
/// @param KernelModule - Module containing the transformed device-side code.
/// @return The resolved function -- nullptr if not unresolved.
static Function *resolveDeviceFunction(Function *Fn, Module &DevMod,
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

void HipABI::transformConstants(Function *Fn) {

  std::map<GetElementPtrInst *, GetElementPtrInst *> GEPMap;

  for (BasicBlock &BB : *Fn) {
    for (Instruction &I : BB) {
      if (auto GEP = dyn_cast<GetElementPtrInst>(&I)) {
        if (auto PTy = dyn_cast<PointerType>(GEP->getType())) {
          auto AddrSpace = GEP->getAddressSpace();
          auto PtrAddrSpace = PTy->getAddressSpace();
          if (AddrSpace != PtrAddrSpace) {
            LLVM_DEBUG(dbgs() << "\t\trepairing GEP addr space.\n"
                              << "\t\t\t[mismatched addrspaces: " << AddrSpace
                              << ", ptr " << PtrAddrSpace << "]\n");

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
        // FIXME: This is not correct! The function operand should be
        // checked to see what address space it expects.
        Instruction *asCast =
            new AddrSpaceCastInst(NewGEP, OldGEP->getType(), "", Call);
        Call->setArgOperand(argNo, asCast);
      } else
        assert(nullptr && "unexpected use of gep");
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
      PointerType *OldPtrTy = dyn_cast<PointerType>(A.getType());
      PointerType *NewPtrTy =
          PointerType::getWithSamePointeeType(OldPtrTy, GlobalAddrSpace);
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
// TODO: Better path here than mutate?  This call is an "extension" to
// serve our testing purproses and not "traditional" LLVM...
#warning "DO NOT USE mutateValueType"
  Fn->mutateValueType(NewFTy);
}

static void transformCallingConv(Function &F) {
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CI->getCallingConv() != CF->getCallingConv()) {
        LLVM_DEBUG(dbgs() << "\t\t\t\tupdated calling convention to "
                          << "match function: " << CF->getName() << "().\n");
        CI->setCallingConv(CF->getCallingConv());
      }
    }
  }
}

/// @brief Transform the given function so it is ready for the final AMDGPU code
/// generation steps.
/// @param F - the function to transform.
/// @return
static void transformForGCN(Function &F, Module &DevMod, Module &KernelModule) {
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
      if (AI->getAddressSpace() != AllocaAddrSpace) {
        LLVM_DEBUG(dbgs() << "\t\t\ttransforming alloca address space from "
                          << AI->getAddressSpace() << " to " << AllocaAddrSpace
                          << ".\n");
        AllocaInst *NewAI =
            new AllocaInst(AI->getType(), AllocaAddrSpace, AI->getArraySize(),
                           AI->getAlign(), AI->getName());
        NewAI->insertBefore(AI);
        AddrSpaceCastInst *CastAI = new AddrSpaceCastInst(NewAI, AI->getType());
        AllocaReplaced[AI] = CastAI;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\t\t\treplacing identifiied call instructions...\n");
  for (auto I : Replaced) {
    CallInst *CI = I.first;
    CallInst *NCI = I.second;
    NCI->insertAfter(CI);
    CI->replaceAllUsesWith(NCI);
    CI->eraseFromParent();
  }

  LLVM_DEBUG(dbgs() << "\t\t\treplacing identifiied alloca instructions...\n");
  for (auto I : AllocaReplaced) {
    AllocaInst *AI = I.first;
    AddrSpaceCastInst *AC = I.second;
    AC->insertAfter(AI);
    AI->replaceAllUsesWith(AC);
    AI->eraseFromParent();
  }
  LLVM_DEBUG(saveFunctionToFile(&F));
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

static std::set<GlobalValue *> &collect(Constant &c,
                                        std::set<GlobalValue *> &seen) {
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

static void promoteConstantExprsToInsts(Function &f) {
  using Operand = std::pair<Instruction *, unsigned>;
  std::map<ConstantExpr *, Operand> exprs;
  do {
    exprs.clear();
    for (inst_iterator i = inst_begin(f), e = inst_end(f); i != e; ++i)
      for (Use &op : i->operands())
        if (auto *cexpr = dyn_cast<ConstantExpr>(&*op))
          exprs[cexpr] = std::make_pair(&*i, op.getOperandNo());

    for (auto &it : exprs) {
      ConstantExpr *expr = it.first;
      Instruction *inst = it.second.first;
      unsigned op = it.second.second;
      Instruction *asInst = expr->getAsInstruction();
      asInst->insertBefore(inst);
      inst->setOperand(op, asInst);
    }
  } while (exprs.size());
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
  Constant *IndexVal = ConstantInt::get(Int32Ty, ItemIndex, ".x");

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
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

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

  // Get entry points into the Hip-centric portion of the Kitsune GPU
  // runtime.
  KitHipLaunchFn = M.getOrInsertFunction("__kitrt_hipLaunchKernel",
                                         VoidTy,    // no return
                                         VoidPtrTy, // fat-binary
                                         VoidPtrTy, // kernel name
                                         VoidPtrTy, // arguments
                                         Int64Ty,   // trip count
                                         VoidPtrTy, // stream
                                         Int64Ty);  // argument size (in bytes)

  KitHipModuleLaunchFn =
      M.getOrInsertFunction("__kitrt_hipLaunchModuleKernel",
                            VoidTy,    // no return
                            VoidPtrTy, // module ptr
                            VoidPtrTy, // kernel name
                            VoidPtrTy, // arguments
                            Int64Ty,   // trip count
                            VoidPtrTy, // stream
                            Int64Ty);  // argument size (in bytes)

  KitHipWaitFn =
      M.getOrInsertFunction("__kitrt_hipStreamSynchronize", VoidTy, VoidPtrTy);

  KitHipMemPrefetchFn =
      M.getOrInsertFunction("__kitrt_hipMemPrefetch", VoidTy, VoidPtrTy);
  KitHipStreamMemPrefetchFn = M.getOrInsertFunction(
      "__kitrt_hipStreamMemPrefetch", // create new stream.
      VoidPtrTy,                      // corresponding stream.
      VoidPtrTy);                     // pointer to prefetch.
  KitHipMemPrefetchOnStreamFn = M.getOrInsertFunction(
      "__kitrt_hipMemPrefetchOnStream", // on given stream.
      VoidTy,                           // no return.
      VoidPtrTy,                        // pointer to prefetch.
      VoidPtrTy);                       // run in this stream.

  KitHipModuleLoadDataFn =
      M.getOrInsertFunction("__kitrt_hipModuleLoadData", VoidPtrTy, VoidPtrTy);
  KitHipGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_hipGetGlobalSymbol",
                            VoidPtrTy,  // return the device pointer for symbol.
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // HIP module

  KitHipMemcpySymbolToDevFn =
      M.getOrInsertFunction("__kitrt_hipMemcpySymbolToDevice",
                            VoidTy,   // returns
                            Int32Ty,  // host pointer
                            Int64Ty,  // device pointer
                            Int64Ty); // number of bytes to copy
}

HipLoop::~HipLoop() { /* no-op */
}

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

  // TODO: process loop prior to outlining to do GPU/HIP-specific things
  // like capturing global variables, etc.
  LLVM_DEBUG(dbgs() << "hipabi: PREPROCESSING TAPIR LOOP.\n"
                    << "\ttransforming into kernel '" << KernelName << "'.\n");

  // Collect the top-level entities (Function, GlobalVariable, GlobalAlias
  // and GlobalIFunc) that are used in the outlined loop. Since the outlined
  // loop will live in the KernelModule, any GlobalValues will need to be
  // cloned into the KernelModule (with different details for the specific
  // type of value).
  LLVM_DEBUG(dbgs() << "\t\t- gathering and analyzing global values...\n");
  std::set<GlobalValue *> UsedGlobalValues;
  Loop &L = *TL.getLoop();
  for (Loop *SL : L) {
    for (BasicBlock *BB : SL->blocks())
      collect(*BB, UsedGlobalValues);
  }

  for (BasicBlock *BB : L.blocks())
    collect(*BB, UsedGlobalValues);

  const DataLayout &DL = KernelModule.getDataLayout();
  unsigned GlobalAddrSpace = DL.getDefaultGlobalsAddressSpace();
  LLVM_DEBUG(dbgs() << "\tNOTE: AMDGPU default global addr space: "
                    << GlobalAddrSpace << ".\n");

  // Clone global variables (TODO: and aliases).
  LLVM_DEBUG(dbgs() << "\tcloning global variables...\n");
  for (GlobalValue *V : UsedGlobalValues) {
    if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
      GlobalVariable *NewGV = nullptr;
      if (GV->isConstant()) {
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), true /*isConstant*/,
            GlobalValue::InternalLinkage, GV->getInitializer(),
            GV->getName() + ".dev_gv", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal,
            std::optional<unsigned>(ConstAddrSpace));
      } else {
        LLVM_DEBUG(dbgs() << "\t\tglobal is non-constant...\n");
        // If GV is non-constant we will need to
        // create a device-side version that will
        // have the host-side value copied over
        // prior to launching the corresponding
        // kernel.
        NewGV = new GlobalVariable(
            KernelModule, GV->getValueType(), false /*isConstant*/,
            GlobalValue::LinkageTypes::ExternalLinkage, GV->getInitializer(),
            GV->getName() + ".dev_gv", (GlobalVariable *)nullptr,
            GlobalValue::NotThreadLocal,
            std::optional<unsigned>(ConstAddrSpace));
        NewGV->setExternallyInitialized(true);
        NewGV->setVisibility(GlobalValue::ProtectedVisibility);
        // Flag the GV for post-processing (e.g., insert copy calls).
        // TODO: rename for clarity...
        TTarget->pushGlobalVariable(GV);
      }
      // HIP (appears) to require protected visibility!  Without
      // this the runtime won't be able to find GV for
      // host <-> device transfers.
      NewGV->setDSOLocal(GV->isDSOLocal());
      NewGV->setAlignment(GV->getAlign());
      VMap[GV] = NewGV;
      LLVM_DEBUG(dbgs() << "\t\tcreated device-side global variable '"
                        << NewGV->getName() << "'.\n");
    } else if (dyn_cast<GlobalAlias>(V))
      llvm_unreachable("kitsune: GlobalAlias not implemented.");
  }

  // Create declarations for all functions first. These may be needed in the
  // global variables and aliases.
  LLVM_DEBUG(dbgs() << "\tcreating function decls.\n");
  for (GlobalValue *G : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(G)) {
      Function *DF = resolveDeviceFunction(F, *TTarget->getLibDeviceModule(),
                                           KernelModule);
      if (not DF) {
        IRBuilder<> B(F->getContext());
        LLVM_DEBUG(dbgs() << "\t\t* create device side function: "
                          << demangle(F->getName().str())
                          << "(demangled name)\n");
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
      llvm_unreachable("kitsune: GlobalIFunc not yet supported.");
    }
  }

  // Now clone any function bodies that need to be cloned. This should be
  // done as late as possible so that the VMap is populated with any other
  // global values that need to be remapped.
  for (GlobalValue *v : UsedGlobalValues) {
    if (Function *F = dyn_cast<Function>(v)) {
      if (F->size()) {
        SmallVector<ReturnInst *, 8> Returns;
        Function *DeviceF = cast<Function>(VMap[F]);
        if (DeviceF) {
          LLVM_DEBUG(dbgs() << "\t\tcloning function '" << DeviceF->getName()
                            << "' (" << demangle(DeviceF->getName().str())
                            << ") into kernel module.\n");
          CloneFunctionInto(DeviceF, F, VMap,
                            CloneFunctionChangeType::DifferentModule, Returns,
                            "");

          DeviceF->removeFnAttr("target-cpu");
          DeviceF->removeFnAttr("target-features");

          // Exceptions are not supported on the device side, so remove any
          // related attributes...
          DeviceF->removeFnAttr(Attribute::UWTable);
          DeviceF->addFnAttr(Attribute::NoUnwind);

          if (OptLevel > 1 &&
              not DeviceF->hasFnAttribute(Attribute::NoInline)) {
            // Try to encourage inlining at high optimization levels.
            DeviceF->addFnAttr(Attribute::AlwaysInline);
            LLVM_DEBUG(dbgs() << "\t\t\tset always inline attribute.\n");
          }
          DeviceF->addFnAttr("target-cpu", GPUArch);
          const std::string target_feature_str =
              "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-"
              "insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+"
              "flat-address-space,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-"
              "insts,+s-memrealtime,+s-memtime-inst";
          DeviceF->addFnAttr("target-features", target_feature_str.c_str());
          DeviceF->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
          DeviceF->setCallingConv(CallingConv::Fast);
        }
      }
    } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(v)) {
      GlobalVariable *NewGV = cast<GlobalVariable>(VMap[GV]);
      LLVM_DEBUG(dbgs() << "\tvisiting global variable address space details "
                        << "for " << NewGV->getName() << "\n");
      // for (Use &GVUse : llvm::make_early_inc_range(NewGV->uses())) {
      // }
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

  // TODO: Need to build target-specific string... and decide if we
  // really need this...
  const std::string target_feature_str =
      "+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-"
      "insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+"
      "flat-address-space,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+mai-"
      "insts,+s-memrealtime,+s-memtime-inst";
  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("uniform-work-group-size", "true");
  std::string AttrVal = std::string("1,") + llvm::utostr(1024);
  KernelF->addFnAttr("amdgpu-flat-work-group-size", AttrVal);
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
    Grainsize = ConstantInt::get(PrimaryIV->getType(), 1);
    // DefaultGrainSize.getValue());
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
  return std::move(BCM);
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

StructType *HipLoop::createKernelArgsType() {
  int ArgCount = OrderedInputs.size();
  LLVM_DEBUG(dbgs() << "arg count: " << ArgCount << "\n");
  std::vector<Type *> TyVec;
  for (Value *V : OrderedInputs) {
    Type *Ty = V->getType();
    TyVec.push_back(Ty);
    LLVM_DEBUG(dbgs() << "adding type: " << *Ty << "\n");
  }
  LLVM_DEBUG(dbgs() << "arg count: " << TyVec.size() << "\n");

  ArrayRef<Type *> ArgTypes(TyVec);
  LLVMContext &Ctx = M.getContext();
  StructType *ArgStructTy = StructType::get(Ctx, ArgTypes, false);
  LLVM_DEBUG(dbgs() << "args struct: " << *ArgStructTy << "\n");
  return ArgStructTy;
}

void HipLoop::processOutlinedLoopCall(TapirLoopInfo &TL, TaskOutlineInfo &TOI,
                                      DominatorTree &DT) {

  LLVM_DEBUG(dbgs() << "hipabi: PROCESSING OUTLINED LOOP CALL.\n"
                    << "\tkernel: " << KernelName << "\n");
  LLVMContext &Ctx = M.getContext();
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());

  // FIXME: Do we need to do this here or could we move it into
  // the post process module stage where we process the other
  // functions in the kernel module???
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForGCN(F, *TTarget->getLibDeviceModule(), KernelModule);

  BasicBlock &EBB = Parent->getEntryBlock();
  IRBuilder<> EB(&EBB.front());

  Value *prefetchStream = nullptr;
  if (not CodeGenStreams) {
    LLVM_DEBUG(dbgs() << "\tstream code generation is off.\n");
    // If we are going to use the default stream we
    // set the main prefetch stream to null and it
    // will propagate through all prefetch and launch
    // calls.
    prefetchStream = ConstantPointerNull::get(VoidPtrTy);
  }

  Type *Int64Ty = Type::getInt64Ty(Ctx);
  const DataLayout &DL = M.getDataLayout();
  StructType *KernelArgsTy = createKernelArgsType();
  AllocaInst *KernelArgs = EB.CreateAlloca(KernelArgsTy, nullptr, ".kern.args");
  auto ArgsAllocSize = DL.getTypeAllocSize(KernelArgsTy);
  Value *ArgsSize = ConstantInt::get(Int64Ty, ArgsAllocSize);

  unsigned int i = 0;
  for (Value *V : OrderedInputs) {
    Type *VTy = V->getType();
    // TODO: Do we ever need to Handle pass-by-value structs?
    assert(not VTy->isAggregateType() &&
           "aggregate-typed kernel arguments not yet supported!");

    Value *ArgPtr = B.CreateStructGEP(KernelArgsTy, KernelArgs, i);
    B.CreateStore(V, ArgPtr);
    i++;

    if (not DisablePrefetch) {
      if (VTy->isPointerTy()) {
        Value *VoidPP = B.CreateBitCast(V, VoidPtrTy);
        if (prefetchStream == nullptr) { // stream codegen enabled.
          assert(KitHipStreamMemPrefetchFn &&
                 "no kitsune hip stream mem prefetch function!");
          prefetchStream = B.CreateCall(KitHipStreamMemPrefetchFn, {VoidPP});
        } else {
          assert(KitHipMemPrefetchOnStreamFn &&
                 "no kitsune hip mem prefetch on stream function!");
          LLVM_DEBUG(dbgs() << "\t\t*issue prefetch for arg #" << i << "\n");
          B.CreateCall(KitHipMemPrefetchOnStreamFn, {VoidPP, prefetchStream});
        }
      }
    }
  }
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

  // We can't get to the complete fat binary data until all loops in the
  // input module have been processed (i.e., the complete kernel module is
  // populated, converted to GCN, turned into an assembled binary, etc.).
  // Because of this we create a "stand in" (proxy) here and will replace
  // it later in the ABI's transformation pipeline.
  Constant *ProxyFBGV =
      tapir::getOrInsertFBGlobal(M, "_hipabi.proxy_fatbin", VoidPtrTy);
  Value *ProxyFBPtr = B.CreateLoad(VoidPtrTy, ProxyFBGV);

  Value *TCCI = nullptr;
  if (TripCount->getType() != Int64Ty)
    // It is not clear that this is actually signed.
    TCCI = B.CreateIntCast(TripCount, Int64Ty, true);
  else
    TCCI = TripCount; // Simplify cases in launch code gen below...

  if (not TTarget->hasGlobalVariables()) {
    LLVM_DEBUG(dbgs() << "\tcreating kernel launch (no globals)...\n");
    Value *VPKernArgs = B.CreateBitCast(KernelArgs, VoidPtrTy);
    assert(KitHipLaunchFn && "no kitsune hip launch function!");
    B.CreateCall(KitHipLaunchFn, {ProxyFBPtr, KNameParam, VPKernArgs, TCCI,
                                  prefetchStream, ArgsSize});
  } else {
    LLVM_DEBUG(dbgs() << "\tcreating kernel launch (w/ globals).\n");
    Value *VPKernArgs = B.CreateBitCast(KernelArgs, VoidPtrTy);
    Value *HipModule = B.CreateCall(KitHipModuleLoadDataFn, {ProxyFBPtr});
    B.CreateCall(KitHipModuleLaunchFn, {HipModule, KNameParam, VPKernArgs, TCCI,
                                        prefetchStream, ArgsSize});
  }
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

  LLVM_DEBUG(saveModuleToFile(&InputModule));
  LLVM_DEBUG(dbgs() << "hipabi: creating target for module: '" << M.getName()
                    << "'\n");

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

  llvm::CodeGenOpt::Level TMOptLevel;
  llvm::CodeModel::Model TMCodeModel = CodeModel::Model::Large;

  if (OptLevel == 0)
    TMOptLevel = CodeGenOpt::Level::None;
  else if (OptLevel == 1)
    TMOptLevel = CodeGenOpt::Level::Less;
  else if (OptLevel == 2)
    TMOptLevel = CodeGenOpt::Level::Default;
  else if (OptLevel >= 3)
    TMOptLevel = CodeGenOpt::Level::Aggressive;
  std::string Features = "";

  if (EnableXnack) // TODO: feature is arch specific. need to cross-check.
    // NOTE: If the HSA_XNACK enviornment variable is not set this feature
    // can result in a crash that would appear to be an incorrect/corrupt
    // fatbinary.   Calling the runtime _kitrt_hipEnableXnack() will
    // auto-set the environment variable (now done via the global ctor).
    Features += "+xnack,+xnack-support";
  else
    Features += "-xnack,+xnack-support";

  if (EnableSRAMECC) // TODO: feature is arch specific. need to cross-check.
    Features += ",+sramecc";
  else
    Features += ",-sramecc";

  if (Use64ElementWavefront) // TODO: feature is arch specific. need to cross
                             // check.
    Features += ",-wavefrontsize16,-wavefrontsize32,+wavefrontsize64";
  else
    Features += ",-wavefrontsize16,+wavefrontsize32,-wavefrontsize64";

  AMDTargetMachine = AMDGPUTarget->createTargetMachine(
      TargetTriple.getTriple(), GPUArch, Features.c_str(), TargetOptions(),
      Reloc::PIC_, TMCodeModel, TMOptLevel);

  LLVM_DEBUG(dbgs() << "\ttarget feature string:\n\t\t"
                    << AMDTargetMachine->getTargetFeatureString() << "\n\n");
  KernelModule.setTargetTriple(TargetTriple.str());
  KernelModule.setDataLayout(AMDTargetMachine->createDataLayout());
  ROCmModulesLoaded = false;
}

HipABI::~HipABI() { /* no-op */
}

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
      const std::string GCNFile = "amdgcn/bitcode/" + BCFile;
      LLVM_DEBUG(dbgs() << "\t\t* " << GCNFile << "\n");
      std::optional<std::string> BCFPath =
          sys::Process::FindInEnvPath("ROCM_PATH", GCNFile);
      if (not BCFPath)
        report_fatal_error("Unable to find rocm bitcode file! "
                           "Is ROCM_PATH set in your enviornment?");
      if (LibDeviceModule == nullptr) {
        LibDeviceModule = parseIRFile(*BCFPath, SMD, Ctx);
        if (LibDeviceModule == nullptr) {
          SMD.print(BCFPath->c_str(), llvm::errs());
          report_fatal_error("Failed to parse bitcode file!");
        }
      } else {
        std::unique_ptr<Module> BCModule;
        BCModule = parseIRFile(*BCFPath, SMD, Ctx);
        if (BCModule == nullptr) {
          SMD.print(BCFPath->c_str(), llvm::errs());
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

void HipABI::preProcessFunction(Function &F, TaskInfo &TI,
                                bool OutliningTapirLoops) {
  // no-op
}

void HipABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops) {
    LLVMContext &Ctx = M.getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    FunctionCallee KitHipSyncFn =
        M.getOrInsertFunction("__kitrt_hipSynchronizeStreams",
                              VoidTy); // no return & no parameters

    for (Value *SR : SyncRegList) {
      for (Use &U : SR->uses()) {
        if (auto *SyncI = dyn_cast<SyncInst>(U.getUser()))
          CallInst::Create(KitHipSyncFn, "", &*SyncI->getSuccessor(0)->begin());
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
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  FunctionCallee KitHipGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_hipGetGlobalSymbol",
                            VoidPtrTy,  // device pointer
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // HIP "module"

  FunctionCallee KitHipMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitrt_hipMemcpySymbolToDevice",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            VoidPtrTy, // device ptr
                            Int64Ty);  // num bytes

  auto &FnList = M.getFunctionList();
  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {

        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {
            if (CFn->getName().startswith("__kitrt_hipLaunchKernel")) {
              Value *HipFatbin = CastInst::CreateBitOrPointerCast(
                  BundleBin, VoidPtrTy, "_hipbin.fatbin", CI);
              LLVM_DEBUG(dbgs() << "\t\t* patching launch: " << *CI << "\n");
              CI->setArgOperand(0, HipFatbin);
            } else if (CFn->getName().startswith("__kitrt_hipModuleLoadData")) {
              Value *HipFatbin = CastInst::CreateBitOrPointerCast(
                  BundleBin, VoidPtrTy, "_hipbin.fatbin", CI);
              LLVM_DEBUG(dbgs()
                         << "\t\t* patching module launch: " << *CI << "\n");
              CI->setArgOperand(0, HipFatbin);
              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              //
              // TODO: assert here that NI indeed points to the launch.
              //
              assert(NI && "unexpected null instruction!");
              for (auto &HostGV : GlobalVars) {

                // Lookup the matching device-side global...
                std::string DevVarName = HostGV->getName().str() + ".dev_gv";
                LLVM_DEBUG(dbgs() << "\t\t* processing global: "
                                  << HostGV->getName() << "\n");
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr = CallInst::Create(KitHipGetGlobalSymbolFn,
                                                 {SymName, CI}, "", NI);
                // Copy the value from host to device...
                Value *VGVPtr =
                    CastInst::CreatePointerCast(HostGV, VoidPtrTy, "", NI);
                uint64_t NumBytes = DL.getTypeAllocSize(HostGV->getValueType());
                CallInst::Create(
                    KitHipMemcpyToDeviceFn,
                    {VGVPtr, DevPtr, ConstantInt::get(Int64Ty, NumBytes)}, "",
                    NI);
              }
            }
          }
        }
      }
    }
  }

  GlobalVariable *ProxyFB = M.getGlobalVariable("_hipabi.proxy_fatbin", true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(BundleBin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\t\treplacing and removing proxy fatbin ptr.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    LLVM_DEBUG(dbgs() << "\t\tWARNING! "
                      << "whoopsie... unable to find proxy fatbin ptr!\n"
                      << "something might be broken...\n\n");
  }
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
    LLVM_DEBUG(dbgs() << "\trunning module optimization passes.\n");
    PipelineTuningOptions pto;
    pto.LoopVectorization = OptLevel > 2;
    pto.SLPVectorization = OptLevel > 2;
    pto.LoopUnrolling = true;
    pto.LoopInterleaving = true;
    pto.LoopStripmine = true;
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassBuilder pb(AMDTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    AMDTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[OptLevel];
    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(optLevel);
    mpm.addPass(VerifierPass());
    LLVM_DEBUG(dbgs() << "\t\t* module: " << KernelModule.getName() << "\n");
    mpm.run(KernelModule, mam);
  }

  legacy::PassManager PassMgr;
  if (AMDTargetMachine->addPassesToEmitFile(PassMgr, ObjFile->os(), nullptr,
                                            CodeGenFileType::CGFT_ObjectFile,
                                            false))
    report_fatal_error("hipabi: AMDGPU target failed!");

  PassMgr.run(KernelModule);
  LLVM_DEBUG(dbgs() << "\toptimizations and code gen complete.\n\n");
  LLVM_DEBUG(dbgs() << "\t\tobject file: " << ObjFile->getFilename() << "\n");
  return std::move(ObjFile);
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

  auto LLD = sys::findProgramByName("ld.lld");
  if ((EC = LLD.getError()))
    report_fatal_error("executable 'ld.lld' not found! "
                       "check your path?");
  opt::ArgStringList LDDArgList;
  LDDArgList.push_back(LLD->c_str());
  // LDDArgList.push_back("--no-undefined");
  LDDArgList.push_back("-shared");
  LDDArgList.push_back("--eh-frame-hdr");
  LDDArgList.push_back("--plugin-opt=-amdgpu-internalize-symbols");
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

  return std::move(LinkedObjFile);
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
  BundleGV = new GlobalVariable(M, BundleArray->getType(), true,
                                GlobalValue::PrivateLinkage, BundleArray,
                                "__hip_fatbin");
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
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);
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
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = IntTy;
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

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
    PointerType *VoidDevPtrTy =
        Type::getInt8PtrTy(Ctx, DevGV->getAddressSpace());
    Value *DevGVAddrCast =
        B.CreatePointerBitCastOrAddrSpaceCast(DevGV, VoidPtrTy);
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
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      HIPABI_PREFIX + ".ctor." + sys::path::filename(M.getName()).str(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  // Tuck some calls in that initialize the Kitsune runtime.  This includes
  // enabling xnack and explicitly initializing HIP (even though documentation
  // suggests it is optional).
  if (EnableXnack) {
    FunctionCallee KitRTEnableXnackFn =
        M.getOrInsertFunction("__kitrt_hipEnableXnack", VoidTy);
    CtorBuilder.CreateCall(KitRTEnableXnackFn, {});
  }
  LLVM_DEBUG(dbgs() << "\tadd runtime initialization...\n");
  FunctionCallee KitRTInitFn = M.getOrInsertFunction("__kitrt_hipInit", VoidTy);
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
  // makes these calls but it is unclear what (and when) this is actually
  // necessary...

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
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
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
      M.getOrInsertFunction("__kitrt_hipDestroy", VoidTy);
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
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
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
  Wrapper->setAlignment(Align(DL.getPrefTypeAlignment(Wrapper->getType())));

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
  } else {
    LLVM_DEBUG(
        dbgs() << "WARNING: received null ctor -- initialization skipped?\n");
  }
}

void HipABI::postProcessModule() {
  // At this point all tapir constructs in the input module (M) have
  // been transformed (e.g., outlined) into the "kernel module"
  // (KernelModule) and we can now take the necessary steps to wrap up
  // the necessary steps for module-wide changes for both modules.
  LLVM_DEBUG(dbgs() << "\n\n"
                    << "hipabi: POST PROCESSING the kernel (device) '"
                    << KernelModule.getName() << "' and input '" << M.getName()
                    << "' modules.\n");
  LLVM_DEBUG(saveModuleToFile(&KernelModule, KernelModule.getName().str() +
                                                 ".post.unoptimized"));

  if (Function *puts = KernelModule.getFunction("puts")) {
    Value *printf = KernelModule.getFunction("printf");
    if (not printf) {
      LLVMContext &context = KernelModule.getContext();
      Type *paramTys[] = {Type::getInt8PtrTy(context)};
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

  LLVM_DEBUG(
      saveModuleToFile(&KernelModule, KernelModule.getName().str() + ".final"));

  // EXPERIMENTAL: We have removed code from the host side and
  // inserted some additional code.  Re-run a series of optimization
  // passes -- in general the return on investment here is probably
  // pretty low but we have yet to dig into any details.  For now
  // we will only run this at the highest optimization levels.
  if (RunPostOpts && OptLevel > 2) {
    LLVM_DEBUG(dbgs() << "hipabi: Running EXPERIMENTAL post-transform "
                      << "host-side optimization pass.\n");

    PipelineTuningOptions pto;
    pto.LoopVectorization = OptLevel > 2;
    pto.SLPVectorization = OptLevel > 2;
    pto.LoopUnrolling = true;
    pto.LoopInterleaving = true;
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
    AMDTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel optLevel = optLevels[OptLevel];
    if (OptLevel <= 3) // unsigned...
      optLevel = optLevels[OptLevel];

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
  std::string ModuleName = sys::path::filename(M.getName()).str();
  std::string KernelName;

  if (M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module, use a line number-based
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
