//===- CudaABI.cpp - Lower Tapir to the Kitsune GPU back end -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Kitsune+Tapir CUDA ABI to transform Tapir
// IR into CUDA-centric code that includes PTX code as well as
// supporting calls into the Kitsune runtime to produce an associated
// set of GPU kernels for the Tapir parallel constructs.
//
// All Tapir constructs in a module are converted into device-side
// (outlined) functions that are stored in a specific "kernel" LLVM
// Module.  This process follows the callback sequence supported by
// Tapir's lowering mechanisms as well as components such as the NVPTX
// target (https://llvm.org/docs/NVPTXUsage.html).  Further details of
// some of the structure required for code transformation are taken
// from CUDA's code gen of CUDA programs.
//
//===----------------------------------------------------------------------===//
//
#include "llvm/Transforms/Tapir/CudaABI.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
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
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Tapir/Outline.h"
#include "llvm/Transforms/Tapir/TapirGPUUtils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Vectorize.h"

#include <fstream>
#include <sstream>

using namespace llvm;

#define DEBUG_TYPE "cuabi" // support for -debug-only=cudabi

static const std::string CUABI_PREFIX = "__cuabi";
static const std::string CUABI_KERNEL_NAME_PREFIX = CUABI_PREFIX + ".kern.";

// ---- CUDA transformation-specific command line arguments.
//
//   Usage: -mllvm -cuabi-[option...]
//

// Select a specific target NVIDIA GPU architecture.
//
// This will be passed directly on to ptxas.
//
// NOTE: At this point in time we do not provide support for the older range
// of GPU architectures (e.g., we favor 64-bit and SM_60 or newer, which
// follows the trends of longer term CUDA support.  Although exposed here, we
// have not tested 32-bit host support.
//

#ifndef _CUDAABI_DEFAULT_ARCH
#define _CUDAABI_DEFAULT_ARCH "sm_86"
#endif

/// Target GPU architecture.
// This is handed to ptxas to do the codegen (i.e., use the same arch string).
// We have only tested with SM_75 and later so YMMV with earlier targets.
static cl::opt<std::string>
    GPUArch("cuabi-arch", cl::init(_CUDAABI_DEFAULT_ARCH), cl::NotHidden,
            cl::desc("Target GPU architecture for CUDA ABI transformation."
                     "(default: " _CUDAABI_DEFAULT_ARCH ")"));

/// Select 32- vs. 64-bit host architecture. Passed directly to ptxas.
// TODO: This can be deprecated for our use cases.
static cl::opt<std::string>
    HostMArch("cuabi-march", cl::init("64"), cl::Hidden,
              cl::desc("Specify 32- or 64-bit host architecture."
                       " (default=64-bit)."));

/// Enable verbose mode in the second tools (e.g., ptxas).
static cl::opt<bool>
    Verbose("cuabi-verbose", cl::init(false), cl::NotHidden,
            cl::desc("Enable verbose mode for cuda toolchain components. "
                     "(default=off)"));

/// Enable debug mode. Passed directly to ptxas.
static cl::opt<bool>
    Debug("cuabi-debug", cl::init(false), cl::NotHidden,
          cl::desc("Enable debug information for GPU device code. "
                   " (default=false)"));

/// Surpress the generation of debug sections in the final object
/// file.  This option is ignored if not used with the debug or
/// generate-line-info options.
static cl::opt<bool>
    SurpressDBInfo("cuabi-surpress-debug-info", cl::init(false), cl::Hidden,
                   cl::desc(" Do not generate debug information sections "
                            "in final output object file."));

/// Generate line information for all generated GPU kernels.  Passed
/// directly to 'ptxas' as '--generate-line-info'.
static cl::opt<bool>
    GenLineInfo("cuabi-generate-line-info", cl::init(false), cl::NotHidden,
                cl::desc("Generate line information for generated GPU "
                         "kernels. (default=false)"));

/// Enable stack bounds checking. Passed directly to 'ptxas', which
/// will automatically turn this on if 'device-debug' or 'opt-level=0'
/// is used.
static cl::opt<bool>
    BoundsCheck("cuabi-bounds-check", cl::init(false), cl::NotHidden,
                cl::desc("Enable GPU static pointer bounds checking. "
                         "(default=false)"));

/// Set the optimization level for the compiler.  Values can be 0, 1,
/// 2, or 3; following standard compiler practice. Passed directly to
/// 'ptxas' (does not necessarily need to align with the main compiler
/// flags).
static cl::opt<unsigned>
    OptLevel("cuabi-opt-level", cl::init(3), cl::NotHidden,
             cl::desc("Specify the GPU kernel optimization level."));

/// Enable an extra set of passes over the host-side code after the
/// code has been transformed (e.g., loops replaced with kernel launch
/// calls).
static cl::opt<bool> RunHostPostOpt(
    "cuabi-run-post-opts", cl::init(false), cl::NotHidden,
    cl::desc("Run an additional, post transform, optimization pass."));

/// Enable expensive optimizations to allow the compiler to use the
/// maximum amount of resources (memory and time).  This will follow
/// the behavior of 'ptxas' -- if the optimization level is >= 2 this
/// is enabled.
static cl::opt<bool>
    AllowExpensiveOpts("cuabi-enable-expensive-optimizations", cl::init(false),
                       cl::Hidden,
                       cl::desc("Enable expensive optimizations that use "
                                "maximum available resources."));

/// Disable the generation of floating point multiply add instructions.
/// This is passed on to 'ptxas' to disable the contraction of floating
/// point multiply-add operations (FMAD, FFMA, or DFMA).  This is
/// equivalent to passing '-fmad false' to 'ptxas'.
static cl::opt<bool>
    DisableFMA("cuabi-disable-fma", cl::init(false), cl::Hidden,
               cl::desc("Disable the generation of FMA instructions."
                        "(default=false)"));

/// Disable the optimizer's constant bank optimizations.  Passed directly
/// to 'ptxas' as '--disable-optimizer-constants'.
static cl::opt<bool>
    DisableConstantBank("cuabi-disable-constant-bank", cl::init(false),
                        cl::Hidden,
                        cl::desc("Disable the use of the constants bank in "
                                 "GPU code generation. (default=false)"));

/// Set the CUDA ABI's default grain size value.  This is used internally
/// by the transform.
static cl::opt<unsigned> DefaultGrainSize(
    "cuabi-default-grainsize", cl::init(1), cl::Hidden,
    cl::desc("The default grainsize used by the transform "
             "when analysis fails to determine one. (default=1)"));

/// Keep the complete set of intermediate files around after compilation.  This
/// includes LLVM IR, PTX, and the fatbinary file.
static cl::opt<bool> KeepIntermediateFiles(
    "cuabi-keep-files", cl::init(false), cl::Hidden,
    cl::desc("Keep all the intermediate files on disk after"
             "successsful completion of the transforms "
             "various steps."));

/// Generate code to prefetch data prior to kernel launches.  This is literally
/// in the few lines right before a launch so obviously less than ideal.
static cl::opt<bool>
    CodeGenPrefetch("cuabi-prefetch", cl::init(true), cl::Hidden,
                    cl::desc("Enable generation of calls to do data "
                             "prefetching for UVM-based kernel  "
                             "parameters."));

/// Generate prefetch and kernel launch code as a combined stream of
/// operations.
static cl::opt<bool>
    CodeGenStreams("cuabi-streams", cl::init(false), cl::Hidden,
                   cl::desc("Generate prefetch and kernel launches "
                            "as a combined set of stream operations."));

/// Should PTX files be included in the fat binary images produced by the
/// transform?
static cl::opt<bool>
    EmbedPTXInFatbinaries("cuabi-embed-ptx", cl::init(false), cl::Hidden,
                          cl::desc("Embed intermediate PTX files in the "
                                   "fatbinaries used by the CUDA ABI "
                                   "transformation."));

/// Provide a hard-coded default value for the number of threads per block to
/// use in kernel launches.  This provides a compile-time mechanisms for
/// setting this value and it will persist throughout the execution of the
/// associated compilation unit(s).  The runtime internally currently uses the
/// equations,
///
///  ``unsigned blockSize = 4 * warpSize;``
///  ``blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;``
///
/// to determine the overall set of launch parameters.  This is mostly meant
/// for experimentation and testing.
static cl::opt<unsigned>
    DefaultThreadsPerBlock("cuabi-threads-per-block", cl::init(256), cl::Hidden,
                           cl::desc("Set the runtime system's value for "
                                    "the default number of threads per block. "
                                    "(default=256)"));

static cl::opt<unsigned>
    DefaultBlocksPerGrid("cuabi-blocks-per-grid", cl::init(0), cl::Hidden,
                         cl::desc("Hard-code the runtime system's value for "
                                  "the number of blocks per grid in kernel "
                                  "launches. (default=0=disabled)"));

/*
 * TODO: work here needs to be done to support these additional arguments
 * to expose more of the ptxas feature set within the ABI transform.
 *
 *
 /// Specify the maximum number of registers that GPU functions can use.
 /// Until a function-specific limit, a higher value will generally
 /// increase the performance of individual GPU threads that execute this
 /// function. However, because thread registers are allocated from a
 /// global register pool on each GPU, a higher value of this option
 /// will also reduce the maximum thread block size, thereby reducing the
 /// amount of thread parallelism.
 static cl::opt<int>
 MaxRegCount("cuabi-maxregcount", cl::init(-1), cl::Hidden,
             cl::desc("Specify the max number t of registers that GPU functions
 " "can use."));
 /// Control the aggressiveness of optimizations that affect register usage.
 /// ([0..10], default = 5) Higher values aggressively optimize the source
 /// program, trading off additional register usage for potential
 /// improvements in the generated code. Lower values inhibit optimizations
 /// that aggressively increase register usage. This option can work in
 /// conjunction with -maxrregcount and kernel launch bounds. (NVIDIA states:
 /// This is a BETA feature for advanced users and there is no guarantee that
 /// the implementation stays consistent between ptxas releases.)
 static cl::opt<unsigned>
 RegUsageLevel("cuabi-register-usage-level", cl::init(5), cl::Hidden,
               cl::desc("Experimental feature -- see 'ptxas' documentation. "
                        "(default=5)"));
*/

// Adapted from Transforms/Utils/ModuleUtils.cpp
// TODO: Technically we only use this to add a global ctor for
// dealing with the nuances of CUDA kernels so perhaps we'd be
// better off renaming this to match our specific use case?
static void appendToGlobalArray(const char *Array, Module &M, Constant *C,
                                int Priority, Constant *Data) {

  IRBuilder<> IRB(M.getContext());
  FunctionType *FnTy = FunctionType::get(IRB.getVoidTy(), false);

  // Get the current set of static global constructors and add
  // the new ctor to the list.
  SmallVector<Constant *, 16> CurrentCtors;
  StructType *EltTy = StructType::get(
      IRB.getInt32Ty(), PointerType::getUnqual(FnTy), IRB.getInt8PtrTy());
  if (GlobalVariable *GVCtor = M.getNamedGlobal(Array)) {
    if (Constant *Init = GVCtor->getInitializer()) {
      unsigned N = Init->getNumOperands();
      CurrentCtors.reserve(N + 1);
      for (unsigned i = 0; i != N; ++i)
        CurrentCtors.push_back(cast<Constant>(Init->getOperand(i)));
    }
    GVCtor->eraseFromParent();
  }

  // Build a 3 field global_ctor entry.  We don't take a comdat key.
  Constant *CSVals[3];
  CSVals[0] = IRB.getInt32(Priority);
  CSVals[1] = C;
  CSVals[2] = Data ? ConstantExpr::getPointerCast(Data, IRB.getInt8PtrTy())
                   : Constant::getNullValue(IRB.getInt8PtrTy());
  Constant *RuntimeCtorInit =
      ConstantStruct::get(EltTy, makeArrayRef(CSVals, EltTy->getNumElements()));

  CurrentCtors.push_back(RuntimeCtorInit);

  // Create a new initializer.
  ArrayType *AT = ArrayType::get(EltTy, CurrentCtors.size());
  Constant *NewInit = ConstantArray::get(AT, CurrentCtors);

  // Create the new global variable and replace all uses of
  // the old global variable with the new one.
  (void)new GlobalVariable(M, NewInit->getType(), false,
                           GlobalValue::AppendingLinkage, NewInit, Array);
}

/// Take the NVIDIA CUDA 'sm_' architecture format and convert it into
/// the 'compute_' form.  Note that we require CUDA 11 or greater and
/// we have removed support for sm_2x and sm_3x architectures.

static std::string virtualArchForCudaArch(StringRef Arch) {
  // TODO: We've scaled back some from the full suite of Nvidia targets
  // as we are going in assuming we will support only CUDA 11 or greater.
  // We should probably raise an error for sm_2x and sm_3x targets.
  return llvm::StringSwitch<std::string>(Arch)
      // sm_20 (Fermi) is deprecated as of CUDA 9.
      // sm_3X (Kepler) is deprecated as of CUDA 11.
      .Case("sm_50", "compute_50") // Maxwell (to be deprecated w/ CUDA 12?)
      .Case("sm_52", "compute_52") //
      .Case("sm_53", "compute_53") //
      .Case("sm_60", "compute_60") // Pascal
      .Case("sm_61", "compute_61") //
      .Case("sm_62", "compute_62") //
      .Case("sm_70", "compute_70") // Volta
      .Case("sm_72", "compute_72") //
      .Case("sm_75", "compute_75") // Turing
      .Case("sm_80", "compute_80") // Ampere
      .Case("sm_86", "compute_86") //
      .Case("sm_87", "compute_87") //
      .Default("unknown");
}

static std::string PTXVersionFromCudaVersion() {
#ifdef CUDATOOLKIT_VERSION
  Twine CudaVersionStr = Twine(CUDATOOLKIT_VERSION_MAJOR) + Twine(".") +
                         Twine(CUDATOOLKIT_VERSION_MINOR);
#else
#pragma message("warning, no CUDA Toolkit version info available.")
  Twine CudaVersionStr = "unknown";
#endif

  return llvm::StringSwitch<std::string>(CudaVersionStr.str())
      // TODO: These CUDA to PTX version translations will have
      // to be watched between CUDA and LLVM resources.  It is
      // not uncommon for LLVM to lag well behind CUDA PTX versions.
      // The details below are based on Cuda 11.6 and LLVM 13.x.
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
      .Case("11.6", "+ptx72") // TODO: should be at best ptx76.
      .Case("11.7", "+ptx72") // TODO: should be at best ptx77.
      .Case("11.8", "+ptx72") // TODO: should be at best ptx78.
      .Default("+ptx72");
}

// Some named values to make optimization levels a bit
// easier to read in the code.
const unsigned OptLevel0 = 0;
const unsigned OptLevel1 = 1;
const unsigned OptLevel2 = 2;
const unsigned OptLevel3 = 3;
// Code size.
const unsigned SizeLevel0 = 0;
const unsigned SizeLevel1 = 1;
const unsigned SizeLevel2 = 2;

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
//  The NVVMReflect pass replaces conditionals with constants and it will
//  often leave behind dead code. Therefore, it is recommended that
//  NVVMReflect be executed early in the optimization pipeline before
//  dead-code elimination.  The NVPTX TargetMachine knows how to schedule
//  NVVMReflect at the beginning of your pass manager; just use the following
//  code when setting up your pass manager:
//
//    std::unique_ptr<TargetMachine> TM = ...;
//    PassManagerBuilder PMBuilder(...);
//    if (TM)
//      TM->adjustPassManager(PMBuilder);
//

// Helper function to configure the details of our post-Tapir transformation
// passes.

/// Add a set of customized optimization passes over the LLVM code
/// post Tapir transformation but prior to conversion to PTX.  The
/// 'OptLevel' parameter controls the compiler optimization level
/// for performance, while 'SizeLevel' controls the size optimization
/// target.
static void runKernelOptimizationPasses(Module &KM,
                                        unsigned OptLevel = OptLevel3,
                                        unsigned SizeLevel = OptLevel2) {
  // TODO: Need to spend some time exploring the selected set of passes here.
  // assert(0 && "Need to switch to new pass manager");
  if (OptLevel > 0) {
    LLVM_DEBUG(dbgs() << "\tcuabi: optimizing generated kernel module...\n");
    legacy::PassManager PM;
    PM.add(createReassociatePass());
    PM.add(createGVNPass());
    PM.add(createCFGSimplificationPass());
    PM.add(createDeadStoreEliminationPass());
    PM.add(createCFGSimplificationPass());
    PM.add(createVerifierPass());
    PM.run(KM);
    LLVM_DEBUG(dbgs() << "\t\tpasses (+verifier) complete.\n");
  }
}

// NOTE: The NextKernelID variable below is not thread safe.
// This currently isn't an issue but if LLVM at some point
// in the future starts supporting multiple compilation threads
// (e.g. over Modules for example) the support code for kernel
// IDs will have a race...

/// Static ID for kernel naming -- each encountered kernel (loop)
/// during compilation will receive a unique ID.
unsigned CudaLoop::NextKernelID = 0;

CudaLoop::CudaLoop(Module &M, Module &KM, const std::string &KN, CudaABI *T,
                   bool MakeUniqueName)
    : LoopOutlineProcessor(M, KM), TTarget(T), KernelName(KN),
      KernelModule(KM) {

  if (MakeUniqueName) {
    std::string UN = KN + "_" + Twine(NextKernelID).str();
    NextKernelID++;
    KernelName = UN;
  }

  LLVM_DEBUG(dbgs() << "cuabi: creating cuda loop outliner:\n"
                    << "\tkernel name: " << KernelName << "\n"
                    << "\tmodule     : " << KernelModule.getName() << "\n\n");

  LLVMContext &Ctx = KernelModule.getContext();
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

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
  // These are a layer deeper than the interface used by the GPUABI.  While we
  // could codegen straight to the Cuda (Driver) API, the higher level calls
  // help to simplify codegen calls.
  KitCudaLaunchFn = M.getOrInsertFunction("__kitrt_cuLaunchKernel",
                                          VoidTy,       // no return
                                          VoidPtrTy,    // fat-binary
                                          VoidPtrTy,    // kernel name
                                          VoidPtrPtrTy, // arguments
                                          Int64Ty,      // trip count
                                          VoidPtrTy);   // stream

  KitCudaModuleLaunchFn = M.getOrInsertFunction("__kitrt_cuLaunchModuleKernel",
                                                VoidTy,       // no return
                                                VoidPtrTy,    // CUDA module
                                                VoidPtrTy,    // kernel name
                                                VoidPtrPtrTy, // arguments
                                                Int64Ty,      // trip count
                                                VoidPtrTy);   // stream

  KitCudaSyncFn = M.getOrInsertFunction("__kitrt_cuSynchronizeStreams",
                                        VoidTy); // no return & no parameters

  // Interface to runtime's prefetching support.
  KitCudaMemPrefetchFn =
      M.getOrInsertFunction("__kitrt_cuMemPrefetch", // on default stream
                            VoidTy,                  // no return.
                            VoidPtrTy);              // pointer to prefetch

  KitCudaStreamMemPrefetchFn =
      M.getOrInsertFunction("__kitrt_cuStreamMemPrefetch", // create new stream.
                            VoidPtrTy,  // corresponding stream.
                            VoidPtrTy); // pointer to prefetch.

  KitCudaMemPrefetchOnStreamFn =
      M.getOrInsertFunction("__kitrt_cuMemPrefetchOnStream", // on given stream.
                            VoidTy,                          // no return.
                            VoidPtrTy,  // pointer to prefetch.
                            VoidPtrTy); // run in this stream.

  KitCudaCreateFBModuleFn =
      M.getOrInsertFunction("__kitrt_cuCreateFBModule", VoidPtrTy, VoidPtrTy);
  KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_cuGetGlobalSymbol",
                            Int64Ty,    // return the device pointer for symbol.
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // CUDA module

  KitCudaMemcpySymbolToDeviceFn =
      M.getOrInsertFunction("__kitrt_cuMemcpySymbolToDevice",
                            VoidTy,   // returns
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
  LLVM_DEBUG(dbgs() << "\tcuabi: preprocessing parallel loop for kernel '"
                    << KernelName << "', in module '" << KernelModule.getName()
                    << "'.\n");

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
      // we can copy symbol information over from the host.
      GlobalValue::LinkageTypes LinkType;

      if (GV->hasInitializer())
        LinkType = GlobalValue::InternalLinkage;
      else
        LinkType = GlobalValue::ExternalLinkage;
      GlobalVariable *NewGV = nullptr;
      // If GV is a constant we can clone the entire
      // variable over, including the initalizer
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
        LLVM_DEBUG(dbgs() << "\tanalyzing missing (device-side) function '"
                          << F->getName() << "'.\n");
        Function *LF = resolveLibDeviceFunction(F);
        if (LF && not KernelModule.getFunction(LF->getName())) {
          LLVM_DEBUG(dbgs() << "\ttransformed to libdevice function '"
                            << LF->getName() << "'.\n");
          DeviceF = Function::Create(LF->getFunctionType(), F->getLinkage(),
                                     LF->getName(), KernelModule);
        } else {
          LLVM_DEBUG(dbgs() << "\tcreated device function '" << F->getName()
                            << "'.\n");
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
  KernelF->removeFnAttr("personality");
  KernelF->addFnAttr("target-cpu", GPUArch);
  KernelF->addFnAttr("target-features",
                     PTXVersionFromCudaVersion() + "," + GPUArch);
  NamedMDNode *Annotations =
      KernelModule.getOrInsertNamedMetadata("nvvm.annotations");
  SmallVector<Metadata *, 3> AV;
  AV.push_back(ValueAsMetadata::get(KernelF));
  AV.push_back(MDString::get(Ctx, "kernel"));
  AV.push_back(
      ValueAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), 1)));
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
    Grainsize = ConstantInt::get(PrimaryIV->getType(), 1);
    // DefaultGrainSize.getValue());
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
/*
static Function *getVprintfDeclaration(llvm::Module &M) {
  Type *ArgTypes[] = {Type::getInt8PtrTy(M.getContext()),
                      Type::getInt8PtrTy(M.getContext())};
  FunctionType* VprintFnTy = Function::get(Type::getInt32Ty(M.getContext()),
                                           ArgTypes, false);
  if (auto *F = M.getFunction("vprintf")) {
    assert(F->getFunctionType() == VprintFnTy);
    return F;
  }

  return Function::Create(VprintFnTy,
                          GlobalVariable::ExternalLinkage,
                          "vprintf", &M);
}

// Transform what was a host-side call to printf() into a call to
// the PTX vprintf() system call.  vprintf() takes two arguments,
// a format string (a la printf) and a pointer to a buffer containing
// the varargs.  So the transformation is from:
//
//   printf("string %s is %d characters long.\n", str, len);
//
// to
//
//   struct Vargs {
//     Arg1 a1;
//     Arg2 a2;
//     Arg3 a3;
//   };
//   char *VArgBuffer = alloca(sizeof(Vargs);
//   *(Vargs*)VArgBuffer = {a1, a2, a3};
//   vprintf("string %s is %d characters long.\n", VArgBuffer);
//
// The VArgBuffer should be aligned to the max of the arguments and each
// argument should be aligned to its own preferred alignment.
//
/*
Value *CudaLooop::emitPrintfCall(const Function *PrintFn) {
  unsigned ArgCount = 0;
  SmallVector<Type*, 8> ArgTypes;
  for(auto Arg = PrintFn->arg_begin(); Arg != PrintFn->arg_end(); ++Arg) {
    ArgTypes.push_back(Arg->getType());
    ArgCount++;
  }

  Value *ArgBufferPtr;
  if (ArgCount <= 1) {
    ArgBufferPtr = ConstantPointerNull::get(Type::getInt8PtrTy(Ctx));
  } else {
    Type *ArgTy = StructType::create(ArgTypes, "vprintf_args");
    Value *ArgArray = Builder.CreateAlloca(ArgTy);
    for(int i = 1; i < ArgCount; i++) {
      Value *P = Builder.CreateStructGEP(ArgTy, ArgArray, i - 1);
      Value *Arg = Arg->;
      Builder.CreateAlignedStore(Arg, P, DL.getPrefTypeAlign(Arg->getType()));
    }
    ArgBufferPtr = Builder.CreatePointerCast(ArgArray, Type::getInt8PtrTy(Ctx));
  }
}
*/

Function *CudaLoop::resolveLibDeviceFunction(Function *Fn) {
  std::unique_ptr<Module> &LDM = TTarget->getLibDeviceModule();
  const std::string NVPrefix = "__nv_";

  // Handle special cases where code generation can be a bit more
  // complex; e.g., printf().
  if (Fn->getName() == "printf" || Fn->getName() == "fprintf") {
    report_fatal_error("cuabi: printf is currently unsupported "
                       "in parallel loops... :-(\n");
  }

  std::string FnName;

  // Are we dealing with an intrinsic like those generated by -ffast-math?
  if (Fn->isIntrinsic()) {
    if (Fn->getName().str().compare(0, 9, "llvm.nvvm") == 0)
      return nullptr;
    else if (Fn->getName() == "llvm.sqrt.f32")
      FnName = "sqrtf";
    else if (Fn->getName() == "llvm.sqrt.f64")
      FnName = "sqrt";
    else if (Fn->getName() == "llvm.cos.f32")
      FnName = "fast_cosf";
    else if (Fn->getName() == "llvm.cos.f64")
      FnName = "cos";
    else if (Fn->getName() == "llvm.sin.f32")
      FnName = "fast_sinf";
    else if (Fn->getName() == "llvm.sin.f64")
      FnName = "sin";
    else if (Fn->getName() == "llvm.tan.f32")
      FnName = "fast_tanf";
    else if (Fn->getName() == "llvm.tan.f64")
      FnName = "tan";
    else {
      // report_fatal_error("cuabi: no transform for llvm intrinsic!");
      return nullptr;
    }
  } else
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

  LLVM_DEBUG(dbgs() << "Transforming function '" << F.getName() << "' "
                    << "in preparation for PTX generation.\n");

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
  LLVM_DEBUG(
      dbgs() << "cuabi: search for unresolved calls in outlined kernel...\n");
  std::list<CallInst *> Replaced;
  for (auto I = inst_begin(&F); I != inst_end(&F); I++) {
    if (auto CI = dyn_cast<CallInst>(&*I)) {
      Function *CF = CI->getCalledFunction();
      if (CF->size() == 0) {
        Function *DF = resolveLibDeviceFunction(CF);
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

  LLVM_DEBUG(dbgs() << "\tprocessing outlined loop call for kernel '"
                    << KernelName << "' w/ " << OrderedInputs.size()
                    << " arguments.\n");

  LLVMContext &Ctx = M.getContext();
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);

  Function *Parent = TOI.ReplCall->getFunction();
  Value *TripCount = OrderedInputs[0];
  BasicBlock *RCBB = TOI.ReplCall->getParent();
  BasicBlock *NBB = RCBB->splitBasicBlock(TOI.ReplCall);
  TOI.ReplCall->eraseFromParent();

  IRBuilder<> B(&NBB->front());
  Function &F = *KernelModule.getFunction(KernelName.c_str());
  transformForPTX(F);

  BasicBlock &EBB = Parent->getEntryBlock();
  IRBuilder<> EB(&EBB.front());

  ArrayType *ArrayTy = ArrayType::get(VoidPtrTy, OrderedInputs.size());
  Value *ArgArray = EB.CreateAlloca(ArrayTy);
  unsigned int i = 0;
  Value *prefetchStream = nullptr;
  if (not CodeGenStreams) {
    // If we are going to use the default stream we set the main prefetch stream
    // to null and it will propagate through all prefetch and the final launch
    // call.
    prefetchStream = ConstantPointerNull::get(VoidPtrTy);
  }

  for (Value *V : OrderedInputs) {
    Value *VP = EB.CreateAlloca(V->getType());
    B.CreateStore(V, VP);
    Value *VoidVPtr = B.CreateBitCast(VP, VoidPtrTy);
    Value *ArgPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, i);
    B.CreateStore(VoidVPtr, ArgPtr);
    i++;

    if (CodeGenPrefetch) { // TODO: Only for >= 2 opt level?
      Type *VT = V->getType();
      if (VT->isPointerTy()) {
        Value *VoidPP = B.CreateBitCast(V, VoidPtrTy);
        if (prefetchStream == nullptr) { // stream codegen enabled...
          LLVM_DEBUG(dbgs() << "creating initial prefetch stream.\n");
          prefetchStream = B.CreateCall(KitCudaMemPrefetchFn, {VoidPP},
                                        "_cuabi.prefetch_stream");
        } else {
          LLVM_DEBUG(dbgs() << "code gen prefetch.\n");
          B.CreateCall(KitCudaMemPrefetchOnStreamFn, {VoidPP, prefetchStream});
        }
      }
    }
  }

  const DataLayout &DL = M.getDataLayout();
  Value *argsPtr = B.CreateConstInBoundsGEP2_32(ArrayTy, ArgArray, 0, 0);
  // Generate a call to launch the kernel.
  Constant *KNameCS = ConstantDataArray::getString(Ctx, KernelName);
  GlobalVariable *KNameGV =
      new GlobalVariable(M, KNameCS->getType(), true,
                         GlobalValue::PrivateLinkage, KNameCS, ".str");
  KNameGV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  Type *StrTy = KNameGV->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(StrTy), 0),
                       ConstantInt::get(DL.getIndexType(StrTy), 0)};
  Constant *KNameParam =
      ConstantExpr::getGetElementPtr(KNameGV->getValueType(), KNameGV, Zeros);

  // We can't get to the complete fat binary data until all loops in the
  // input module have been processed (i.e., the complete kernel module is
  // poplated, converted to PTX, turned into an assembled binary, etc.).
  // Because of this we create a "stand in" (dummy) here and will replace
  // it later in the ABI's transformaiton pipeline.
  Constant *DummyFBGV =
      tapir::getOrInsertFBGlobal(M, "_cuabi.dummy_fatbin", VoidPtrTy);
  Value *DummyFBPtr = B.CreateLoad(VoidPtrTy, DummyFBGV);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Value *TCCI = nullptr;
  if (TripCount->getType() != Int64Ty) {
    TCCI = CastInst::CreateIntegerCast(TripCount, Int64Ty, false);
    B.Insert(TCCI, "tcci");
  } else
    TCCI = TripCount; // Simplify cases in launch code gen below...

  if (not TTarget->hasGlobalVariables()) {
    LLVM_DEBUG(dbgs() << "\tcreating kernel launch (no globals).\n");
    B.CreateCall(KitCudaLaunchFn,
                 {DummyFBPtr, KNameParam, argsPtr, TCCI, prefetchStream});
  } else {
    LLVM_DEBUG(dbgs() << "\tcreating kernel launch (w/ globals).\n");
    Value *CuModule = B.CreateCall(KitCudaCreateFBModuleFn, {DummyFBPtr});
    B.CreateCall(KitCudaModuleLaunchFn,
                 {CuModule, KNameParam, argsPtr, TCCI, prefetchStream});
  }
}

CudaABI::CudaABI(Module &M)
    : TapirTarget(M),
      KM(Twine(CUABI_PREFIX + sys::path::filename(M.getName())).str(),
         M.getContext()) {

  LLVM_DEBUG(dbgs() << "cuabi: creating tapir target for module '"
                    << M.getName() << "' (w/ kernel module: '" << KM.getName()
                    << "')\n");

  // Create a module (KM) to hold all device side functions for all parallel
  // constructs in the module we are processing (M). At present a loop processor
  // will be created for each construct and is then responsible for the steps
  // required to prepare the "kernel" module (KM) for code generation to PTX.

  // Build the details required to have a PTX code generation path ready to go
  // at completion of the module processing; see postProcessModule() for when
  // that stage is kicked off via the Tapir layer.
  std::string ArchString;
  if (HostMArch == "64")
    ArchString = "nvptx64";
  else
    ArchString = "nvptx";
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
      Reloc::PIC_, CodeModel::Large, CodeGenOpt::Aggressive);

  KM.setTargetTriple(TT.str());
  KM.setDataLayout(PTXTargetMachine->createDataLayout());

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
    LLVMContext &Ctx = KM.getContext();
    llvm::SMDiagnostic SMD;
    std::optional<std::string> CudaPath = sys::Process::FindInEnvPath(
        "CUDA_HOME", "nvvm/libdevice/libdevice.10.bc");
    if (!CudaPath) {
      CudaPath = sys::Process::FindInEnvPath("CUDA_PATH",
                                             "nvvm/libdevice/libdevice.10.bc");
      if (!CudaPath)
        report_fatal_error("Unable to load cuda libdevice.10.bc!");
    }

    LibDeviceModule = parseIRFile(*CudaPath, SMD, Ctx);
    if (not LibDeviceModule)
      report_fatal_error("Failed to parse cuda libdevice.10.bc!");
  }

  return LibDeviceModule;
}

Value *CudaABI::lowerGrainsizeCall(CallInst *GrainsizeCall) {
  // TODO: The grainsize on the GPU is a completely different beast
  // than the CPU cases Tapir was originally designed for.  At present
  // keeping the grainsize at 1 has almost always shown to yeild the
  // best results in terms of performance.  We have yet to really do
  // a detailed study of the aspects here so consider anything done
  // here as a lot of remaining work and exploration.
  Value *Grainsize =
      ConstantInt::get(GrainsizeCall->getType(), DefaultGrainSize);
  // Replace uses of grainsize intrinsic call with a computed grainsize value.
  GrainsizeCall->replaceAllUsesWith(Grainsize);
  // TODO: ??? GrainsizeCall->eraseFromParent();
  return Grainsize;
}

void CudaABI::lowerSync(SyncInst &SI) {
  // no-op...
  // The CUDA transformations split the code into device and host
  // side modules.
}

void CudaABI::addHelperAttributes(Function &F) { /* no-op */
}

void CudaABI::preProcessFunction(Function &F, TaskInfo &TI,
                                 bool OutliningTapirLoops) { /* no-op */
}

void CudaABI::postProcessFunction(Function &F, bool OutliningTapirLoops) {
  if (OutliningTapirLoops) {
    LLVMContext &Ctx = M.getContext();
    Type *VoidTy = Type::getVoidTy(Ctx);
    FunctionCallee KitCudaSyncFn =
        M.getOrInsertFunction("__kitrt_cuSynchronizeStreams",
                              VoidTy); // no return & no parameters

    for (Value *SR : SyncRegList) {
      for (Use &U : SR->uses()) {
        if (auto *SyncI = dyn_cast<SyncInst>(U.getUser()))
          CallInst::Create(KitCudaSyncFn, "",
                           &*SyncI->getSuccessor(0)->begin());
      }
    }
    SyncRegList.clear();
  }
}

void CudaABI::postProcessHelper(Function &F) { /* no-op */
}

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

  LLVM_DEBUG(dbgs() << "\tassembling PTX file '" << PTXFile->getFilename()
                    << "'.\n");

  std::error_code EC;
  auto PTXASExe = sys::findProgramByName("ptxas");
  if ((EC = PTXASExe.getError()))
    report_fatal_error("'ptxas' not found. "
                       "Is a CUDA installation in your path?");

  // We'll create an "assembled" PTX file that is named the
  // same as the LLVM module we've just processed.
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
  PTXASArgList.push_back(PTXASExe->c_str());

  // TODO: Do we need/want to add support for generating relocatable code?
  // PTXASArgList.push_back("-c");

  // --machine <bits>: Specify 32-bit vs. 64-bit host architecture.
  PTXASArgList.push_back("--machine"); // host (32- vs. 64-bit)
  PTXASArgList.push_back(HostMArch.c_str());
  // --gpu-name <gpu name>: Specify name of GPU to generate code for.
  // (e.g., 'sm_70','sm_72','sm_75','sm_80','sm_86', 'sm_87')
  PTXASArgList.push_back("--gpu-name"); // target gpu architecture.
  PTXASArgList.push_back(GPUArch.c_str());
  if (Verbose) {
    // --verbose: prints code generation statistics.
    PTXASArgList.push_back("--verbose");
    PTXASArgList.push_back("-warn-spills");
  }
  if (Debug) {
    // TODO: It currently isn't possible to use both debug and an optimization
    // flags with ptxas -- need to check for this case.  Doing so will lead to
    // spew from our ptxas invocation.
    PTXASArgList.push_back("--device-debug");
  }

  if (OptLevel > 3) {
    errs() << "warning -- cuda abi transform: "
           << "unknown optimization level.\n"
           << "\twill use level 3 instead.\n";
    OptLevel = 3;
  }

  if (not Debug) {
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
      break;
    default:
      llvm_unreachable_internal("unhandled/unexpected optimization level",
                                __FILE__, __LINE__);
      break;
    }
  }

  if (OptLevel < 2 && AllowExpensiveOpts) {
    PTXASArgList.push_back("--allow-expensive-optimizations");
    PTXASArgList.push_back("true");
  }
  if (DisableFMA) {
    PTXASArgList.push_back("--fmad");
    PTXASArgList.push_back("false");
  }
  if (DisableConstantBank)
    PTXASArgList.push_back("--disable-optimizer-constants");

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
  LLVM_DEBUG(dbgs() << "\tptxas command line:\n";
             unsigned c = 0; for (auto dbg_arg
                                  : PTXASArgs) {
               dbgs() << "\t\t" << c << ": " << dbg_arg << "\n";
               c++;
             } dbgs() << "\n\n";);

  // Finally we are ready to execute ptxas...
  std::string ErrMsg;
  bool ExecFailed;
  int ExecStat = sys::ExecuteAndWait(*PTXASExe, PTXASArgs, std::nullopt, {},
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

  LLVM_DEBUG(dbgs() << "\tpatching kernel launch calls...\n");

  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  // Look up a global (device-side) symbol via a module
  // created from the fat binary.
  FunctionCallee KitCudaGetGlobalSymbolFn =
      M.getOrInsertFunction("__kitrt_cuGetGlobalSymbol",
                            Int64Ty,    // device pointer
                            CharPtrTy,  // symbol name
                            VoidPtrTy); // CUDA module

  FunctionCallee KitCudaMemcpyToDeviceFn =
      M.getOrInsertFunction("__kitrt_cuMemcpySymbolToDevice",
                            VoidTy,    // returns
                            VoidPtrTy, // host ptr
                            Int64Ty,   // device ptr
                            Int64Ty);  // num bytes

  // There are two forms of kernel launch we need to search for.  The first
  // is a kernel launch without any global variables in use.  In this case
  // we have a simple replacement of the first parameter with the now complete
  // fat binary.
  //
  // The second case is a kernel launch with globals.  In this case, we need to
  // find the corresponding global within the fat binary and then issue a copy
  // of the host side data to the device (prior to the kernel launch).
  // Therefore this path is bit more complex as we have to find the creation of
  // the CUDA module that requires the updated fat binary, then fetch the
  // device pointer for each global, issue a corresponding memcpy, and then
  // launch the kernel.
  auto &FnList = M.getFunctionList();
  for (auto &Fn : FnList) {
    for (auto &BB : Fn) {
      for (auto &I : BB) {

        if (CallInst *CI = dyn_cast<CallInst>(&I)) {
          if (Function *CFn = CI->getCalledFunction()) {
            if (CFn->getName().startswith("__kitrt_cuLaunchKernel")) {
              LLVM_DEBUG(dbgs() << "\t\t* patching launch: " << *CI << "\n");
              Value *CFatbin;
              CFatbin = CastInst::CreateBitOrPointerCast(Fatbin, VoidPtrTy,
                                                         "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);
            } else if (CFn->getName().startswith("__kitrt_cuCreateFBModule")) {
              Value *CFatbin;
              CFatbin = CastInst::CreateBitOrPointerCast(Fatbin, VoidPtrTy,
                                                         "_cubin.fatbin", CI);
              CI->setArgOperand(0, CFatbin);

              Instruction *NI = CI->getNextNonDebugInstruction();
              // Unless someting else has monkeyed with our generated code
              // NI should be the launch call.  We need the following code
              // to go between the call instruction and the launch.
              assert(NI && "unexpected null instruction!");
              for (auto &HostGV : GlobalVars) {
                std::string DevVarName = HostGV->getName().str() + "_devvar";
                LLVM_DEBUG(dbgs() << "\t\t* processing global: "
                                  << HostGV->getName() << "\n");
                Value *SymName =
                    tapir::createConstantStr(DevVarName, M, DevVarName);
                Value *DevPtr =
                    CallInst::Create(KitCudaGetGlobalSymbolFn, {SymName, CI},
                                     ".cuabi_devptr", NI);
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

  GlobalVariable *ProxyFB = M.getGlobalVariable("_cuabi.dummy_fatbin", true);
  if (ProxyFB) {
    Constant *CFB =
        ConstantExpr::getPointerCast(Fatbin, VoidPtrTy->getPointerTo());
    LLVM_DEBUG(dbgs() << "\tcleaning up dummy fatbin global.\n");
    ProxyFB->replaceAllUsesWith(CFB);
    ProxyFB->eraseFromParent();
  } else {
    LLVM_DEBUG(dbgs() << "\t\tWARNING! "
                      << "whoopsie... unable to find proxy fatbin ptr!\n"
                      << "something might be broken...\n\n");
  }
}

CudaABIOutputFile CudaABI::createFatbinaryFile(CudaABIOutputFile &AsmFile) {
  std::error_code EC;
  SmallString<255> FatbinFilename(AsmFile->getFilename());
  sys::path::replace_extension(FatbinFilename, ".cufatbin");
  CudaABIOutputFile FatbinFile;
  FatbinFile = std::make_unique<ToolOutputFile>(FatbinFilename, EC,
                                                sys::fs::OpenFlags::OF_None);

  LLVM_DEBUG(dbgs() << "\tcreating fatbinary file '"
                    << FatbinFile->getFilename() << "'.\n");

  // TODO: LLVM docs suggest we shouldn't be using findProgramByName()...
  auto FatbinaryExe = sys::findProgramByName("fatbinary");
  if ((EC = FatbinaryExe.getError()))
    report_fatal_error("'fatbinary' not found. "
                       "Is a CUDA installation in your path?");

  opt::ArgStringList FatbinaryArgList;
  FatbinaryArgList.push_back(FatbinaryExe->c_str());

  if (HostMArch == "32")
    FatbinaryArgList.push_back("--32");
  else if (HostMArch == "64")
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
      sys::ExecuteAndWait(*FatbinaryExe, FatbinaryArgs, std::nullopt, {},
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
  LLVM_DEBUG(dbgs() << "\tread fat binary image, " << Fatbinary->getBufferSize()
                    << " bytes.\n");

  LLVMContext &Ctx = M.getContext();
  Type *Int8Ty = Type::getInt8Ty(Ctx);
  Constant *FatbinArray = ConstantDataArray::getRaw(
      StringRef(Fatbinary->getBufferStart(), Fatbinary->getBufferSize()),
      Fatbinary->getBufferSize(), Int8Ty);

  // Create a global variable to hold the fatbinary image.
  GlobalVariable *FatbinaryGV;
  FatbinaryGV = new GlobalVariable(M, FatbinArray->getType(), true,
                                   GlobalValue::PrivateLinkage, FatbinArray,
                                   "_cuabi_fatbin_ptr");

  // At this point the fatbinary is not in an adequate form for use within the
  // executable for the CUDA runtime nor the command line toolset that comes
  // with CUDA distributions (e.g., cuobjdump).  We complete the embedding in
  // embedFatbinary().
  return FatbinaryGV;
}

void CudaABI::bindGlobalVariables(Value *Handle, IRBuilder<> &B) {
  LLVMContext &Ctx = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *IntTy = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *VarSizeTy = Int64Ty;
  PointerType *CharPtrTy = Type::getInt8PtrTy(Ctx);

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
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  PointerType *VoidPtrPtrTy = VoidPtrTy->getPointerTo();
  Type *IntTy = Type::getInt32Ty(Ctx);

  Function *CtorFn = Function::Create(
      FunctionType::get(VoidTy, VoidPtrTy, false), GlobalValue::InternalLinkage,
      CUABI_PREFIX + ".ctor." + KM.getName(), &M);

  BasicBlock *CtorEntryBB = BasicBlock::Create(Ctx, "entry", CtorFn);
  IRBuilder<> CtorBuilder(CtorEntryBB);
  const DataLayout &DL = M.getDataLayout();

  // Tuck the call to initialize the Kitsune runtime into the constructor;
  // this in turn will initialized CUDA...
  FunctionCallee KitRTInitFn = M.getOrInsertFunction("__kitrt_cuInit", VoidTy);
  CtorBuilder.CreateCall(KitRTInitFn, {});

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
  Type *VoidPtrTy = Type::getInt8PtrTy(Ctx);
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
      M.getOrInsertFunction("__kitrt_cuDestroy", VoidTy);
  DtorBuilder.CreateCall(KitRTDestroyFn, {});

  DtorBuilder.CreateRetVoid();
  return DtorFn;
}

void CudaABI::registerFatbinary(GlobalVariable *Fatbinary) {
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
#define FATBIN_CONTROL_SECTION_NAME ".nvFatBinSegment"
#define FATBIN_DATA_SECTION_NAME ".nv_fatbin"

  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  PointerType *VoidPtrTy = Type::getInt8PtrTy(Ctx);
  Type *IntTy = Type::getInt32Ty(Ctx);

  const DataLayout &DL = M.getDataLayout();

  Type *FatbinStrTy = Fatbinary->getType();
  Constant *Zeros[] = {ConstantInt::get(DL.getIndexType(FatbinStrTy), 0),
                       ConstantInt::get(DL.getIndexType(FatbinStrTy), 0)};

  // TODO: The section name below is a complete shot in the dark.
  // The CUDA fatbinary header suggests, "The section that contains the fatbin
  // data itself (put in a separate section so it is easy to find".
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
  Wrapper->setAlignment(Align(DL.getPrefTypeAlignment(Wrapper->getType())));

  // The rest of the registration details are tucked into a constructor
  // entry...
  Function *CtorFn = createCtor(Fatbinary, Wrapper);
  if (CtorFn) {
    FunctionType *CtorFnTy = FunctionType::get(VoidTy, false);
    Type *CtorFnPtrTy =
        PointerType::get(CtorFnTy, M.getDataLayout().getProgramAddressSpace());
    appendToGlobalArray("llvm.global_ctors", M,
                        ConstantExpr::getBitCast(CtorFn, CtorFnPtrTy), 65536,
                        nullptr);
  }
}

CudaABIOutputFile CudaABI::generatePTX() {
  std::error_code EC;

  if (KeepIntermediateFiles) {
    std::unique_ptr<ToolOutputFile> IRFile;
    SmallString<255> IRFileName(KM.getName());
    sys::path::replace_extension(IRFileName, ".tapir");
    IRFile = std::make_unique<ToolOutputFile>(IRFileName, EC,
                                              sys::fs::OpenFlags::OF_None);
    KM.print(IRFile->os(), nullptr);
    IRFile->keep();
  }

  // Take the intermediate form code in the kernel module (KM) and
  // generate a PTX file.  The PTX file will be named the same as
  // the original input source module (M) with the extension changed
  // to PTX.
  std::string ModelPTXFileName =
      std::string(CUABI_PREFIX) + "%%-%%-%%_" + KM.getName().str();
  SmallString<1024> PTXFileName;
  sys::fs::createUniquePath(ModelPTXFileName.c_str(), PTXFileName, true);
  sys::path::replace_extension(PTXFileName, ".ptx");

  LLVM_DEBUG(dbgs() << "\tgenerating PTX file '" << PTXFileName << "'.\n");

  std::unique_ptr<ToolOutputFile> PTXFile;
  PTXFile = std::make_unique<ToolOutputFile>(PTXFileName, EC,
                                             sys::fs::OpenFlags::OF_None);
  PTXFile->keep();

  KM.addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz", true);

  if (OptLevel > 0) {
    if (OptLevel > 3) 
      OptLevel = 3;
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
    PassBuilder pb(PTXTargetMachine, pto);
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    PTXTargetMachine->registerPassBuilderCallbacks(pb);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    OptimizationLevel optLevels[] = {
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    };
    OptimizationLevel OptimizationLevel = optLevels[OptLevel];
    ModulePassManager mpm = pb.buildPerModuleDefaultPipeline(OptimizationLevel);
    mpm.addPass(VerifierPass());
    LLVM_DEBUG(dbgs() << "\t\t* module: " << KM.getName() << "\n");
    mpm.run(KM, mam);
  }

  // Setup the passes and request that the output goes to the
  // specified PTX file.
  legacy::PassManager PassMgr;
  if (PTXTargetMachine->addPassesToEmitFile(PassMgr, PTXFile->os(), nullptr,
                                            CodeGenFileType::CGFT_AssemblyFile,
                                            false))
    report_fatal_error("Cuda ABI transform -- PTX generation failed!");
  PassMgr.run(KM);
  return std::move(PTXFile);
}

void CudaABI::postProcessModule() {
  // At this point all suitable constructs in the module (M) have been processed
  // by the outliner(s).  We expect the kernel module (KM) to be populated with
  // PTX-ready LLVM form (see the CudaLoop processor for details).
  //
  // We now start the high-level stages for creating a CUDA executable target:
  //
  //   1. Generate and assmble PTX.
  //   2. Create a fat binary image and "inline" it into
  //      the host-side module (M).
  //   3. Register the fatbinary (and assocaited details)
  //      with the CUDA runtime layer(s).
  //   4. Clean up the host side code to refer to the
  //      fat binary content.
  //
  LLVM_DEBUG(dbgs() << "cuabi: post processing module '" << M.getName()
                    << "'\n");

  auto L = Linker(KM);
  if (LibDeviceModule) {
    LLVM_DEBUG(dbgs() << "cuabi: linking in cuda libdevice module.\n");
    L.linkInModule(std::move(LibDeviceModule), Linker::LinkOnlyNeeded);
  }

  CudaABIOutputFile PTXFile = generatePTX();
  CudaABIOutputFile AsmFile = assemblePTXFile(PTXFile);
  CudaABIOutputFile FatbinFile = createFatbinaryFile(AsmFile);
  GlobalVariable *Fatbinary = embedFatbinary(FatbinFile);

  finalizeLaunchCalls(M, Fatbinary);
  registerFatbinary(Fatbinary);
  if (RunHostPostOpt) {
    // assert(0 && "Need to switch to new pass manager");
    legacy::PassManager PM;
    legacy::FunctionPassManager FPM(&M);
    PassManagerBuilder PMB;
    PMB.Inliner = createFunctionInliningPass(OptLevel, 0, false);
    PMB.OptLevel = OptLevel;
    PMB.SizeLevel = 0; // No size optimizations.
    PMB.VerifyInput = 1;
    PMB.DisableUnrollLoops = false;
    PMB.LoopVectorize = true;
    PMB.SLPVectorize = true;
    PMB.populateFunctionPassManager(FPM);
    PMB.populateModulePassManager(PM);
    FPM.doInitialization();
    for (Function &Fn : M)
      FPM.run(Fn);
    FPM.doFinalization();
    PM.run(M);
  }

  if (!KeepIntermediateFiles) {
    sys::fs::remove(PTXFile->getFilename());
    sys::fs::remove(AsmFile->getFilename());
    sys::fs::remove(FatbinFile->getFilename());
  }
}

LoopOutlineProcessor *
CudaABI::getLoopOutlineProcessor(const TapirLoopInfo *TL) {
  // Create a CUDA loop outline processor for transforming parallel tapir loop
  // constructs into suitable GPU device code.  We hand the outliner the kernel
  // module (KM) as the destination for all generated (device-side) code.

  // std::string ModuleName = sys::path::filename(KM.getName()).str();
  // PTX dislikes names containing '.' -- replace them with
  // underscores.  TODO: Is this still true?
  // std::replace(ModuleName.begin(), ModuleName.end(), '.', '_');
  // std::replace(ModuleName.begin(), ModuleName.end(), '-', '_');

  Loop *TheLoop = TL->getLoop();
  Function *Fn = TheLoop->getHeader()->getParent();
  std::string KN = Fn->getName().str();

  if (M.getNamedMetadata("llvm.dbg.cu") || M.getNamedMetadata("llvm.dbg")) {
    // If we have debug info in the module go ahead and use a line number
    // based naming scheme for kernel names. This is purely for some extra
    // context (and sanity?) on the compiler development side...
    unsigned LineNumber = TL->getLoop()->getStartLoc()->getLine();
    KN = CUABI_PREFIX + KN + "_" + Twine(LineNumber).str();
  } else {
    // In the non-debug mode we use a consecutive numbering scheme for our
    // kernel names (this is currently handled via the 'make unique' parameter).
    KN = CUABI_PREFIX + KN;
  }

  CudaLoop *CLOP = new CudaLoop(M, KM, KN, this);
  return CLOP;
}
