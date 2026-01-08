//===- TapirTargetOptions.cpp - Options for the tapir targets -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for the TapirTargetOptions object. Also contains any command
// line options shared by some or all tapir targets.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Core/TapirTargetOptions.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Frontend/KitsuneOptions.h"
#include "kitsune/Support/OptznLevelUtils.h"
#include "kitsune/Support/ToString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Process.h"

#include <optional>

using namespace llvm;

// The options here are those that are used to initialize the TapirTargetOptions
// object *only*. Several tapir targets have additional command line options
// that can be used to tweak their behavior. Those are intended for
// experimentation. If any are deemed to be generally useful, they should be
// added here and a corresponding frontend option should be created for them.
//
// Some of the options here are also used by some Kitsune tools. In those
// cases, the tools create a TapirTargetOptions object, so these options remain
// static here. To keep the help messages

// -------------------- options common to all tapir targets --------------------

static cl::opt<TTID>
    clTapir("tapir", cl::desc("The primary tapir target"), cl::init(TTID::Nolo),
            cl::value_desc("target"), cl::cat(cl::catKitClOpts),
            cl::values(clEnumValN(TTID::Nolo, "nolo", ""),
                       clEnumValN(TTID::Serial, "serial", ""),
                       clEnumValN(TTID::Cuda, "cuda", ""),
                       clEnumValN(TTID::Hip, "hip", ""),
                       clEnumValN(TTID::OpenCilk, "opencilk", ""),
                       clEnumValN(TTID::GPU, "gpu", ""),
                       clEnumValN(TTID::Qthreads, "qthreads", ""),
                       clEnumValN(TTID::Realm, "realm", ""),
                       clEnumValN(TTID::Lambda, "lambda", ""),
                       clEnumValN(TTID::OMPTask, "omptask", ""),
                       clEnumValN(TTID::OpenMP, "openmp", "")));

/// This was the option originally in tapir, but in Kitsune, we prefer to use
/// --tapir instead.
static cl::alias clTapirTarget("tapir-target", cl::desc("Alias for --tapir"),
                               cl::aliasopt(clTapir),
                               cl::cat(cl::catKitClOpts));

static cl::opt<bool>
    clTapirVerbose("tapir-verbose", cl::init(false),
                   cl::desc("Enable verbose mode in all tapir targets"),
                   cl::cat(cl::catKitClOpts));

static cl::opt<bool>
    clKitrtVerbose("kitrt-verbose", cl::init(false),
                   cl::desc("Enable verbose mode in kitsune's runtime"),
                   cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clLLD("tapir-lld", cl::init(""),
                                  cl::desc("Path to LLD"),
                                  cl::cat(cl::catKitClOpts));

// ------------------ options common to the GPU tapir targets ------------------

static cl::opt<unsigned> clFixedThreadsPerBlock(
    "tapir-gpu-tpb", cl::init(0),
    cl::desc("Use a fixed number of threads per block for all GPU kernel "
             "launches unless overridden with pragmas. If this is not provided "
             "the threads per block will be calculated by Kitsune's runtime. "
             "Can be at most 1024"),
    cl::cat(cl::catKitClOpts));

static cl::opt<unsigned> clMaxThreadsPerBlock(
    "tapir-gpu-max-tpb", cl::init(0),
    cl::desc(
        "The maximum number of threads per block to launch. If this is not "
        "provided, Kitsune's runtime is free to launch as many threads per "
        "block as it sees fit"),
    cl::cat(cl::catKitClOpts));

static cl::opt<bool>
    clGPUPrefetch("tapir-gpu-prefetch",
                  cl::init(KitsuneOptions::defaultGPUPrefetch),
                  cl::desc("Enable generation of calls to prefetch managed "
                           "memory between host and device"),
                  cl::cat(cl::catKitClOpts));

// ------------------------- cuda tapir target options -------------------------

static const std::string clCudaArchHelp =
    join_items(KitsuneOptions::defaultCudaArch,
               "NVIDIA GPU architecture (default = ", ")");
static cl::opt<std::string>
    clCudaArch("tapir-cuda-arch",
               cl::init(KitsuneOptions::defaultCudaArch.str()),
               cl::desc(clCudaArchHelp), cl::cat(cl::catKitClOpts));

static cl::opt<std::string>
    clCudaVirtArch("tapir-cuda-virt-arch", cl::init(""),
                   cl::desc("NVIDIA GPU virtual architecture"),
                   cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clCudaFeatures(
    "tapir-cuda-features", cl::init(""),
    cl::desc("The target features to use in the cuda tapir target"),
    cl::cat(cl::catKitClOpts));

static cl::opt<std::string>
    clCudaRuntimeBCFile("tapir-cuda-runtime-bc", cl::init(""),
                        cl::desc("Path to the cuda runtime bitcode file"),
                        cl::cat(cl::catKitClOpts));

// ------------------------- hip tapir target options -------------------------

static const std::string clHipArchHelp = join_items(
    KitsuneOptions::defaultHipArch, "AMD GPU architecture (default = ", ")");
static cl::opt<std::string>
    clHipArch("tapir-hip-arch", cl::init(KitsuneOptions::defaultHipArch.str()),
              cl::desc(clHipArchHelp), cl::cat(cl::catKitClOpts));

static const std::string clHipSRAMECCHelp = join_items(
    toString(KitsuneOptions::defaultHipSRAMECC),
    "Whether to enable the sramecc target feature (default = '", "')");
static cl::opt<MaybeBool> clHipSRAMECC(
    "tapir-hip-sramecc", cl::init(KitsuneOptions::defaultHipSRAMECC),
    cl::desc(clHipSRAMECCHelp),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the sramecc- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the sramecc+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the sramecc feature unset")),
    cl::cat(cl::catKitClOpts));

static const std::string clHipXnackHelp =
    join_items(toString(KitsuneOptions::defaultHipXnack),
               "Whether to enable the xnack target feature (default = '", "')");
static cl::opt<MaybeBool> clHipXnack(
    "tapir-hip-xnack", cl::init(KitsuneOptions::defaultHipXnack),
    cl::desc(clHipXnackHelp),
    cl::values(
        clEnumValN(MaybeBool::Off, "off", "Set the xnack- target feature"),
        clEnumValN(MaybeBool::On, "on", "Set the xnack+ target feature"),
        clEnumValN(MaybeBool::Any, "any", "Leave the xnack feature unset")),
    cl::cat(cl::catKitClOpts));

static cl::opt<std::string> clHipFeatures(
    "tapir-hip-features", cl::init(""),
    cl::desc("The target features to use in the hip tapir target"),
    cl::cat(cl::catKitClOpts));

static cl::list<std::string> clHipRuntimeBCFiles(
    "tapir-hip-runtime-bcs",
    cl::desc("The bitcode files to use in the hip tapir target"),
    cl::cat(cl::catKitClOpts), cl::CommaSeparated);

// ----------------------- opencilk tapir target options -----------------------

static cl::opt<std::string> clOpenCilkRuntimeBCFile(
    "tapir-opencilk-runtime-bc", cl::init(""),
    cl::desc("Path to the bitcode file for the OpenCilk runtime bitcode file"));

static cl::alias
    clOpenCilkRuntimeBCPath("opencilk-runtime-bc-path", cl::NotHidden,
                            cl::aliasopt(clOpenCilkRuntimeBCFile),
                            cl::desc("Alias for --tapir-opencilk-runtime-bc"));

// -----------------------------------------------------------------------------

TapirTargetOptions::TapirTargetOptions(TTID tt) : tt(tt) {}

void TapirTargetOptions::setOptznLevelFrom(OptimizationLevel optLevel) {
  unsigned speedupLevel = optLevel.getSpeedupLevel();
  unsigned sizeLevel = optLevel.getSizeLevel();
  OptznLevel optznLevel = createOptznLevelFrom(speedupLevel, sizeLevel);

  setOptznLevel(optznLevel);
}

std::optional<TapirTargetOptions>
TapirTargetOptions::createFromCommandLine(OptznLevel optLevel) {
  if (clTapir.getNumOccurrences()) {
    TapirTargetOptions tto(clTapir);

    // No validation of inputs is done here. This is intentional since these
    // command line options are primarily for internal use. Obviously, tools
    // such as opt use these too, but it is probably safe to assume that anyone
    // using opt directly is sufficiently expert. If they are not, well ...

    // Set common tapir target options
    tto.tapirVerbose = clTapirVerbose;
    tto.kitrtVerbose = clTapirVerbose || clKitrtVerbose;
    tto.optLevel = optLevel;
    tto.fpOpFusionMode = codegen::getFuseFPOps();
    tto.lld = clLLD;
    if (clFixedThreadsPerBlock)
      tto.fixedThreadsPerBlock = clFixedThreadsPerBlock;
    if (clMaxThreadsPerBlock)
      tto.maxThreadsPerBlock = clMaxThreadsPerBlock;
    tto.gpuPrefetch = clGPUPrefetch;

    // Set cuda tapir target options
    tto.cudaArch = clCudaArch;
    tto.cudaVirtArch = clCudaVirtArch;
    tto.cudaTargetFeatures = clCudaFeatures;
    tto.cudaRuntimeBCFile = clCudaRuntimeBCFile;

    // Set hip tapir target options
    tto.hipArch = clHipArch;
    tto.hipSRAMECC = clHipSRAMECC;
    tto.hipXnack = clHipXnack;
    tto.hipTargetFeatures = clHipFeatures;
    tto.hipRuntimeBCFiles = clHipRuntimeBCFiles;

    // Set opencilk tapir target options
    tto.openCilkRuntimeBCFile = clOpenCilkRuntimeBCFile;

    // FIXME: This is here purely for debugging because it was in HipABI.cpp
    // originally. It really should go away.
    if (std::optional<std::string> tpb =
            sys::Process::GetEnv("KITHIP_THREADS_PER_BLOCK")) {
      if (clFixedThreadsPerBlock)
        errs() << "kitsune[hipabi]: Note that KITHIP_THREADS_PER_BLOCK is "
               << "overriding command line args.\n";
      tto.fixedThreadsPerBlock = std::stoi(tpb.value());
    }
    return tto;
  }
  return std::nullopt;
}

std::optional<TapirTargetOptions>
TapirTargetOptions::createFromCommandLine(unsigned speedupLevel) {
  return createFromCommandLine(createOptznLevelFrom(speedupLevel));
}

std::optional<TapirTargetOptions>
TapirTargetOptions::createFromCommandLine(char optLevel) {
  return createFromCommandLine(createOptznLevelFrom(optLevel));
}

std::optional<TapirTargetOptions>
TapirTargetOptions::create(const KitsuneOptions &opts, OptznLevel optLevel,
                           FPOpFusionMode fpOpFusionMode) {
  if (std::optional<TTID> tt = opts.getTTID()) {
    TapirTargetOptions tto(*tt);

    // Set common tapir target options.
    tto.tapirVerbose = opts.getTapirVerbose();
    tto.kitrtVerbose = opts.getTapirVerbose() or opts.getKitrtVerbose();
    tto.fpOpFusionMode = fpOpFusionMode;
    tto.optLevel = optLevel;
    tto.lld = opts.getLLD();

    // Set tapir target options shared by GPU-centric tapir targets.
    tto.fixedThreadsPerBlock = opts.getFixedThreadsPerBlock();
    tto.maxThreadsPerBlock = opts.getMaxThreadsPerBlock();
    tto.gpuPrefetch = opts.getGPUPrefetch();

    // Set cuda tapir target options.
    tto.cudaArch = opts.getCudaArch();
    tto.cudaVirtArch = opts.getCudaVirtArch();
    tto.cudaTargetFeatures = opts.getCudaFeatures();
    tto.cudaRuntimeBCFile = opts.getCudaRuntimeBCFile();

    // Set hip tapir target options.
    tto.hipArch = opts.getHipArch();
    tto.hipSRAMECC = opts.getHipSRAMECC();
    tto.hipXnack = opts.getHipXnack();
    tto.hipTargetFeatures = opts.getHipFeatures();
    tto.hipRuntimeBCFiles = opts.getHipRuntimeBCFiles();

    // Set opencilk tapir target options.
    tto.openCilkRuntimeBCFile = opts.getOpenCilkRuntimeBCFile();

    return tto;
  }
  return std::nullopt;
}

std::unique_ptr<TapirTargetOptions> TapirTargetOptions::clone() const {
  TapirTargetOptions *clone = new TapirTargetOptions;
  *clone = *this;

  return std::unique_ptr<TapirTargetOptions>(clone);
}

void TapirTargetOptions::print(raw_ostream &os, bool all) const {
  os << "Tapir target options:\n";
  os << "  Primary:                 " << tt << "\n";
  os << "  Compiler verbose:        " << getTapirVerbose() << "\n";
  os << "  Runtime verbose:         " << getKitrtVerbose() << "\n";
  os << "  Optimization level:      " << getOptznLevel() << "\n";
  os << "  FP fusion:               " << getFPOpFusionMode() << "\n";
  if (all || tt == TTID::Cuda || tt == TTID::Hip) {
    os << "  GPU fixed threads/block: " << getFixedThreadsPerBlock() << "\n";
    os << "  GPU max threads/block:   " << getMaxThreadsPerBlock() << "\n";
    os << "  GPU prefetch:            " << getGPUPrefetch() << "\n";
  }
  if (all || tt == TTID::Cuda) {
    os << "  Cuda arch:               " << getCudaArch() << "\n";
    os << "  Cuda virtual arch:       " << getCudaVirtArch() << "\n";
    os << "  Cuda target features:    " << getCudaTargetFeatures() << "\n";
    os << "  Cuda bitcode file:       " << getCudaRuntimeBCFile() << "\n";
  }
  if (all || tt == TTID::Hip) {
    os << "  Hip arch:                " << getHipArch() << "\n";
    os << "  Hip sramecc:             " << getHipSRAMECC() << "\n";
    os << "  Hip xnack:               " << getHipXnack() << "\n";
    os << "  Hip target features:     " << getHipTargetFeatures() << "\n";
    os << "  Hip bitcode files: [\n";
    for (StringRef file : getHipRuntimeBCFiles())
      os << "    " << file << "\n";
    os << "  ]";
    os << "  LLD:                     " << getLLD() << "\n";
  }
  if (all || tt == TTID::OpenCilk) {
    os << "  Opencilk bitcode file:   " << getOpenCilkRuntimeBCFile() << "\n";
  }
}
