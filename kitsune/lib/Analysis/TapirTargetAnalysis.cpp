//===- ModuleSummaryAnalysis.cpp - Module summary index builder -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An interface for information about the tapir targets needed by a module.
//
//===----------------------------------------------------------------------===//

#include "kitsune/Analysis/TapirTargetAnalysis.h"
#include "kitsune/Config/config.h"
#include "kitsune/Core/CommandLineOptions.h"
#include "kitsune/Core/TapirTargets.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TapirLoopHints.h"
#include "llvm/Analysis/TapirTaskInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/TapirUtils.h"

#include <set>

#define DEBUG_TYPE "tapir-target-analysis"

using namespace llvm;

static cl::opt<bool>
    clDumpTTO("dump-tapir-target-options", cl::init(false),
              cl::desc("Dump the tapir target options if they have been set"),
              cl::Hidden, cl::cat(cl::catKitClOpts));

/// Empty vector of tapir targets to be used when
/// @ref TapirTargetInfo::getRequiredTTs is called with a function that does not
/// contain any tapir loops.
static const std::vector<TTID> noTTs;

static std::unique_ptr<TapirTarget>
createTT(TTID id, Module &m, const TapirTargetOptions &tto) {
  // Yes, this is absolutely hideous. We should try to find a nicer way than
  // this horrendous conditionally compiled mess!
  switch (id) {
  case TTID::Nolo:
    return nullptr;

  case TTID::Serial:
    return std::make_unique<SerialABI>(m, tto);

#if KITSUNE_CUDA_ENABLED
  case TTID::Cuda:
    return std::make_unique<CudaABI>(m, tto);
#endif // KITSUNE_CUDA_ENABLED

#if KITSUNE_HIP_ENABLED
  case TTID::Hip:
    return std::make_unique<HipABI>(m, tto);
#endif // KITSUNE_HIP_ENABLED

#if KITSUNE_LAMBDA_ENABLED
  case TTID::Lambda:
    return std::make_unique<LambdaABI>(m, tto);
#endif // KITSUNE_LAMBDA_ENABLED

#if KITSUNE_OMPTASK_ENABLED
  case TTID::OMPTask:
    return std::make_unique<OMPTask>(m, tto);
#endif // KITSUNE_OMPTASK_ENABLED

#if KITSUNE_OPENCILK_ENABLED
  case TTID::OpenCilk:
    return std::make_unique<OpenCilkABI>(m, tto);
#endif // KITSUNE_OPENCILK_ENABLED

#if KITSUNE_OPENMP_ENABLED
  case TTID::OpenMP:
    llvm_unreachable("OpenMP ABI is out of date");
#endif // KITSUNE_OPENMP_ENABLED

#if KITSUNE_QTHREADS_ENABLED
  case TTID::Qthreads:
    return std::make_unique<QthreadsABI>(m);
#endif // KITSUNE_QTHREADS_ENABLED

#if KITSUNE_REALM_ENABLED
  case TTID::Realm:
    return std::make_unique<RealmABI>(m);
#endif // KITSUNE_REALM_ENABLED
       
#if KITSUNE_GPU_ENABLED
  case TTID::GPU:
    return std::make_unique<GPUABI>(m, tto);
#endif // KITSUNE_REALM_ENABLED

  default:
    llvm_unreachable("createTT: TTID not handled");
  }
}

TapirTargetInfo::TapirTargetInfo(std::optional<TapirTargetOptions> ttOpts)
    : ttOpts(ttOpts) {}

void TapirTargetInfo::computeRequiredTTs(Module &m, GetLoopInfo getLoopInfo,
                                         GetTaskInfo getTaskInfo) {
  ttsInFunc.clear();
  ttsInModule.clear();

  std::set<TTID> ttsForModule;
  for (Function &f : m.functions()) {
    if (not f.size())
      continue;

    LoopInfo &li = getLoopInfo(f);
    TaskInfo &ti = getTaskInfo(f);
    std::set<TTID> ttsForFunc;
    for (const Loop *tl : li) {
      for (const Loop *loop : post_order(tl)) {
        if (getTaskIfTapirLoopStructure(loop, &ti)) {
          TTID tt = ttOpts->getTTID();
          if (std::optional<TTID> hintTT = TapirLoopHints(loop).getLoopTarget())
            tt = *hintTT;
          ttsForFunc.insert(tt);
          ttsForModule.insert(tt);
        }
      }
    }

    // TODO: If there are multiple tapir targets required by a function, they
    // should be sorted in an "ideal" order for processing. This is because
    // some targets may change the function in ways that make it more difficult
    // to process another.
    //
    // However, this whole multi-target code needs to be reconsidered. It is not
    // at all clear that there exists an ordering that will work for any given
    // pair of tapir targets. We may need to restrict the tapir targets that
    // loops may be attributed with or the order in which the attributes can
    // appear.
    this->ttsInFunc[&f].assign(ttsForFunc.begin(), ttsForFunc.end());
  }
  this->ttsInModule.assign(ttsForModule.begin(), ttsForModule.end());
}

void TapirTargetInfo::addTT(TTID id, TapirTarget *tt) { tts[id] = tt; }

bool TapirTargetInfo::hasTT(TTID id) const { return tts.find(id) != tts.end(); }

TapirTarget *TapirTargetInfo::getTT(TTID id) const {
  assert(hasTT(id) && "TapirTarget has been created for ID");
  return tts.at(id);
}

TTID TapirTargetInfo::getTTID() const {
  assert(ttOpts && "Tapir target options have not been set");
  return ttOpts->getTTID();
}

std::optional<TTID> TapirTargetInfo::getTTIDOrNull() const {
  if (ttOpts)
    return ttOpts->getTTID();
  return std::nullopt;
}

const TapirTargetOptions &TapirTargetInfo::getOptions() const {
  assert(ttOpts && "Tapir target options have not been set");
  return *ttOpts;
}

ArrayRef<TTID> TapirTargetInfo::getRequiredTTs(Function &f) const {
  if (ttsInFunc.find(&f) == ttsInFunc.end())
    return noTTs;
  return ttsInFunc.at(&f);
}

ArrayRef<TTID> TapirTargetInfo::getRequiredTTs(Module &) const {
  return ttsInModule;
}

bool TapirTargetInfo::invalidate(Module &, const PreservedAnalyses &pa,
                                 ModuleAnalysisManager::Invalidator &) {
  // Just checking if the CFG is preserved should work.
  // If loop analyses are not preserved, then this analysis is invalid.
  auto lac = pa.getChecker<LoopAnalysis>();
  return not(lac.preserved() or lac.preservedSet<AllAnalysesOn<Function>>());
}

AnalysisKey TapirTargetAnalysis::Key;

TapirTargetAnalysis::TapirTargetAnalysis(std::optional<TapirTargetOptions> tto)
    : ttInfo(tto) {
  if (clDumpTTO and ttInfo.hasTTID())
    ttInfo.getOptions().print(outs(), /*all=*/true);
}

TapirTargetAnalysis::Result
TapirTargetAnalysis::run(Module &m, ModuleAnalysisManager &mam) {
  // If a primary tapir target has not been set, don't do anything more since
  // the lack of a primary tapir target implies that tapir lowering has not been
  // enabled.
  if (not ttInfo.hasTTID())
    return ttInfo;

  auto &fam = mam.getResult<FunctionAnalysisManagerModuleProxy>(m).getManager();
  auto getLoopInfo = [&](Function &f) -> LoopInfo & {
    return fam.getResult<LoopAnalysis>(f);
  };
  auto getTaskInfo = [&](Function &f) -> TaskInfo & {
    return fam.getResult<TaskAnalysis>(f);
  };

  ttInfo.computeRequiredTTs(m, getLoopInfo, getTaskInfo);

  const TapirTargetOptions &tto = ttInfo.getOptions();
  std::vector<TTID> ids = ttInfo.getRequiredTTs(m);
  ids.push_back(ttInfo.getTTID());
  for (TTID id : ids) {
    if (not ttInfo.hasTT(id)) {
      tts[id] = createTT(id, m, tto);
      ttInfo.addTT(id, tts.at(id).get());
    }
  }

  return ttInfo;
}

char TapirTargetAnalysisWrapperPass::ID = 0;
INITIALIZE_PASS(TapirTargetAnalysisWrapperPass, DEBUG_TYPE,
                "Tapir Target Analysis", false, true)

TapirTargetAnalysisWrapperPass::TapirTargetAnalysisWrapperPass()
    : ImmutablePass(ID), ttInfo(std::nullopt) {
  initializeTapirTargetAnalysisWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

TapirTargetAnalysisWrapperPass::TapirTargetAnalysisWrapperPass(
    std::optional<TapirTargetOptions> ttOpts)
    : ImmutablePass(ID), ttInfo(ttOpts) {
  initializeTapirTargetAnalysisWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

void TapirTargetAnalysisWrapperPass::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesAll();
}

TapirTargetAnalysisWrapperPass::Result
TapirTargetAnalysisWrapperPass::getResult() const {
  return ttInfo;
}

ModulePass *llvm::createTapirTargetAnalysisWrapperPass(
    std::optional<TapirTargetOptions> ttOpts) {
  return new TapirTargetAnalysisWrapperPass(ttOpts);
}
