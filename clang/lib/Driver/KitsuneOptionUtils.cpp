//===- KitsuneOptionUtils.cpp - Utilities for Kitsune-specific options ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities to parse Kitsune-specific command line options
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/KitsuneOptionUtils.h"
#include "kitsune/Frontend/KitsuneOptions.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/OptionUtils.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"

#include <sstream>

using namespace clang;
using namespace clang::driver;
using namespace clang::driver::options;
using namespace llvm;
using namespace llvm::opt;

using llvm::driver::KitsuneOptions;

static unsigned reportInvalidOptimizationLevel(const opt::Arg &a,
                                               DiagnosticsEngine &diags) {
  diags.Report(diag::err_drv_kitsune_bad_opt_level) << a.getSpelling();

  // The maximum supported optimization level is 3. Return an unsupported
  // level because we shouldn't return the default here. The caller should
  // be prepared to deal with such an occurrence.
  return 4;
}

unsigned clang::getSpeedupLevel(const opt::ArgList &args,
                                DiagnosticsEngine &diags) {
  unsigned defaultSpeedup = KitsuneOptions::defaultSpeedupLevel;
  if (const opt::Arg *a = args.getLastArg(OPT_O_Group)) {
    const opt::Option &option = a->getOption();
    if (option.matches(OPT_O0)) {
      return 0;
    } else if (a->getNumValues()) {
      StringRef v = a->getValue();
      if (v == "1" || v == "g")
        return 1;
      else if (v == "2" || v == "s" || v == "z")
        return 2;
      else if (v == "3")
        return 3;
    }
    return reportInvalidOptimizationLevel(*a, diags);
  }
  return defaultSpeedup;
}

unsigned clang::getSizeLevel(const opt::ArgList &args,
                             DiagnosticsEngine &diags) {
  const opt::Arg *a = args.getLastArg(OPT_O_Group);
  if (a && a->getNumValues()) {
    StringRef v = a->getValue();
    if (v == "s")
      return 1;
    else if (v == "z")
      return 2;
  }
  return 0;
}

std::optional<TTID> clang::parseTapirTargetIfValid(const opt::ArgList &args) {
  if (!args.hasArg(OPT_tapir_EQ))
    return std::nullopt;
  return createTTIDFrom(args.getLastArgValue(OPT_tapir_EQ));
}

std::optional<StringRef>
clang::getTapirTargetConfigFileName(const opt::ArgList &args) {
  if (!args.hasArg(OPT_tapir_EQ))
    return std::nullopt;

  // Even if the value of the --tapir option is invalid, this will get called.
  std::optional<TTID> tt = createTTIDFrom(args.getLastArgValue(OPT_tapir_EQ));
  if (!tt)
    return std::nullopt;

  switch (*tt) {
  case TTID::Nolo:
    return "nolo.cfg";
  case TTID::Serial:
    return "serial.cfg";
  case TTID::Cuda:
    return "cuda.cfg";
  case TTID::Hip:
    return "hip.cfg";
  case TTID::Lambda:
    return "lambda.cfg";
  case TTID::OMPTask:
    return "omptask.cfg";
  case TTID::OpenCilk:
    return "opencilk.cfg";
  case TTID::OpenMP:
    return "openmp.cfg";
  case TTID::Qthreads:
    return "qthreads.cfg";
  case TTID::Realm:
    return "realm.cfg";
  case TTID::GPU:
    return "gpu.cfg";
  }
  llvm_unreachable("getTapirTargetConfigFile: TTID not handled");
}

static std::vector<std::string>
parseCommaSeparatedList(StringRef s, DiagnosticsEngine &diags) {
  std::vector<std::string> list;
  std::string tok;
  std::istringstream iss(s.str());
  while (std::getline(iss, tok, ','))
    list.push_back(tok);
  return list;
}

static void parseKitsuneCommonGPUArgs(KitsuneOptions &opts, const ArgList &args,
                                      const OptTable &optTable,
                                      DiagnosticsEngine &diags) {
  if (const Arg *a = args.getLastArg(OPT_tapir_gpu_tpb_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setFixedThreadsPerBlock(n);
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_gpu_max_tpb_EQ)) {
    unsigned n;
    StringRef val = a->getValue();
    val.getAsInteger(10, n);
    opts.setMaxThreadsPerBlock(n);
  }

  opts.setGPUPrefetch(args.hasFlag(OPT_tapir_gpu_prefetch,
                                   OPT_tapir_gpu_no_prefetch,
                                   KitsuneOptions::defaultGPUPrefetch));
}

static bool parseKitsuneCudaArgs(KitsuneOptions &opts, const ArgList &args,
                                 const OptTable &optTable,
                                 DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_cuda_arch_EQ, OPT_tapir_cuda_virt_arch_EQ,
      OPT_tapir_cuda_features_EQ, OPT_tapir_cuda_runtime_bc_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setCudaArch(args.getLastArgValue(OPT_tapir_cuda_arch_EQ));
  opts.setCudaVirtArch(args.getLastArgValue(OPT_tapir_cuda_virt_arch_EQ));
  opts.setCudaFeatures(args.getLastArgValue(OPT_tapir_cuda_features_EQ));
  opts.setCudaRuntimeBCFile(args.getLastArgValue(OPT_tapir_cuda_runtime_bc_EQ));

  parseKitsuneCommonGPUArgs(opts, args, optTable, diags);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneHipArgs(KitsuneOptions &opts, const ArgList &args,
                                const OptTable &optTable,
                                DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  const OptSpecifier requiredOpts[] = {
      OPT_tapir_hip_arch_EQ,        OPT_tapir_hip_features_EQ,
      OPT_tapir_hip_runtime_bcs_EQ, OPT_tapir_lld_EQ,
      OPT_tapir_hip_sramecc_EQ,     OPT_tapir_hip_xnack_EQ};
  for (OptSpecifier opt : requiredOpts)
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setHipArch(args.getLastArgValue(OPT_tapir_hip_arch_EQ));
  opts.setHipFeatures(args.getLastArgValue(OPT_tapir_hip_features_EQ));
  for (StringRef file : parseCommaSeparatedList(
           args.getLastArgValue(OPT_tapir_hip_runtime_bcs_EQ), diags))
    opts.addHipRuntimeBCFile(file);
  opts.setLLD(args.getLastArgValue(OPT_tapir_lld_EQ));

  if (const Arg *a = args.getLastArg(OPT_tapir_hip_sramecc_EQ)) {
    StringRef val = a->getValue();
    if (std::optional<llvm::MaybeBool> mb = llvm::createMaybeBoolFrom(val))
      opts.setHipSramECC(*mb);
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  if (const Arg *a = args.getLastArg(OPT_tapir_hip_xnack_EQ)) {
    StringRef val = a->getValue();
    if (std::optional<llvm::MaybeBool> mb = llvm::createMaybeBoolFrom(val))
      opts.setHipXnack(*mb);
    else
      diags.Report(diag::err_drv_invalid_argument_to_option)
          << val << a->getOption().getName();
  }

  parseKitsuneCommonGPUArgs(opts, args, optTable, diags);

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneLambdaArgs(KitsuneOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneLambdaargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOMPTaskArgs(KitsuneOptions &opts, const ArgList &args,
                                    const OptTable &optTable,
                                    DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneOMPTaskargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOpenCilkArgs(KitsuneOptions &opts, const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  for (OptSpecifier opt : {OPT_tapir_opencilk_runtime_bc_EQ})
    if (!args.hasArg(opt))
      diags.Report(diag::err_drv_kitsune_missing_required)
          << optTable.getOptionName(opt);

  if (diags.getNumErrors() > numErrorsBefore)
    return false;

  opts.setOpenCilkRuntimeBCFile(
      args.getLastArgValue(OPT_tapir_opencilk_runtime_bc_EQ));

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneOpenMPArgs(KitsuneOptions &opts, const ArgList &args,
                                   const OptTable &optTable,
                                   DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneOpenMPargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneQthreadsArgs(KitsuneOptions &opts, const ArgList &args,
                                     const OptTable &optTable,
                                     DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneQthreadsargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneRealmArgs(KitsuneOptions &opts, const ArgList &args,
                                  const OptTable &optTable,
                                  DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  // Don't hit unreachable if an error has already occurred
  if (!numErrorsBefore)
    llvm_unreachable("NOT IMPLEMENTED: ParseKitsuneRealmargs");

  return diags.getNumErrors() == numErrorsBefore;
}

static bool parseKitsuneTTArgs(KitsuneOptions &kitOpts, TTID tt,
                               const ArgList &args, const OptTable &optTable,
                               DiagnosticsEngine &diags) {
  switch (tt) {
  case llvm::TTID::Nolo:
    return true;
  case llvm::TTID::Cuda:
    return parseKitsuneCudaArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Hip:
    return parseKitsuneHipArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Lambda:
    return parseKitsuneLambdaArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OMPTask:
    return parseKitsuneOMPTaskArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OpenCilk:
    return parseKitsuneOpenCilkArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::OpenMP:
    return parseKitsuneOpenMPArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Qthreads:
    return parseKitsuneQthreadsArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Realm:
    return parseKitsuneRealmArgs(kitOpts, args, optTable, diags);
  case llvm::TTID::Serial:
  case llvm::TTID::GPU: // TODO:?
    return true;
  }
  llvm_unreachable("ParseKitsuneTTArgs: TTID not handled");
}

bool clang::parseKitsuneArgs(KitsuneOptions &kitOpts, const char *argv0,
                             const ArgList &args, const OptTable &optTable,
                             DiagnosticsEngine &diags) {
  unsigned numErrorsBefore = diags.getNumErrors();

  kitOpts.setKitsuneFrontend(IsKitsuneFrontend(argv0));
  kitOpts.setStripmineLoops(args.hasArg(OPT_fstripmine));
  kitOpts.setPrintBeforeFirst(args.hasArg(OPT_print_before_first));

  kitOpts.setTapirVerbose(args.hasArg(OPT_tapir_verbose));
  kitOpts.setKitrtVerbose(args.hasArg(OPT_kitrt_verbose));

  if (args.hasArg(OPT_tapir_EQ)) {
    StringRef argVal = args.getLastArgValue(OPT_tapir_EQ);
    if (std::optional<llvm::TTID> tt = createTTIDFrom(argVal)) {
      parseKitsuneTTArgs(kitOpts, *tt, args, optTable, diags);
      kitOpts.setTTID(*tt);
    }
  }

  kitOpts.setKokkos(args.hasArg(OPT_kokkos));
  kitOpts.setKokkosNoInit(args.hasArg(OPT_kokkos_no_init));

  return diags.getNumErrors() == numErrorsBefore;
}
