//=- ToString.cpp - String and serialization functions ------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Implementation of additional string utilities and serialization functions.
///
//===----------------------------------------------------------------------===//

#include "kitsune/Support/ToString.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

std::string llvm::toString(const TTID &tt) {
  switch (tt) {
  case TTID::Nolo:
    return "nolo";
  case TTID::Serial:
    return "serial";
  case TTID::Cuda:
    return "cuda";
  case TTID::Hip:
    return "hip";
  case TTID::Lambda:
    return "lambda";
  case TTID::OMPTask:
    return "omptask";
  case TTID::OpenCilk:
    return "opencilk";
  case TTID::OpenMP:
    return "openmp";
  case TTID::Qthreads:
    return "qthreads";
  case TTID::Realm:
    return "realm";
  case TTID::GPU:
    return "gpu";
  }
  llvm_unreachable("toString: TTID not handled");
}

std::string llvm::toString(const MaybeBool &v) {
  switch (v) {
  case MaybeBool::Off:
    return "off";
  case MaybeBool::On:
    return "on";
  case MaybeBool::Any:
    return "any";
  }
  llvm_unreachable("toString: MaybeBool value not handled");
}

std::string llvm::toString(const OptznLevel &optLevel) {
  switch (optLevel) {
  case OptznLevel::O0:
    return "O0";
  case OptznLevel::O1:
    return "O1";
  case OptznLevel::O2:
    return "O2";
  case OptznLevel::O3:
    return "O3";
  case OptznLevel::Os:
    return "Os";
  case OptznLevel::Oz:
    return "Os";
  }
  llvm_unreachable("operator<<: OptznLevel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const TTID &tt) {
  os << toString(tt);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const std::optional<TTID> &tt) {
  if (tt.has_value())
    return os << *tt;
  else
    return os << "<<<std::nullopt>>>";
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const TapirSpawnStrategy &strategy) {
  switch (strategy) {
  case TapirSpawnStrategy::Sequential:
    return os << "Sequential";
  case TapirSpawnStrategy::DivideAndConquer:
    return os << "Divide and conquer";
  case TapirSpawnStrategy::GPU:
    return os << "GPU";
  }
  llvm_unreachable("operator<<: TapirSpawnStrategy not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const MaybeBool &v) {
  os << toString(v);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const OptznLevel &optLevel) {
  os << toString(optLevel);
  return os;
}

raw_ostream &llvm::operator<<(raw_ostream &os, const CodeGenOptLevel &cgOpt) {
  switch (cgOpt) {
  case CodeGenOptLevel::None:
    return os << "none (O0)";
  case CodeGenOptLevel::Less:
    return os << "less (O1)";
  case CodeGenOptLevel::Default:
    return os << "default (O2)";
  case CodeGenOptLevel::Aggressive:
    return os << "aggressive (O3)";
  }
  llvm_unreachable("operator<<: CodeGenOptLevel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const CodeModel::Model &cm) {
  switch (cm) {
  case CodeModel::Tiny:
    return os << "tiny";
  case CodeModel::Small:
    return os << "small";
  case CodeModel::Kernel:
    return os << "kernel";
  case CodeModel::Medium:
    return os << "medium";
  case CodeModel::Large:
    return os << "large";
  }
  llvm_unreachable("operator<<: CodeModel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const Reloc::Model &rm) {
  switch (rm) {
  case Reloc::Static:
    return os << "static";
  case Reloc::PIC_:
    return os << "pic";
  case Reloc::DynamicNoPIC:
    return os << "dynamic-no-pic";
  case Reloc::ROPI:
    return os << "ropi";
  case Reloc::RWPI:
    return os << "rwpi";
  case Reloc::ROPI_RWPI:
    return os << "ropi-rwpi";
  }
  llvm_unreachable("operator<<: RelocModel not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const BasicBlockSection &sect) {
  switch (sect) {
  case BasicBlockSection::All:
    return os << "all";
  case BasicBlockSection::List:
    return os << "list";
  case BasicBlockSection::Preset:
    return os << "preset";
  case BasicBlockSection::None:
    return os << "none";
  }
  llvm_unreachable("operator<<: BasicBlockSection not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const DebuggerKind &dbgKind) {
  switch (dbgKind) {
  case DebuggerKind::Default:
    return os << "default";
  case DebuggerKind::GDB:
    return os << "gdb";
  case DebuggerKind::LLDB:
    return os << "lldb";
  case DebuggerKind::SCE:
    return os << "sce";
  case DebuggerKind::DBX:
    return os << "dbx";
  }
  llvm_unreachable("operator<<: DebuggerKind not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const EABI &eabi) {
  switch (eabi) {
  case EABI::Unknown:
    return os << "unknown";
  case EABI::Default:
    return os << "default";
  case EABI::EABI4:
    return os << "eabi4";
  case EABI::EABI5:
    return os << "eabi5";
  case EABI::GNU:
    return os << "gnu";
  }
  llvm_unreachable("operator<<: EABI not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const FloatABI::ABIType &abi) {
  switch (abi) {
  case FloatABI::Default:
    return os << "default";
  case FloatABI::Soft:
    return os << "soft";
  case FloatABI::Hard:
    return os << "hard";
  }
  llvm_unreachable("operator<<: FloatABI::ABIType not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const FPOpFusion::FPOpFusionMode &mode) {
  switch (mode) {
  case FPOpFusion::Fast:
    return os << "fast";
  case FPOpFusion::Standard:
    return os << "standard";
  case FPOpFusion::Strict:
    return os << "strict";
  }
  llvm_unreachable("operator<<: FPOpFusionMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const GlobalISelAbortMode &m) {
  switch (m) {
  case GlobalISelAbortMode::Disable:
    return os << "disable";
  case GlobalISelAbortMode::Enable:
    return os << "enable";
  case GlobalISelAbortMode::DisableWithDiag:
    return os << "disable-with-diag";
  }
  llvm_unreachable("operator<<: GlobalISelAbortMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const SwiftAsyncFramePointerMode &mode) {
  switch (mode) {
  case SwiftAsyncFramePointerMode::DeploymentBased:
    return os << "deployment-based";
  case SwiftAsyncFramePointerMode::Always:
    return os << "always";
  case SwiftAsyncFramePointerMode::Never:
    return os << "never";
  }
  llvm_unreachable("operator<<: SwiftAsyncFramePointerMode not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const ThreadModel::Model &model) {
  switch (model) {
  case ThreadModel::POSIX:
    return os << "posix";
  case ThreadModel::Single:
    return os << "single";
  }
  llvm_unreachable("operator<<: ThreadModel::Model not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os,
                              const MCTargetOptions::DwarfDirectory &dd) {
  switch (dd) {
  case MCTargetOptions::DisableDwarfDirectory:
    return os << "disable";
  case MCTargetOptions::EnableDwarfDirectory:
    return os << "enable";
  case MCTargetOptions::DefaultDwarfDirectory:
    return os << "default";
  }
  llvm_unreachable("operator<<: DwarfDirectory not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const DebugCompressionType &dc) {
  switch (dc) {
  case DebugCompressionType::None:
    return os << "none";
  case DebugCompressionType::Zlib:
    return os << "zlib";
  case DebugCompressionType::Zstd:
    return os << "zstd";
  }
  llvm_unreachable("operator<<: DebugCompressionType not handled");
}

raw_ostream &llvm::operator<<(raw_ostream &os, const EmitDwarfUnwindType &du) {
  switch (du) {
  case EmitDwarfUnwindType::Always:
    return os << "always";
  case EmitDwarfUnwindType::NoCompactUnwind:
    return os << "no-compact-unwind";
  case EmitDwarfUnwindType::Default:
    return os << "default";
  }
  llvm_unreachable("operator<<: EmitDwarfUnwindType not handled");
}
