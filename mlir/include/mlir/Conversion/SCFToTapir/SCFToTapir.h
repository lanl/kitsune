//===- ConvertSCFToTapir.h - Pass entrypoint -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SCFTOTAPIR_SCFTOTAPIR_H_
#define MLIR_CONVERSION_SCFTOTAPIR_SCFTOTAPIR_H_

#include <memory>
#include <vector>

namespace mlir {
struct LogicalResult;
class Pass;
class RewritePattern;

// Owning list of rewriting patterns.
class RewritePatternSet;

/// Collect a set of patterns to lower from scf.for, scf.if, and
/// loop.terminator to CFG operations within the Tapir dialect, in particular
/// convert structured control flow into CFG branch-based control flow.
void populateParallelToTapirConversionPatterns(RewritePatternSet &patterns);

/// Creates a pass to convert scf.for, scf.if and loop.terminator ops to CFG.
std::unique_ptr<Pass> createLowerToTapirPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SCFTOTAPIR_SCFTOTAPIR_H_
