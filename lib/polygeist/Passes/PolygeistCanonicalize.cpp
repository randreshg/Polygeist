//===- PolygeistCanonicalize.cpp - Cutom canonicalizer------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_POLYGEISTCANONICALIZE
#include "mlir/Pass/Pass.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Passes/Passes.h.inc"

#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Dialect.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"

using namespace mlir;
using namespace polygeist;

namespace {
struct PolygeistCanonicalizePass
    : public impl::PolygeistCanonicalizeBase<PolygeistCanonicalizePass> {
  PolygeistCanonicalizePass() = default;
  PolygeistCanonicalizePass(const GreedyRewriteConfig &config,
                            ArrayRef<std::string> disabledPatterns,
                            ArrayRef<std::string> enabledPatterns)
      : config(config) {
    this->topDownProcessingEnabled = config.getUseTopDownTraversal();
    this->enableRegionSimplification = config.getRegionSimplificationLevel() != GreedySimplifyRegionLevel::Disabled;
    this->maxIterations = config.getMaxIterations();
    this->maxNumRewrites = config.getMaxNumRewrites();
    this->disabledPatterns = disabledPatterns;
    this->enabledPatterns = enabledPatterns;
  }
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Set the config from possible pass options set in the meantime.
    config.setUseTopDownTraversal(topDownProcessingEnabled);
    config.setRegionSimplificationLevel(enableRegionSimplification
        ? GreedySimplifyRegionLevel::Aggressive
        : GreedySimplifyRegionLevel::Disabled);
    config.setMaxIterations(maxIterations);
    config.setMaxNumRewrites(maxNumRewrites);

    // The polygeist dialect is marked as a dependency to this pass and that
    // causes all of the custom canonicalizers (which are not neccessarily only
    // for polygeist ops) to get imported

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    patterns = std::make_shared<FrozenRewritePatternSet>(
        std::move(owningPatterns), disabledPatterns, enabledPatterns);
    return success();
  }
  void runOnOperation() override {
    LogicalResult converged =
        applyPatternsGreedily(getOperation(), *patterns, config);
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    if (testConvergence && failed(converged))
      signalPassFailure();
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createPolygeistCanonicalizePass() {
  return std::make_unique<PolygeistCanonicalizePass>();
}
/// Creates an instance of the Canonicalizer pass with the specified config.
std::unique_ptr<Pass> mlir::polygeist::createPolygeistCanonicalizePass(
    const GreedyRewriteConfig &config, ArrayRef<std::string> disabledPatterns,
    ArrayRef<std::string> enabledPatterns) {
  return std::make_unique<PolygeistCanonicalizePass>(config, disabledPatterns,
                                                     enabledPatterns);
}
