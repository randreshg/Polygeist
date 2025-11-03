//===- CGStmt.cc - Emit MLIR IRs by walking stmt-like AST nodes-*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IfScope.h"
#include "ValueCategory.h"
#include "clang-mlir.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Frontend/OpenMP/OMP.h.inc"
#include <csignal>

#define DEBUG_TYPE "CGStmt"

using namespace mlir;
using namespace mlir::arith;

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

/// Create or fetch an OpenMP reduction declaration for addition on the given
/// element type. The declaration is inserted at module scope if not present.
static mlir::omp::ReductionDeclareOp
getOrCreateAddReductionDecl(mlir::OpBuilder &builder, mlir::Operation *anchor,
                            mlir::Type elementTy) {
  auto *ctx = builder.getContext();
  // Build a stable symbol name based on the element type.
  std::string symName;
  if (auto it = dyn_cast<IntegerType>(elementTy)) {
    symName = ("add_i" + Twine(it.getWidth())).str();
  } else if (isa<IndexType>(elementTy)) {
    symName = "add_index";
  } else if (auto ft = dyn_cast<FloatType>(elementTy)) {
    unsigned bw = ft.getWidth();
    symName = ("add_f" + Twine(bw)).str();
  } else {
    // Fallback generic name; still unique per type via print.
    std::string tyStr;
    {
      llvm::raw_string_ostream os(tyStr);
      elementTy.print(os);
    }
    symName = ("add_" + tyStr);
  }

  // Find the containing module.
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  assert(module && "expected to be inside a ModuleOp");

  // Try to lookup an existing declaration.
  if (auto existing =
          module.lookupSymbol<mlir::omp::ReductionDeclareOp>(symName))
    return existing;

  // Create a new declaration at the start of the module body.
  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto decl = builder.create<mlir::omp::ReductionDeclareOp>(
      builder.getUnknownLoc(), builder.getStringAttr(symName),
      mlir::TypeAttr::get(elementTy));

  // Build initializer region: yield zero of element type.
  {
    Region &init = decl.getInitializerRegion();
    Block *initBlock = new Block();
    init.push_back(initBlock);
    initBlock->addArgument(elementTy, builder.getUnknownLoc());
    OpBuilder ib(ctx);
    ib.setInsertionPointToStart(initBlock);
    Value zero;
    if (auto it = dyn_cast<IntegerType>(elementTy)) {
      zero =
          ib.create<ConstantIntOp>(builder.getUnknownLoc(), 0, it.getWidth());
    } else if (auto ft = dyn_cast<FloatType>(elementTy)) {
      zero = ib.create<arith::ConstantOp>(builder.getUnknownLoc(), elementTy,
                                          builder.getFloatAttr(elementTy, 0.0));
    } else if (isa<IndexType>(elementTy)) {
      zero = ib.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 0);
    } else {
      // Default to integer 0 of i64 and rely on casts later if needed.
      zero = ib.create<ConstantIntOp>(builder.getUnknownLoc(), 0, 64);
    }
    ib.create<mlir::omp::YieldOp>(builder.getUnknownLoc(), zero);
  }

  // Build combiner region: yield lhs + rhs.
  {
    Region &comb = decl.getReductionRegion();
    Block *combBlock = new Block();
    comb.push_back(combBlock);
    combBlock->addArgument(elementTy, builder.getUnknownLoc());
    combBlock->addArgument(elementTy, builder.getUnknownLoc());
    OpBuilder cb(ctx);
    cb.setInsertionPointToStart(combBlock);
    Value lhs = combBlock->getArgument(0);
    Value rhs = combBlock->getArgument(1);
    Value sum;
    if (elementTy.isIntOrIndex())
      sum = cb.create<arith::AddIOp>(builder.getUnknownLoc(), lhs, rhs);
    else if (isa<FloatType>(elementTy))
      sum = cb.create<arith::AddFOp>(builder.getUnknownLoc(), lhs, rhs);
    else
      sum = lhs; // Unsupported: act as identity to keep verifier happy.
    cb.create<mlir::omp::YieldOp>(builder.getUnknownLoc(), sum);
  }

  // Omit atomic region for generality; it is optional.

  return decl;
}

bool MLIRScanner::getLowerBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto *init = fors->getInit();
  if (auto *declStmt = dyn_cast<DeclStmt>(init))
    if (declStmt->isSingleDecl()) {
      auto loc = getMLIRLocation(declStmt->getBeginLoc());
      auto *decl = declStmt->getSingleDecl();
      if (auto *varDecl = dyn_cast<VarDecl>(decl)) {
        if (varDecl->hasInit()) {
          mlir::Value val = VisitVarDecl(varDecl).getValue(loc, builder);
          descr.setName(varDecl);
          descr.setType(val.getType());
          LLVM_DEBUG(descr.getType().print(llvm::dbgs()));

          if (descr.getForwardMode())
            descr.setLowerBound(val);
          else {
            val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
            descr.setUpperBound(val);
          }
          return true;
        }
      }
    }

  // BinaryOperator 0x7ff7aa17e938 'int' '='
  // |-DeclRefExpr 0x7ff7aa17e8f8 'int' lvalue Var 0x7ff7aa17e758 'i' 'int'
  // -IntegerLiteral 0x7ff7aa17e918 'int' 0
  if (auto *binOp = dyn_cast<clang::BinaryOperator>(init))
    if (binOp->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign)
      if (auto *declRefStmt = dyn_cast<DeclRefExpr>(binOp->getLHS())) {
        auto loc = getMLIRLocation(binOp->getExprLoc());
        mlir::Value val = Visit(binOp->getRHS()).getValue(loc, builder);
        val = builder.create<IndexCastOp>(
            loc, mlir::IndexType::get(builder.getContext()), val);
        descr.setName(cast<VarDecl>(declRefStmt->getDecl()));
        descr.setType(getMLIRType(declRefStmt->getDecl()->getType()));
        if (descr.getForwardMode())
          descr.setLowerBound(val);
        else {
          val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
          descr.setUpperBound(val);
        }
        return true;
      }
  return false;
}

// Make sure that the induction variable initialized in
// the for is the same as the one used in the condition.
bool matchIndvar(const Expr *expr, VarDecl *indVar) {
  while (const auto *IC = dyn_cast<ImplicitCastExpr>(expr)) {
    expr = IC->getSubExpr();
  }
  if (const auto *declRef = dyn_cast<DeclRefExpr>(expr)) {
    const auto *declRefName = declRef->getDecl();
    if (declRefName == indVar)
      return true;
  }
  return false;
}

bool MLIRScanner::getUpperBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto *cond = fors->getCond();
  if (auto *binaryOp = dyn_cast<clang::BinaryOperator>(cond)) {
    auto *lhs = binaryOp->getLHS();
    auto loc = getMLIRLocation(binaryOp->getExprLoc());
    if (!matchIndvar(lhs, descr.getName()))
      return false;

    if (descr.getForwardMode()) {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LE)
        return false;

      auto *rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(loc, builder);
      val = builder.create<IndexCastOp>(
          loc, mlir::IndexType::get(val.getContext()), val);
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_LE)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setUpperBound(val);
      return true;
    } else {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GE)
        return false;

      auto *rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(loc, builder);
      val = builder.create<IndexCastOp>(
          loc, mlir::IndexType::get(val.getContext()), val);
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_GT)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setLowerBound(val);
      return true;
    }
  }
  return false;
}

bool MLIRScanner::getConstantStep(clang::ForStmt *fors,
                                  mlirclang::AffineLoopDescriptor &descr) {
  auto *inc = fors->getInc();
  if (auto *unaryOp = dyn_cast<clang::UnaryOperator>(inc))
    if (unaryOp->isPrefix() || unaryOp->isPostfix()) {
      bool forwardLoop =
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc ||
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc;
      descr.setStep(1);
      descr.setForwardMode(forwardLoop);
      return true;
    }
  return false;
}

bool MLIRScanner::isTrivialAffineLoop(clang::ForStmt *fors,
                                      mlirclang::AffineLoopDescriptor &descr) {
  if (!getConstantStep(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getConstantStep -> false\n");
    return false;
  }
  if (!getLowerBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getLowerBound -> false\n");
    return false;
  }
  if (!getUpperBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getUpperBound -> false\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "isTrivialAffineLoop -> true\n");
  return true;
}

void MLIRScanner::buildAffineLoopImpl(
    clang::ForStmt *fors, mlir::Location loc, mlir::Value lb, mlir::Value ub,
    const mlirclang::AffineLoopDescriptor &descr) {
  auto affineOp = builder.create<affine::AffineForOp>(
      loc, lb, builder.getSymbolIdentityMap(), ub,
      builder.getSymbolIdentityMap(), descr.getStep(),
      /*iterArgs=*/std::nullopt);

  auto &reg = affineOp.getRegion();

  auto val = (mlir::Value)affineOp.getInductionVar();

  reg.front().clear();

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToEnd(&reg.front());

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&er.getRegion().back());
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&er.getRegion().back());

  if (!descr.getForwardMode()) {
    val = builder.create<SubIOp>(loc, val, lb);
    val = builder.create<SubIOp>(
        loc, builder.create<SubIOp>(loc, ub, getConstantIndex(1)), val);
  }
  auto idx = builder.create<IndexCastOp>(loc, descr.getType(), val);
  assert(params.find(descr.getName()) != params.end());
  params[descr.getName()].store(loc, builder, idx);

  // TODO: set loop context.
  Visit(fors->getBody());

  builder.setInsertionPointToEnd(&reg.front());
  builder.create<affine::AffineYieldOp>(loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
}

void MLIRScanner::buildAffineLoop(
    clang::ForStmt *fors, mlir::Location loc,
    const mlirclang::AffineLoopDescriptor &descr) {
  mlir::Value lb = descr.getLowerBound();
  mlir::Value ub = descr.getUpperBound();
  buildAffineLoopImpl(fors, loc, lb, ub, descr);
}

ValueCategory MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getForLoc());

  mlirclang::AffineLoopDescriptor affineLoopDescr;
  if (Glob.scopLocList.isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, loc, affineLoopDescr);
  } else {

    if (auto *s = fors->getInit()) {
      Visit(s);
    }

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<ConstantIntOp>(loc, true, 1);

    LoopContext lctx{builder.create<mlir::memref::AllocaOp>(loc, type),
                     builder.create<mlir::memref::AllocaOp>(loc, type)};
    builder.create<mlir::memref::StoreOp>(loc, truev, lctx.noBreak);

    auto *toadd = builder.getInsertionBlock()->getParent();
    auto &condB = *(new Block());
    toadd->getBlocks().push_back(&condB);
    auto &bodyB = *(new Block());
    toadd->getBlocks().push_back(&bodyB);
    auto &exitB = *(new Block());
    toadd->getBlocks().push_back(&exitB);

    builder.create<mlir::cf::BranchOp>(loc, &condB);

    builder.setInsertionPointToStart(&condB);

    if (auto *s = fors->getCond()) {
      auto condRes = Visit(s);
      auto cond = condRes.getValue(loc, builder);
      if (auto mt = dyn_cast<mlir::MemRefType>(cond.getType())) {
        cond = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(mt.getElementType(),
                                       mt.getMemorySpaceAsInt()),
            cond);
      }
      if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
        auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
        cond = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cond.getType().cast<mlir::IntegerType>();
      if (ty.getWidth() != 1) {
        cond = builder.create<arith::CmpIOp>(
            loc, CmpIPredicate::ne, cond,
            builder.create<ConstantIntOp>(loc, 0, ty));
      }
      auto nb = builder.create<mlir::memref::LoadOp>(
          loc, lctx.noBreak, std::vector<mlir::Value>());
      cond = builder.create<AndIOp>(loc, cond, nb);
      builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
    } else {
      auto cond = builder.create<mlir::memref::LoadOp>(
          loc, lctx.noBreak, std::vector<mlir::Value>());
      builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::memref::StoreOp>(
        loc,
        builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                             std::vector<mlir::Value>()),
        lctx.keepRunning, std::vector<mlir::Value>());

    loops.push_back(lctx);
    Visit(fors->getBody());

    builder.create<mlir::memref::StoreOp>(
        loc,
        builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                             std::vector<mlir::Value>()),
        lctx.keepRunning, std::vector<mlir::Value>());
    if (auto *s = fors->getInc()) {
      IfScope scope(*this);
      Visit(s);
    }
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        !isTerminator(&builder.getInsertionBlock()->back())) {
      builder.create<mlir::cf::BranchOp>(loc, &condB);
    }

    builder.setInsertionPointToStart(&exitB);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXForRangeStmt(clang::CXXForRangeStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getForLoc());

  if (auto *s = fors->getInit()) {
    Visit(s);
  }
  Visit(fors->getRangeStmt());
  Visit(fors->getBeginStmt());
  Visit(fors->getEndStmt());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);

  LoopContext lctx{builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)};
  builder.create<mlir::memref::StoreOp>(loc, truev, lctx.noBreak);

  auto *toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::cf::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto *s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(loc, builder);
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
  } else {
    auto cond = builder.create<mlir::memref::LoadOp>(
        loc, lctx.noBreak, std::vector<mlir::Value>());
    builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                           std::vector<mlir::Value>()),
      lctx.keepRunning, std::vector<mlir::Value>());

  loops.push_back(lctx);
  Visit(fors->getLoopVarStmt());
  Visit(fors->getBody());

  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                           std::vector<mlir::Value>()),
      lctx.keepRunning, std::vector<mlir::Value>());
  if (auto *s = fors->getInc()) {
    IfScope scope(*this);
    Visit(s);
  }
  loops.pop_back();
  if (builder.getInsertionBlock()->empty() ||
      !isTerminator(&builder.getInsertionBlock()->back())) {
    builder.create<mlir::cf::BranchOp>(loc, &condB);
  }

  builder.setInsertionPointToStart(&exitB);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPSingleDirective(clang::OMPSingleDirective *par) {
  auto loc = getMLIRLocation(par->getBeginLoc());
  IfScope scope(*this);

  builder.create<omp::BarrierOp>(loc);
  auto affineOp = builder.create<omp::MasterOp>(loc);
  builder.create<omp::BarrierOp>(loc);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  affineOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  Visit(cast<CapturedStmt>(par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  builder.create<scf::YieldOp>(loc);
  allocationScope = oldScope;
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPTaskDirective(clang::OMPTaskDirective *task) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(task->getBeginLoc());

  // Map to store original variable mappings
  std::map<VarDecl *, ValueCategory> prevInduction;

  /// Handle the clauses in the task directive
  mlir::Value ifExprVal = nullptr;
  mlir::Value finalExprVal = nullptr;
  mlir::UnitAttr untiedAttr = nullptr;
  mlir::UnitAttr mergeableAttr = nullptr;
  mlir::Value priorityVal = nullptr;

  SmallVector<Value, 4> inReductionVars; // If needed
  ArrayAttr inReductionsAttr = nullptr;  // If needed

  SmallVector<Attribute, 4> dependKindAttrs;
  SmallVector<Value, 4> dependVars;

  // Handle other clauses like allocate_vars, allocators_vars if present
  SmallVector<Value, 4> allocateVars;
  SmallVector<Value, 4> allocatorsVars;

  // Iterate over clauses in the OMPTaskDirective
  for (auto *f : task->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_if: {
      // Extract if expression
      auto *ifClause = cast<OMPIfClause>(f);
      ifExprVal = Visit(ifClause->getCondition()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_final: {
      auto *finalClause = cast<OMPFinalClause>(f);
      finalExprVal = Visit(finalClause->getCondition()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_untied: {
      untiedAttr = mlir::UnitAttr::get(builder.getContext());
    } break;
    case llvm::omp::OMPC_mergeable: {
      mergeableAttr = mlir::UnitAttr::get(builder.getContext());
    } break;
    case llvm::omp::OMPC_priority: {
      auto *priorityClause = cast<OMPPriorityClause>(f);
      priorityVal = Visit(priorityClause->getPriority()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_depend: {
      auto *depClause = cast<OMPDependClause>(f);
      auto depKind = depClause->getDependencyKind();

      mlir::omp::ClauseTaskDepend pbKind;
      switch (depKind) {
      case OMPC_DEPEND_in:
        pbKind = mlir::omp::ClauseTaskDepend::taskdependin;
        break;
      case OMPC_DEPEND_out:
        pbKind = mlir::omp::ClauseTaskDepend::taskdependout;
        break;
      case OMPC_DEPEND_inout:
        pbKind = mlir::omp::ClauseTaskDepend::taskdependinout;
        break;
      default:
        llvm_unreachable("Unknown dependency kind");
      }

      auto kindAttr =
          mlir::omp::ClauseTaskDependAttr::get(builder.getContext(), pbKind);

      for (auto *depExpr : depClause->varlists()) {
        if (auto *arraySection =
                dyn_cast<clang::OMPArraySectionExpr>(depExpr)) {
          // Collect all dimensions from nested array sections
          SmallVector<clang::OMPArraySectionExpr *> sections;
          clang::Expr *currentExpr = depExpr;

          // Traverse nested array sections to collect all dimensions
          while (auto *ase =
                     dyn_cast<clang::OMPArraySectionExpr>(currentExpr)) {
            sections.push_back(ase);
            currentExpr = ase->getBase();
          }

          // Get the base array (innermost expression)
          auto baseVC = Visit(currentExpr);
          mlir::Value base = baseVC.getValue(loc, builder);
          auto memrefType = base.getType().cast<mlir::MemRefType>();
          unsigned rank = memrefType.getRank();

          // Handle pointer indirection for extra dimensions
          // If sections.size() > rank, we need to load through pointer levels
          unsigned indirectionLevels = 0;
          if (sections.size() > rank) {
            indirectionLevels = sections.size() - rank;

            // Process pointer indirections from outermost to innermost
            for (unsigned i = 0; i < indirectionLevels; i++) {
              auto *section = sections[sections.size() - 1 - i];

              // Get the offset for this level
              mlir::Value offset =
                  section->getLowerBound()
                      ? builder
                            .create<arith::IndexCastOp>(
                                loc, builder.getIndexType(),
                                Visit(section->getLowerBound())
                                    .getValue(loc, builder))
                            .getResult()
                      : builder.create<arith::ConstantIndexOp>(loc, 0)
                            .getResult();

              // Load the pointer at this offset
              base = builder.create<mlir::memref::LoadOp>(
                  loc, base, ArrayRef<mlir::Value>{offset});

              // Update memrefType and rank for the loaded value
              memrefType = base.getType().cast<mlir::MemRefType>();
              rank = memrefType.getRank();
            }
          }

          // Prepare subview parameters for remaining dimensions
          SmallVector<mlir::Value> offsets, sizes, strides;
          // Only process the dimensions that remain after indirection
          for (unsigned i = indirectionLevels; i < sections.size(); i++) {
            auto *section = sections[sections.size() - 1 - i];
            // Lower bound (default 0)
            mlir::Value lb =
                section->getLowerBound()
                    ? builder
                          .create<arith::IndexCastOp>(
                              loc, builder.getIndexType(),
                              Visit(section->getLowerBound())
                                  .getValue(loc, builder))
                          .getResult()
                    : builder.create<arith::ConstantIndexOp>(loc, 0)
                          .getResult();

            // Length
            mlir::Value len =
                builder
                    .create<arith::IndexCastOp>(
                        loc, builder.getIndexType(),
                        Visit(section->getLength()).getValue(loc, builder))
                    .getResult();

            // Stride (default 1)
            mlir::Value stride =
                section->getStride()
                    ? builder
                          .create<arith::IndexCastOp>(
                              loc, builder.getIndexType(),
                              Visit(section->getStride())
                                  .getValue(loc, builder))
                          .getResult()
                    : builder.create<arith::ConstantIndexOp>(loc, 1)
                          .getResult();

            offsets.push_back(lb);
            sizes.push_back(len);
            strides.push_back(stride);
          }

          // Create subview for the entire multi-dimensional section
          auto subview = builder.create<mlir::memref::SubViewOp>(
              loc, base, offsets, sizes, strides);

          dependVars.push_back(subview);
          dependKindAttrs.push_back(kindAttr);
        } else {
          // Handle regular variable dependency
          auto vc = Visit(depExpr);
          mlir::Value varVal = vc.getValue(loc, builder);

          if (!varVal.getType().isa<mlir::MemRefType>()) {
            // Create temporary memref for scalar values
            auto memrefType = mlir::MemRefType::get({}, varVal.getType());
            auto alloc =
                builder.create<mlir::memref::AllocaOp>(loc, memrefType);
            builder.create<mlir::memref::StoreOp>(loc, varVal, alloc);
            varVal = alloc;
          }

          dependVars.push_back(varVal);
          dependKindAttrs.push_back(kindAttr);
        }
      }
    } break;
    case llvm::omp::OMPC_private:
    case llvm::omp::OMPC_firstprivate: {
      /// Allocate space for private copies of the variables
      // Iterate through the variables in the clause
      for (auto *stmt : f->children()) {
        VarDecl *name = cast<VarDecl>(cast<DeclRefExpr>(stmt)->getDecl());

        // Save the original mapping
        prevInduction[name] = params[name];
        params.erase(name); // Remove from current symbol table

        bool isArray = false;
        bool LLVMABI = false;
        mlir::Type ty;

        // Determine the type of the variable
        if (Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                 name->getType()))
                .isa<mlir::LLVM::LLVMPointerType>()) {
          LLVMABI = true;
          bool undef;
          ty = Glob.getMLIRType(name->getType(), &undef);
        } else {
          ty = Glob.getMLIRType(name->getType(), &isArray);
        }

        // Allocate space for the private copy
        auto allocOp = createAllocOp(ty, name, /*memtype*/ 0,
                                     /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);

        // Add the private copy to the symbol table
        params[name] = ValueCategory(allocOp, true);

        // Handle initialization for firstprivate
        if (f->getClauseKind() == llvm::omp::OMPC_firstprivate) {
          params[name].store(loc, builder, prevInduction[name], isArray);
        }
      }
    } break;
    case llvm::omp::OMPC_shared: {
      // No action needed
    } break;
    default:
      llvm::errs() << "Unhandled OMP clause in task: "
                   << (int)f->getClauseKind() << "\n";
      task->dump();
    }
  }

  ArrayAttr dependsAttr = nullptr;
  if (!dependKindAttrs.empty()) {
    dependsAttr = builder.getArrayAttr(dependKindAttrs);
  }

  // Create the omp.task operation
  auto taskOp = builder.create<omp::TaskOp>(
      loc, ifExprVal, finalExprVal, untiedAttr, mergeableAttr, inReductionVars,
      inReductionsAttr, priorityVal, dependsAttr, dependVars, allocateVars,
      allocatorsVars);

  // Save the current insertion point and block
  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  // Add a block to the region of omp.task
  taskOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&taskOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  // Visit the body of the captured statement
  Visit(cast<CapturedStmt>(task->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  builder.create<scf::YieldOp>(loc);
  allocationScope = oldScope;
  builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    params[pair.first] = pair.second;
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPTaskLoopDirective(clang::OMPTaskLoopDirective *taskloop) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(taskloop->getBeginLoc());

  // Clause-derived attributes
  mlir::Value ifExprVal = nullptr;
  mlir::Value finalExprVal = nullptr;
  bool untiedFlag = false;
  bool mergeableFlag = false;
  bool nogroupFlag = false;
  mlir::Value priorityVal = nullptr;
  mlir::Value grainSizeValue = nullptr;
  mlir::Value numTasksValue = nullptr;

  SmallVector<VarDecl *, 4> reductionDecls;

  SmallVector<Value, 4> inReductionVars;
  ArrayAttr inReductionsAttr = nullptr;
  SmallVector<Value, 4> allocateVars;
  SmallVector<Value, 4> allocatorsVars;

  for (auto *f : taskloop->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_if: {
      auto *clause = cast<OMPIfClause>(f);
      ifExprVal = Visit(clause->getCondition()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_final: {
      auto *clause = cast<OMPFinalClause>(f);
      finalExprVal = Visit(clause->getCondition()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_untied: {
      untiedFlag = true;
    } break;
    case llvm::omp::OMPC_mergeable: {
      mergeableFlag = true;
    } break;
    case llvm::omp::OMPC_priority: {
      auto *clause = cast<OMPPriorityClause>(f);
      priorityVal = Visit(clause->getPriority()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_grainsize: {
      auto *clause = cast<OMPGrainsizeClause>(f);
      grainSizeValue = Visit(clause->getGrainsize()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_num_tasks: {
      auto *clause = cast<OMPNumTasksClause>(f);
      numTasksValue = Visit(clause->getNumTasks()).getValue(loc, builder);
    } break;
    case llvm::omp::OMPC_nogroup: {
      nogroupFlag = true;
    } break;
    case llvm::omp::OMPC_reduction: {
      auto *rc = cast<OMPReductionClause>(f);
      for (auto *expr : rc->varlists()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl()))
            reductionDecls.push_back(vd);
      }
    } break;
    default:
      llvm::errs() << "Unhandled OMP clause in taskloop: "
                   << (int)f->getClauseKind() << "\n";
      break;
    }
  }

  if (taskloop->getPreInits())
    Visit(taskloop->getPreInits());

  SmallVector<Value, 4> lowerBounds;
  for (auto *init : taskloop->inits()) {
    init = cast<clang::BinaryOperator>(init)->getRHS();
    lowerBounds.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(init).getValue(loc, builder)));
  }

  SmallVector<Value, 4> upperBounds;
  for (auto *final : taskloop->finals()) {
    final = cast<clang::BinaryOperator>(final)->getRHS();
    upperBounds.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(final).getValue(loc, builder)));
  }

  SmallVector<Value, 4> steps;
  for (auto *update : taskloop->updates()) {
    update = cast<clang::BinaryOperator>(update)->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(update))
      update = ce->getSubExpr();
    auto *add = cast<clang::BinaryOperator>(update);
    assert(add->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    auto *rhs = add->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(rhs))
      rhs = ce->getSubExpr();
    auto *mul = cast<clang::BinaryOperator>(rhs);
    assert(mul->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    auto *stepExpr = mul->getRHS();
    steps.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(stepExpr).getValue(loc, builder)));
  }

  std::map<VarDecl *, ValueCategory> prevReduction;
  SmallVector<Attribute, 4> reductionDeclSymbols;
  SmallVector<Value, 4> reductionAccumulators;
  DenseMap<VarDecl *, mlir::Value> reductionAccumulatorForVar;
  DenseMap<VarDecl *, mlir::Value> iterationTemp;
  SmallVector<VarDecl *, 4> reductionOrder;

  if (!reductionDecls.empty()) {
    for (auto *vd : reductionDecls) {
      if (params.find(vd) == params.end())
        continue;
      prevReduction[vd] = params[vd];

      bool isArray = false;
      mlir::Type elemTy = Glob.getMLIRType(vd->getType(), &isArray);
      mlir::Value sharedAddr = prevReduction[vd].val;
      if (auto mt = dyn_cast<mlir::MemRefType>(sharedAddr.getType()))
        elemTy = mt.getElementType();
      else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(sharedAddr.getType()))
        if (pt.getElementType())
          elemTy = pt.getElementType();

      auto decl =
          getOrCreateAddReductionDecl(builder, function.getOperation(), elemTy);
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

      reductionAccumulators.push_back(sharedAddr);
      reductionAccumulatorForVar[vd] = sharedAddr;
      reductionOrder.push_back(vd);
    }
  }

  mlir::ArrayAttr reductionsAttr = nullptr;
  if (!reductionDeclSymbols.empty())
    reductionsAttr = builder.getArrayAttr(reductionDeclSymbols);

  auto taskloopOp = builder.create<omp::TaskLoopOp>(
      loc, lowerBounds, upperBounds, steps,
      /*inclusive=*/false, ifExprVal, finalExprVal, untiedFlag, mergeableFlag,
      inReductionVars, inReductionsAttr, reductionAccumulators, reductionsAttr,
      priorityVal, allocateVars, allocatorsVars, grainSizeValue, numTasksValue,
      nogroupFlag);

  auto oldPoint = builder.getInsertionPoint();
  auto *oldBlock = builder.getInsertionBlock();

  taskloopOp.getRegion().push_back(new Block());
  Block &loopBlock = taskloopOp.getRegion().front();
  for (auto lb : lowerBounds)
    loopBlock.addArgument(lb.getType(), loc);
  auto loopArgs = loopBlock.getArguments();

  builder.setInsertionPointToStart(&loopBlock);
  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto it : llvm::zip(loopArgs, taskloop->counters())) {
    auto *counterExpr = cast<DeclRefExpr>(std::get<1>(it));
    VarDecl *indVar = cast<VarDecl>(counterExpr->getDecl());

    if (params.find(indVar) != params.end()) {
      prevInduction[indVar] = params[indVar];
      params.erase(indVar);
    }

    bool llvmABI = false;
    bool isArray = false;
    mlir::Type indTy = Glob.getMLIRType(indVar->getType(), &isArray);
    if (Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(indVar->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      llvmABI = true;

    auto castIdx = builder.create<IndexCastOp>(loc, indTy, std::get<0>(it));
    auto alloc = createAllocOp(indTy, indVar, /*memspace*/ 0, isArray, llvmABI);
    params[indVar] = ValueCategory(alloc, /*isRef*/ true);
    params[indVar].store(loc, builder, castIdx);
  }

  DenseMap<VarDecl *, ValueCategory> prevMappedReduction;
  if (!prevReduction.empty()) {
    for (auto &pr : prevReduction) {
      VarDecl *vd = pr.first;
      bool llvmABI = false;
      bool isArray = false;
      mlir::Type elemTy = Glob.getMLIRType(vd->getType(), &isArray);
      if (Glob.getMLIRType(
                  Glob.CGM.getContext().getLValueReferenceType(vd->getType()))
              .isa<mlir::LLVM::LLVMPointerType>())
        llvmABI = true;

      if (auto it = params.find(vd); it != params.end()) {
        prevMappedReduction[vd] = it->second;
        params.erase(it);
      }

      auto iterAlloca =
          createAllocOp(elemTy, vd, /*memspace*/ 0, isArray, llvmABI);
      iterationTemp[vd] = iterAlloca;
      params[vd] = ValueCategory(iterAlloca, /*isRef*/ true);
    }
  }

  Visit(taskloop->getBody());

  if (!prevReduction.empty()) {
    for (auto *vd : reductionOrder) {
      auto iterAlloca = iterationTemp.lookup(vd);
      auto accumulator = reductionAccumulatorForVar.lookup(vd);
      if (!iterAlloca || !accumulator)
        continue;
      ValueCategory iterVC(iterAlloca, /*isReference*/ true);
      mlir::Value produced = iterVC.getValue(loc, builder);
      builder.create<mlir::omp::ReductionOp>(loc, produced, accumulator);
    }
    for (auto &pm : prevMappedReduction)
      params[pm.first] = pm.second;
  }

  builder.create<scf::YieldOp>(loc);
  allocationScope = oldScope;
  builder.setInsertionPoint(oldBlock, oldPoint);

  for (auto &entry : prevInduction)
    params[entry.first] = entry.second;
  for (auto &entry : prevReduction)
    params[entry.first] = entry.second;

  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPTaskwaitDirective(clang::OMPTaskwaitDirective *taskwait) {
  auto loc = getMLIRLocation(taskwait->getBeginLoc());
  builder.create<omp::TaskwaitOp>(loc);
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPForDirective(clang::OMPForDirective *fors) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(fors->getBeginLoc());

  // Collect reduction variables (handle only "+" on scalars for now)
  SmallVector<VarDecl *, 4> reductionVars;
  for (auto *cl : fors->clauses()) {
    if (cl->getClauseKind() == llvm::omp::OMPC_reduction) {
      auto *rc = cast<OMPReductionClause>(cl);
      for (auto *expr : rc->varlists()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr)) {
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl()))
            reductionVars.push_back(vd);
        }
      }
    }
  }

  if (fors->getPreInits()) {
    Visit(fors->getPreInits());
  }

  SmallVector<mlir::Value> inits;
  for (auto *f : fors->inits()) {
    assert(f);
    f = cast<clang::BinaryOperator>(f)->getRHS();
    inits.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  SmallVector<mlir::Value> finals;
  for (auto *f : fors->finals()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    finals.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  SmallVector<mlir::Value> incs;
  for (auto *f : fors->updates()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    auto *bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    incs.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  // Prepare reduction metadata: keep shared lvalues and build declare ops.
  std::map<VarDecl *, ValueCategory> prevReduction;
  SmallVector<Attribute, 4> reductionDeclSymbols;
  SmallVector<Value, 4> reductionAccumulators; // accumulator addresses
  DenseMap<VarDecl *, mlir::Value> reductionAccumulatorForVar;
  DenseMap<VarDecl *, mlir::Value> iterationTemp; // per-iteration temp
  SmallVector<VarDecl *, 4> reductionOrder;
  if (!reductionVars.empty()) {
    for (auto *name : reductionVars) {
      if (params.find(name) == params.end())
        continue;

      // Save shared mapping and keep shared lvalue as accumulator.
      prevReduction[name] = params[name];

      // Determine element type of the reduction variable.
      bool isArray = false;
      mlir::Type elemTy = Glob.getMLIRType(name->getType(), &isArray);
      if (auto mt =
              dyn_cast<mlir::MemRefType>(prevReduction[name].val.getType()))
        elemTy = mt.getElementType();
      else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(
                   prevReduction[name].val.getType()))
        if (pt.getElementType())
          elemTy = pt.getElementType();

      // Create or fetch the add reduction declaration.
      auto decl =
          getOrCreateAddReductionDecl(builder, function.getOperation(), elemTy);
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

      // Use the shared address directly as the accumulator (memref or ptr).
      Value acc = prevReduction[name].val;
      reductionAccumulators.push_back(acc);
      reductionAccumulatorForVar[name] = acc;
      reductionOrder.push_back(name);
    }
  }

  auto affineOp = builder.create<omp::WsLoopOp>(loc, inits, finals, incs);
  if (!reductionDeclSymbols.empty()) {
    affineOp.setReductionsAttr(builder.getArrayAttr(reductionDeclSymbols));
    affineOp.getReductionVarsMutable().append(reductionAccumulators);
  }
  affineOp.getRegion().push_back(new Block());
  for (auto init : inits)
    affineOp.getRegion().front().addArgument(init.getType(), init.getLoc());
  auto inds = affineOp.getRegion().front().getArguments();

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  builder.create<omp::YieldOp>(loc, ValueRange());
  executeRegion.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto zp : zip(inds, fors->counters())) {
    auto idx = builder.create<IndexCastOp>(
        loc, getMLIRType(fors->getIterationVariable()->getType()),
        std::get<0>(zp));
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());

    if (params.find(name) != params.end()) {
      prevInduction[name] = params[name];
      params.erase(name);
    }

    bool LLVMABI = false;
    bool isArray = false;
    if (Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(name->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    params[name] = ValueCategory(allocop, true);
    params[name].store(loc, builder, idx);
  }

  // If reductions are present, map reduction vars to per-iteration temporaries.
  if (!prevReduction.empty()) {
    for (auto &pr : prevReduction) {
      VarDecl *name = pr.first;
      bool LLVMABI = false;
      bool isArray = false;
      mlir::Type elemTy = Glob.getMLIRType(name->getType(), &isArray);
      if (Glob.getMLIRType(
                  Glob.CGM.getContext().getLValueReferenceType(name->getType()))
              .isa<mlir::LLVM::LLVMPointerType>())
        LLVMABI = true;
      // Shadow the shared mapping inside the loop body.
      if (params.find(name) != params.end())
        params.erase(name);
      auto iterAlloca = createAllocOp(elemTy, name, /*memspace*/ 0,
                                      /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
      iterationTemp[name] = iterAlloca;
      // Remap the variable name inside the body to the per-iteration temp.
      params[name] = ValueCategory(iterAlloca, /*isRef*/ true);
    }
  }

  // TODO: set loop context.
  Visit(fors->getBody());

  // Emit omp.reduction from per-iteration temporaries to the accumulators.
  if (!prevReduction.empty()) {
    for (auto *name : reductionOrder) {
      auto iterAlloca = iterationTemp.lookup(name);
      if (!iterAlloca)
        continue;
      // Load the produced value for this iteration (memref<1xTy> at index 0).
      mlir::Value produced = builder.create<mlir::memref::LoadOp>(
          loc, iterAlloca, std::vector<mlir::Value>({getConstantIndex(0)}));
      // Accumulator address for this variable.
      mlir::Value accumulator = reductionAccumulatorForVar.lookup(name);
      if (!accumulator)
        continue;
      // Build omp.reduction operation.
      builder.create<mlir::omp::ReductionOp>(loc, produced, accumulator);
    }
  }

  builder.create<scf::YieldOp>(loc, ValueRange());

  allocationScope = oldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    params[pair.first] = pair.second;

  // Restore original mappings for reduction variables.
  for (auto &pr : prevReduction)
    params[pr.first] = pr.second;

  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPParallelDirective(clang::OMPParallelDirective *par) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(par->getBeginLoc());
  std::map<VarDecl *, ValueCategory> prevInduction;
  Value numThreads;
  for (auto *f : par->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_private:
    case llvm::omp::OMPC_firstprivate: {
      // Allocate space for private copies of the variables
      for (auto *stmt : f->children()) {
        VarDecl *name = cast<VarDecl>(cast<DeclRefExpr>(stmt)->getDecl());

        prevInduction[name] = params[name];
        params.erase(name);

        bool LLVMABI = false;
        bool isArray = false;
        mlir::Type ty;
        if (Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                 name->getType()))
                .isa<mlir::LLVM::LLVMPointerType>()) {
          LLVMABI = true;
          bool undef;
          ty = Glob.getMLIRType(name->getType(), &undef);
        } else
          ty = Glob.getMLIRType(name->getType(), &isArray);

        auto allocop = createAllocOp(ty, name, /*memtype*/ 0,
                                     /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
        params[name] = ValueCategory(allocop, true);

        // Handle initialization for firstprivate
        if (f->getClauseKind() == llvm::omp::OMPC_firstprivate) {
          params[name].store(loc, builder, prevInduction[name], isArray);
        }
      }
    }

    break;
    case llvm::omp::OMPC_num_threads: {
      auto *numThreadsClause = cast<OMPNumThreadsClause>(f);
      numThreadsClause->getNumThreads();
      numThreads =
          Visit(numThreadsClause->getNumThreads()).getValue(loc, builder);
      break;
    }
    default:
      llvm::errs() << "may not handle omp clause " << (int)f->getClauseKind()
                   << "\n";
    }
  }
  auto affineOp = builder.create<omp::ParallelOp>(
      loc, /*if_expr_var*/ Value{}, numThreads, /*allocate_vars*/ ValueRange{},
      /*allocators_vars*/ ValueRange{}, /*reduction_vars*/ ValueRange{},
      /*reductions*/ ArrayAttr{},
      /*proc_bind_val*/ omp::ClauseProcBindKindAttr{});

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  affineOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&affineOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  Visit(cast<CapturedStmt>(par->getAssociatedStmt())
            ->getCapturedDecl()
            ->getBody());

  builder.create<scf::YieldOp>(loc);
  allocationScope = oldScope;
  builder.setInsertionPoint(oldblock, oldpoint);

  for (auto pair : prevInduction)
    params[pair.first] = pair.second;
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPParallelForDirective(
    clang::OMPParallelForDirective *fors) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(fors->getBeginLoc());

  // Collect reduction variables (support '+' on scalars)
  SmallVector<VarDecl *, 4> reductionVars;

  // Handle schedule clause
  mlir::omp::ClauseScheduleKindAttr scheduleValAttr = nullptr;
  mlir::Value scheduleChunkVar = nullptr;
  mlir::omp::ScheduleModifierAttr scheduleModifierAttr = nullptr;

  for (auto *cl : fors->clauses()) {
    if (cl->getClauseKind() == llvm::omp::OMPC_reduction) {
      auto *rc = cast<OMPReductionClause>(cl);
      for (auto *expr : rc->varlists()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr)) {
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl()))
            reductionVars.push_back(vd);
        }
      }
    } else if (cl->getClauseKind() == llvm::omp::OMPC_schedule) {
      auto *scheduleClause = cast<OMPScheduleClause>(cl);

      // Map Clang schedule kind to MLIR schedule kind
      mlir::omp::ClauseScheduleKind scheduleKind;
      switch (scheduleClause->getScheduleKind()) {
      case OMPC_SCHEDULE_static:
        scheduleKind = mlir::omp::ClauseScheduleKind::Static;
        break;
      case OMPC_SCHEDULE_dynamic:
        scheduleKind = mlir::omp::ClauseScheduleKind::Dynamic;
        break;
      case OMPC_SCHEDULE_guided:
        scheduleKind = mlir::omp::ClauseScheduleKind::Guided;
        break;
      case OMPC_SCHEDULE_auto:
        scheduleKind = mlir::omp::ClauseScheduleKind::Auto;
        break;
      case OMPC_SCHEDULE_runtime:
        scheduleKind = mlir::omp::ClauseScheduleKind::Runtime;
        break;
      default:
        scheduleKind = mlir::omp::ClauseScheduleKind::Static;
        break;
      }

      scheduleValAttr = mlir::omp::ClauseScheduleKindAttr::get(
          builder.getContext(), scheduleKind);

      // Handle chunk size if present
      if (scheduleClause->getChunkSize()) {
        scheduleChunkVar =
            Visit(scheduleClause->getChunkSize()).getValue(loc, builder);
      }

      // Handle schedule modifiers if present
      // Note: Clang's OMPScheduleClause has modifiers, but we'll implement
      // basic support
      // TODO: Add proper modifier handling if needed
    }
  }

  if (fors->getPreInits()) {
    Visit(fors->getPreInits());
  }

  SmallVector<mlir::Value> inits;
  for (auto *f : fors->inits()) {
    assert(f);
    f = cast<clang::BinaryOperator>(f)->getRHS();
    inits.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  SmallVector<mlir::Value> finals;
  for (auto *f : fors->finals()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    finals.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  SmallVector<mlir::Value> incs;
  for (auto *f : fors->updates()) {
    f = cast<clang::BinaryOperator>(f)->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    auto *bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    incs.push_back(builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder)));
  }

  SmallVector<mlir::Value> inds;
  // Use scf::ParallelOp when there is no schedule clause and no reductions.
  if (!scheduleValAttr && reductionVars.empty()) {
    // Prepare per-thread private accumulators for reductions prior to the loop
    std::map<VarDecl *, ValueCategory> prevReduction;
    DenseMap<VarDecl *, mlir::Value> privateAccum;
    if (!reductionVars.empty()) {
      for (auto *name : reductionVars) {
        if (params.find(name) == params.end())
          continue;

        prevReduction[name] = params[name];
        params.erase(name);

        bool isArray = false;
        bool LLVMABI = false;
        mlir::Type elemTy;
        if (Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                 name->getType()))
                .isa<mlir::LLVM::LLVMPointerType>()) {
          LLVMABI = true;
          bool undef;
          elemTy = Glob.getMLIRType(name->getType(), &undef);
        } else {
          elemTy = Glob.getMLIRType(name->getType(), &isArray);
        }

        auto privAlloca =
            createAllocOp(elemTy, name, /*memspace*/ 0,
                          /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
        ValueCategory privVC(privAlloca, /*isRef*/ true);

        // Initialize to identity (0 for '+')
        mlir::Value zero;
        if (auto it = dyn_cast<mlir::IntegerType>(elemTy))
          zero = builder.create<ConstantIntOp>(loc, 0, it.getWidth());
        else if (auto ft = dyn_cast<mlir::FloatType>(elemTy))
          zero = builder.create<arith::ConstantOp>(
              loc, elemTy, builder.getFloatAttr(elemTy, 0.0));
        else
          zero = builder.create<ConstantIntOp>(loc, 0, 64);
        privVC.store(loc, builder, zero);

        params[name] = privVC;
        privateAccum[name] = privAlloca;
      }
    }

    SmallVector<mlir::Value> steps(incs.begin(), incs.end());
    auto parallelOp =
        builder.create<scf::ParallelOp>(loc, inits, finals, steps);
    inds = parallelOp.getInductionVars();

    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();

    builder.setInsertionPointToStart(&parallelOp.getRegion().front());

    auto executeRegion =
        builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
    executeRegion.getRegion().push_back(new Block());
    builder.setInsertionPointToStart(&executeRegion.getRegion().back());

    auto *oldScope = allocationScope;
    allocationScope = &executeRegion.getRegion().back();

    // Handle induction variables
    std::map<VarDecl *, ValueCategory> prevInduction;
    for (auto zp : zip(inds, fors->counters())) {
      auto idx = builder.create<IndexCastOp>(
          loc, getMLIRType(fors->getIterationVariable()->getType()),
          std::get<0>(zp));
      VarDecl *name =
          cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());

      if (params.find(name) != params.end()) {
        prevInduction[name] = params[name];
        params.erase(name);
      }

      bool LLVMABI = false;
      bool isArray = false;
      if (Glob.getMLIRType(
                  Glob.CGM.getContext().getLValueReferenceType(name->getType()))
              .isa<mlir::LLVM::LLVMPointerType>())
        LLVMABI = true;
      else
        Glob.getMLIRType(name->getType(), &isArray);

      auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                   /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
      params[name] = ValueCategory(allocop, true);
      params[name].store(loc, builder, idx);
    }

    // Visit the loop body
    Visit(fors->getBody());

    builder.create<scf::YieldOp>(loc);
    allocationScope = oldScope;
    builder.setInsertionPoint(oldblock, oldpoint);

    // Restore previous variable mappings
    for (auto pair : prevInduction)
      params[pair.first] = pair.second;

    // Combine private accumulators back into shared variables using
    // omp.atomic.update
    if (!prevReduction.empty()) {
      for (auto &pr : prevReduction) {
        VarDecl *name = pr.first;
        auto sharedVC = pr.second; // original shared lvalue (memref or pointer)
        auto priv = privateAccum.lookup(name);
        if (!priv)
          continue;

        // Load the thread-private accumulator value
        auto privVal = builder.create<mlir::memref::LoadOp>(
            loc, priv, std::vector<mlir::Value>({getConstantIndex(0)}));

        // Build omp.atomic.update on the shared address
        mlir::Value xAddr = sharedVC.val;

        // Determine the element type for the update region block argument
        mlir::Type elemTy;
        if (auto mt = dyn_cast<mlir::MemRefType>(xAddr.getType()))
          elemTy = mt.getElementType();
        else if (auto pt =
                     dyn_cast<mlir::LLVM::LLVMPointerType>(xAddr.getType()))
          elemTy =
              pt.getElementType() ? pt.getElementType() : privVal.getType();
        else
          elemTy = privVal.getType();

        auto aupd = builder.create<omp::AtomicUpdateOp>(
            loc, xAddr, mlir::IntegerAttr(),
            mlir::omp::ClauseMemoryOrderKindAttr());
        auto &reg = aupd.getRegion();
        auto *body = new Block();
        body->addArgument(elemTy, loc);
        reg.push_back(body);
        {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(body);
          mlir::Value cur = body->getArgument(0);
          mlir::Value sum;
          mlir::Value curVal = cur;
          if (elemTy.isIntOrIndex())
            sum = builder.create<arith::AddIOp>(loc, curVal, privVal);
          else if (elemTy.isa<mlir::FloatType>())
            sum = builder.create<arith::AddFOp>(loc, curVal, privVal);
          else
            sum = cur; // unsupported type: no-op
          builder.create<omp::YieldOp>(loc, sum);
        }

        // Restore shared mapping for subsequent code
        params[name] = sharedVC;
      }
    }

  } else {
    // Use OpenMP parallel worksharing when scheduling clauses or reductions are
    // present.
    std::map<VarDecl *, ValueCategory> prevReduction;
    SmallVector<Attribute, 4> reductionDeclSymbols;
    SmallVector<Value, 4> reductionAccumulators;
    DenseMap<VarDecl *, mlir::Value> reductionAccumulatorForVar;
    SmallVector<VarDecl *, 4> reductionOrder;
    if (!reductionVars.empty()) {
      for (auto *name : reductionVars) {
        if (params.find(name) == params.end())
          continue;

        prevReduction[name] = params[name];
        params.erase(name);

        bool isArray = false;
        mlir::Type elemTy = Glob.getMLIRType(name->getType(), &isArray);
        mlir::Value sharedAddr = prevReduction[name].val;
        if (auto mt = dyn_cast<mlir::MemRefType>(sharedAddr.getType()))
          elemTy = mt.getElementType();
        else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(
                     sharedAddr.getType()))
          if (pt.getElementType())
            elemTy = pt.getElementType();

        auto decl = getOrCreateAddReductionDecl(
            builder, function.getOperation(), elemTy);
        reductionDeclSymbols.push_back(
            SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

        reductionAccumulators.push_back(sharedAddr);
        reductionAccumulatorForVar[name] = sharedAddr;
        reductionOrder.push_back(name);
      }
    }

    auto parallelOp = builder.create<omp::ParallelOp>(
        loc, /*if_expr_var*/ Value{}, /*num_threads*/ Value{},
        /*allocate_vars*/ ValueRange{}, /*allocators_vars*/ ValueRange{},
        /*reduction_vars*/ ValueRange{}, /*reductions*/ ArrayAttr{},
        /*proc_bind_val*/ omp::ClauseProcBindKindAttr{});

    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();

    parallelOp.getRegion().push_back(new Block());
    builder.setInsertionPointToStart(&parallelOp.getRegion().front());

    auto executeRegion =
        builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
    executeRegion.getRegion().push_back(new Block());
    builder.create<omp::TerminatorOp>(loc);
    builder.setInsertionPointToStart(&executeRegion.getRegion().back());

    auto *oldScope = allocationScope;
    allocationScope = &executeRegion.getRegion().back();

    DenseMap<VarDecl *, mlir::Value> iterationTemp;

    auto wsLoopOp = builder.create<omp::WsLoopOp>(
        loc, inits, finals, incs,
        /*linear_vars*/ ValueRange{},
        /*linear_step_vars*/ ValueRange{},
        /*reduction_vars*/ ValueRange{},
        /*reductions*/ nullptr,
        /*schedule_val*/ scheduleValAttr,
        /*schedule_chunk_var*/ scheduleChunkVar,
        /*schedule_modifier*/ scheduleModifierAttr,
        /*simd_modifier*/ nullptr,
        /*nowait*/ nullptr,
        /*ordered_val*/ nullptr,
        /*order_val*/ nullptr,
        /*inclusive*/ nullptr);

    if (!reductionDeclSymbols.empty()) {
      wsLoopOp.setReductionsAttr(builder.getArrayAttr(reductionDeclSymbols));
      wsLoopOp.getReductionVarsMutable().append(reductionAccumulators);
    }

    wsLoopOp.getRegion().push_back(new Block());
    for (auto init : inits)
      wsLoopOp.getRegion().front().addArgument(init.getType(), init.getLoc());
    auto wsLoopInds = wsLoopOp.getRegion().front().getArguments();
    inds.assign(wsLoopInds.begin(), wsLoopInds.end());

    auto wsLoopOldpoint = builder.getInsertionPoint();
    auto *wsLoopOldblock = builder.getInsertionBlock();

    builder.setInsertionPointToStart(&wsLoopOp.getRegion().front());

    auto wsLoopExecuteRegion =
        builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
    builder.create<omp::YieldOp>(loc, ValueRange());
    wsLoopExecuteRegion.getRegion().push_back(new Block());
    builder.setInsertionPointToStart(&wsLoopExecuteRegion.getRegion().back());

    auto *wsLoopOldScope = allocationScope;
    allocationScope = &wsLoopExecuteRegion.getRegion().back();

    std::map<VarDecl *, ValueCategory> prevInduction;
    for (auto zp : zip(inds, fors->counters())) {
      auto idx = builder.create<IndexCastOp>(
          loc, getMLIRType(fors->getIterationVariable()->getType()),
          std::get<0>(zp));
      VarDecl *name =
          cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());

      if (params.find(name) != params.end()) {
        prevInduction[name] = params[name];
        params.erase(name);
      }

      bool LLVMABI = false;
      bool isArray = false;
      if (Glob.getMLIRType(
                  Glob.CGM.getContext().getLValueReferenceType(name->getType()))
              .isa<mlir::LLVM::LLVMPointerType>())
        LLVMABI = true;
      else
        Glob.getMLIRType(name->getType(), &isArray);

      auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                   /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
      params[name] = ValueCategory(allocop, true);
      params[name].store(loc, builder, idx);
    }

    DenseMap<VarDecl *, ValueCategory> prevMappedReduction;
    if (!prevReduction.empty()) {
      for (auto &pr : prevReduction) {
        VarDecl *name = pr.first;

        bool LLVMABI = false;
        bool isArray = false;
        mlir::Type elemTy = Glob.getMLIRType(name->getType(), &isArray);
        if (Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                 name->getType()))
                .isa<mlir::LLVM::LLVMPointerType>())
          LLVMABI = true;

        if (params.find(name) != params.end())
          prevMappedReduction[name] = params[name];

        auto iterAlloca =
            createAllocOp(elemTy, name, /*memspace*/ 0,
                          /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
        iterationTemp[name] = iterAlloca;
        params[name] = ValueCategory(iterAlloca, /*isRef*/ true);
      }
    }

    Visit(fors->getBody());

    if (!prevReduction.empty()) {
      for (auto *name : reductionOrder) {
        auto iterAlloca = iterationTemp.lookup(name);
        auto accumulator = reductionAccumulatorForVar.lookup(name);
        if (!iterAlloca || !accumulator)
          continue;
        ValueCategory iterVC(iterAlloca, /*isReference*/ true);
        mlir::Value produced = iterVC.getValue(loc, builder);
        builder.create<mlir::omp::ReductionOp>(loc, produced, accumulator);
      }
      for (auto &pm : prevMappedReduction)
        params[pm.first] = pm.second;
    }

    builder.create<scf::YieldOp>(loc);
    allocationScope = wsLoopOldScope;
    builder.setInsertionPoint(wsLoopOldblock, wsLoopOldpoint);

    for (auto pair : prevInduction)
      params[pair.first] = pair.second;

    builder.create<scf::YieldOp>(loc);
    allocationScope = oldScope;
    builder.setInsertionPoint(oldblock, oldpoint);

    for (auto &pr : prevReduction)
      params[pr.first] = pr.second;
  }

  return nullptr;
}

ValueCategory MLIRScanner::VisitDoStmt(clang::DoStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getDoLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto *toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::cf::BranchOp>(loc, &bodyB);

  builder.setInsertionPointToStart(&condB);

  if (auto *s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(loc, builder);
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();

  builder.create<mlir::cf::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitWhileStmt(clang::WhileStmt *stmt) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(stmt->getLParenLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto *toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::cf::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto declStmt = stmt->getConditionVariableDeclStmt())
    Visit(declStmt);
  if (auto *s = stmt->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(loc, builder);
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, 0, ty));
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::cf::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(stmt->getBody());
  loops.pop_back();

  builder.create<mlir::cf::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getIfLoc());
  if (auto declStmt = stmt->getConditionVariableDeclStmt())
    Visit(declStmt);
  auto cond = Visit(stmt->getCond()).getValue(loc, builder);
  assert(cond != nullptr && "must be a non-null");

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();
  if (auto LT = dyn_cast<MemRefType>(cond.getType())) {
    cond = builder.create<polygeist::Memref2PointerOp>(
        loc, LLVM::LLVMPointerType::get(builder.getI8Type()), cond);
  }
  if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
    auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  if (!cond.getType().isa<mlir::IntegerType>()) {
    stmt->dump();
    llvm::errs() << " cond: " << cond << " ct: " << cond.getType() << "\n";
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    cond = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, cond,
        builder.create<ConstantIntOp>(loc, 0, prevTy));
  }
  bool hasElseRegion = stmt->getElse();
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);

  ifOp.getThenRegion().back().clear();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
  Visit(stmt->getThen());
  builder.create<scf::YieldOp>(loc);
  if (hasElseRegion) {
    ifOp.getElseRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    Visit(stmt->getElse());
    builder.create<scf::YieldOp>(loc);
  }

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitSwitchStmt(clang::SwitchStmt *stmt) {
  IfScope scope(*this);
  auto cond = Visit(stmt->getCond())
                  .getValue(getMLIRLocation(stmt->getSwitchLoc()), builder);
  assert(cond != nullptr);
  SmallVector<int64_t> caseVals;

  auto er = builder.create<scf::ExecuteRegionOp>(
      getMLIRLocation(stmt->getSwitchLoc()), ArrayRef<mlir::Type>());
  er.getRegion().push_back(new Block());
  auto oldpoint2 = builder.getInsertionPoint();
  auto *oldblock2 = builder.getInsertionBlock();

  auto &exitB = *(new Block());
  builder.setInsertionPointToStart(&exitB);
  builder.create<scf::YieldOp>(getMLIRLocation(stmt->getSwitchLoc()));
  builder.setInsertionPointToStart(&exitB);

  SmallVector<Block *> blocks;
  bool inCase = false;

  Block *defaultB = &exitB;

  for (auto *cse : stmt->getBody()->children()) {
    if (auto *cses = dyn_cast<CaseStmt>(cse)) {
      auto loc = getMLIRLocation(cses->getCaseLoc());
      auto &condB = *(new Block());

      auto cval = Visit(cses->getLHS());
      if (!cval.val) {
        cses->getLHS()->dump();
      }
      assert(cval.val);
      auto cint = cval.getValue(loc, builder).getDefiningOp<ConstantIntOp>();
      if (!cint) {
        cses->getLHS()->dump();
        llvm::errs() << "cval: " << cval.val << "\n";
      }
      assert(cint);
      caseVals.push_back(cint.value());

      if (inCase) {
        auto noBreak =
            builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<mlir::cf::CondBranchOp>(loc, noBreak, &condB, &exitB);
        loops.pop_back();
      }

      inCase = true;
      er.getRegion().getBlocks().push_back(&condB);
      blocks.push_back(&condB);
      builder.setInsertionPointToStart(&condB);

      auto i1Ty = builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = builder.create<ConstantIntOp>(loc, true, 1);
      loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                       builder.create<mlir::memref::AllocaOp>(loc, type)});
      builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
      builder.create<mlir::memref::StoreOp>(loc, truev,
                                            loops.back().keepRunning);
      Visit(cses->getSubStmt());
    } else if (auto *cses = dyn_cast<DefaultStmt>(cse)) {
      auto loc = getMLIRLocation(cses->getDefaultLoc());
      auto &condB = *(new Block());

      if (inCase) {
        auto noBreak =
            builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<mlir::cf::CondBranchOp>(loc, noBreak, &condB, &exitB);
        loops.pop_back();
      }

      inCase = true;
      er.getRegion().getBlocks().push_back(&condB);
      builder.setInsertionPointToStart(&condB);

      auto i1Ty = builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = builder.create<ConstantIntOp>(loc, true, 1);
      loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                       builder.create<mlir::memref::AllocaOp>(loc, type)});
      builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
      builder.create<mlir::memref::StoreOp>(loc, truev,
                                            loops.back().keepRunning);
      defaultB = &condB;
      Visit(cses->getSubStmt());
    } else {
      Visit(cse);
    }
  }

  if (caseVals.size() == 0) {
    delete &exitB;
    er.erase();
    builder.setInsertionPoint(oldblock2, oldpoint2);
    return nullptr;
  }

  if (inCase)
    loops.pop_back();
  auto loc = getMLIRLocation(stmt->getSwitchLoc());
  builder.create<mlir::cf::BranchOp>(loc, &exitB);

  er.getRegion().getBlocks().push_back(&exitB);

  DenseIntElementsAttr caseValuesAttr;
  ShapedType caseValueType = mlir::VectorType::get(
      static_cast<int64_t>(caseVals.size()), cond.getType());
  auto ity = cond.getType().cast<mlir::IntegerType>();
  if (ity.getWidth() == 64)
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals);
  else if (ity.getWidth() == 32) {
    SmallVector<int32_t> caseVals32;
    for (auto v : caseVals)
      caseVals32.push_back((int32_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals32);
  } else if (ity.getWidth() == 16) {
    SmallVector<int16_t> caseVals16;
    for (auto v : caseVals)
      caseVals16.push_back((int16_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals16);
  } else {
    assert(ity.getWidth() == 8);
    SmallVector<int8_t> caseVals8;
    for (auto v : caseVals)
      caseVals8.push_back((int8_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals8);
  }

  builder.setInsertionPointToStart(&er.getRegion().front());
  builder.create<mlir::cf::SwitchOp>(
      loc, cond, defaultB, ArrayRef<mlir::Value>(), caseValuesAttr, blocks,
      SmallVector<mlir::ValueRange>(caseVals.size(), ArrayRef<mlir::Value>()));
  builder.setInsertionPoint(oldblock2, oldpoint2);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  IfScope scope(*this);
  for (auto *sub : decl->decls()) {
    if (auto *vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else if (isa<TypeAliasDecl, RecordDecl, StaticAssertDecl, TypedefDecl,
                   UsingDecl, UsingDirectiveDecl, EnumConstantDecl, EnumDecl>(
                   sub)) {
    } else {
      emitError(getMLIRLocation(decl->getBeginLoc()))
          << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      llvm_unreachable("unknown sub decl");
    }
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitAttributedStmt(AttributedStmt *AS) {
  emitWarning(getMLIRLocation(AS->getAttrLoc())) << "ignoring attributes\n";
  return Visit(AS->getSubStmt());
}

ValueCategory MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto *a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getBreakLoc());
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  assert(loops.back().noBreak && "no break false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().noBreak);

  return nullptr;
}

ValueCategory MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getContinueLoc());
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueCategory MLIRScanner::VisitLabelStmt(clang::LabelStmt *stmt) {
  auto loc = getMLIRLocation(stmt->getIdentLoc());
  auto *toadd = builder.getInsertionBlock()->getParent();
  Block *labelB;
  auto found = labels.find(stmt);
  if (found != labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    labels[stmt] = labelB;
  }
  toadd->getBlocks().push_back(labelB);
  builder.create<mlir::cf::BranchOp>(loc, labelB);
  builder.setInsertionPointToStart(labelB);
  Visit(stmt->getSubStmt());
  return nullptr;
}

ValueCategory MLIRScanner::VisitGotoStmt(clang::GotoStmt *stmt) {
  auto loc = getMLIRLocation(stmt->getGotoLoc());
  auto *labelstmt = stmt->getLabel()->getStmt();
  Block *labelB;
  auto found = labels.find(labelstmt);
  if (found != labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    labels[labelstmt] = labelB;
  }
  builder.create<mlir::cf::BranchOp>(loc, labelB);
  return nullptr;
}

ValueCategory MLIRScanner::VisitCXXTryStmt(clang::CXXTryStmt *stmt) {
  llvm::errs() << "warning, not performing catches for try: ";
  stmt->dump();
  return Visit(stmt->getTryBlock());
}

ValueCategory MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  bool isArrayReturn = false;
  Glob.getMLIRType(EmittingFunctionDecl->getReturnType(), &isArrayReturn);

  auto loc = getMLIRLocation(stmt->getReturnLoc());

  if (isArrayReturn) {
    auto rv = Visit(stmt->getRetValue());
    assert(rv.val && "expect right value to be valid");
    assert(rv.isReference && "right value must be a reference");
    auto op = function.getArgument(function.getNumArguments() - 1);
    assert(rv.val.getType().cast<MemRefType>().getElementType() ==
               op.getType().cast<MemRefType>().getElementType() &&
           "type mismatch");
    assert(op.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape()[1] ==
           op.getType().cast<MemRefType>().getShape()[1]);

    for (int i = 0; i < op.getType().cast<MemRefType>().getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, rv.val, idx), op, idx);
    }
  } else if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue());
    if (!stmt->getRetValue()->getType()->isVoidType()) {
      if (!rv.val) {
        stmt->dump();
      }
      assert(rv.val && "expect right value to be valid");

      mlir::Value val;
      if (stmt->getRetValue()->isLValue() || stmt->getRetValue()->isXValue()) {
        assert(rv.isReference);
        val = rv.val;
      } else {
        val = rv.getValue(loc, builder);
      }

      auto postTy = returnVal.getType().cast<MemRefType>().getElementType();
      if (auto prevTy = dyn_cast<mlir::IntegerType>(val.getType())) {
        auto ipostTy = postTy.cast<mlir::IntegerType>();
        if (prevTy != ipostTy) {
          val = builder.create<arith::TruncIOp>(loc, ipostTy, val);
        }
      } else if (val.getType().isa<MemRefType>() &&
                 postTy.isa<LLVM::LLVMPointerType>())
        val = builder.create<polygeist::Memref2PointerOp>(loc, postTy, val);
      else if (val.getType().isa<LLVM::LLVMPointerType>() &&
               postTy.isa<MemRefType>())
        val = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, val);
      if (postTy != val.getType()) {
        stmt->dump();
        llvm::errs() << " val: " << val << " postTy: " << postTy
                     << " rv.val: " << rv.val << " rv.isRef"
                     << (int)rv.isReference << " mm: "
                     << (int)(stmt->getRetValue()->isLValue() ||
                              stmt->getRetValue()->isXValue())
                     << "\n";
      }
      assert(postTy == val.getType());
      builder.create<mlir::memref::StoreOp>(loc, val, returnVal);
    }
  }

  assert(loops.size() && "must be non-empty");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  for (auto l : loops) {
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.keepRunning);
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.noBreak);
  }

  return nullptr;
}
