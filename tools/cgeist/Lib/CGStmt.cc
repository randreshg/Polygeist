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

#define DEBUG_TYPE "CGStmt"

using namespace mlir;
using namespace mlir::arith;

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

/// Reduction operator kind for OMP reduction clause.
enum class ReductionKind { Add, Mul, And, Or, Xor, LAnd, LOr, Min, Max };

/// Map Clang OMP reduction clause to ReductionKind using the operator name.
static ReductionKind getReductionKind(clang::OMPReductionClause *rc) {
  auto nameInfo = rc->getNameInfo();
  auto opKind = nameInfo.getName().getCXXOverloadedOperator();
  switch (opKind) {
  case clang::OO_Plus:      return ReductionKind::Add;
  case clang::OO_Star:      return ReductionKind::Mul;
  case clang::OO_Amp:       return ReductionKind::And;
  case clang::OO_Pipe:      return ReductionKind::Or;
  case clang::OO_Caret:     return ReductionKind::Xor;
  case clang::OO_AmpAmp:    return ReductionKind::LAnd;
  case clang::OO_PipePipe:  return ReductionKind::LOr;
  case clang::OO_Less:      return ReductionKind::Min;  // <
  case clang::OO_Greater:   return ReductionKind::Max;  // >
  default: break;
  }
  // For named reductions (min/max), check the identifier name.
  if (auto *id = nameInfo.getName().getAsIdentifierInfo()) {
    llvm::StringRef name = id->getName();
    if (name == "min") return ReductionKind::Min;
    if (name == "max") return ReductionKind::Max;
  }
  llvm::errs() << "Warning: unsupported reduction operator, defaulting to Add\n";
  return ReductionKind::Add;
}

/// Create an identity value for the given reduction kind and scalar type.
static Value createIdentityValue(OpBuilder &b, Location loc,
                                 ReductionKind kind, mlir::Type scalarTy,
                                 bool isSigned) {
  bool isInt = isa<IntegerType>(scalarTy);
  bool isIndex = isa<IndexType>(scalarTy);
  bool isFloat = isa<FloatType>(scalarTy);
  unsigned bw = 64;
  if (auto it = dyn_cast<IntegerType>(scalarTy))
    bw = it.getWidth();

  switch (kind) {
  case ReductionKind::Add:
  case ReductionKind::Or:
  case ReductionKind::Xor:
  case ReductionKind::LOr:
    if (isInt)   return b.create<ConstantIntOp>(loc, 0, bw);
    if (isIndex) return b.create<arith::ConstantIndexOp>(loc, 0);
    if (isFloat) return b.create<arith::ConstantOp>(
                     loc, scalarTy, b.getFloatAttr(scalarTy, 0.0));
    break;
  case ReductionKind::Mul:
  case ReductionKind::LAnd:
    if (isInt)   return b.create<ConstantIntOp>(loc, 1, bw);
    if (isIndex) return b.create<arith::ConstantIndexOp>(loc, 1);
    if (isFloat) return b.create<arith::ConstantOp>(
                     loc, scalarTy, b.getFloatAttr(scalarTy, 1.0));
    break;
  case ReductionKind::And:
    if (isInt) return b.create<ConstantIntOp>(loc, -1, bw);
    break;
  case ReductionKind::Min:
    if (isInt) {
      auto val = isSigned ? APInt::getSignedMaxValue(bw)
                          : APInt::getMaxValue(bw);
      return b.create<ConstantIntOp>(loc, val.getSExtValue(), bw);
    }
    if (isFloat) {
      auto ft = cast<FloatType>(scalarTy);
      return b.create<arith::ConstantOp>(loc, scalarTy,
          b.getFloatAttr(scalarTy,
              APFloat::getLargest(ft.getFloatSemantics()).convertToDouble()));
    }
    break;
  case ReductionKind::Max:
    if (isInt) {
      if (isSigned)
        return b.create<ConstantIntOp>(
            loc, APInt::getSignedMinValue(bw).getSExtValue(), bw);
      else
        return b.create<ConstantIntOp>(loc, 0, bw);
    }
    if (isFloat) {
      auto ft = cast<FloatType>(scalarTy);
      return b.create<arith::ConstantOp>(loc, scalarTy,
          b.getFloatAttr(scalarTy,
              -APFloat::getLargest(ft.getFloatSemantics()).convertToDouble()));
    }
    break;
  }
  llvm::report_fatal_error("createIdentityValue: unsupported type/kind combination");
}

/// Create a combiner operation: result = lhs <op> rhs.
static Value createCombinerOp(OpBuilder &b, Location loc, ReductionKind kind,
                              Value lhs, Value rhs, mlir::Type elementTy,
                              bool isSigned) {
  bool isIntLike = elementTy.isIntOrIndex();
  bool isFloat = isa<FloatType>(elementTy);
  switch (kind) {
  case ReductionKind::Add:
    if (isIntLike) return b.create<arith::AddIOp>(loc, lhs, rhs);
    if (isFloat)   return b.create<arith::AddFOp>(loc, lhs, rhs);
    break;
  case ReductionKind::Mul:
    if (isIntLike) return b.create<arith::MulIOp>(loc, lhs, rhs);
    if (isFloat)   return b.create<arith::MulFOp>(loc, lhs, rhs);
    break;
  case ReductionKind::And:
    if (isIntLike) return b.create<arith::AndIOp>(loc, lhs, rhs);
    break;
  case ReductionKind::Or:
    if (isIntLike) return b.create<arith::OrIOp>(loc, lhs, rhs);
    break;
  case ReductionKind::Xor:
    if (isIntLike) return b.create<arith::XOrIOp>(loc, lhs, rhs);
    break;
  case ReductionKind::LAnd: {
    if (isIntLike) {
      auto zero = b.create<arith::ConstantOp>(
          loc, lhs.getType(), b.getIntegerAttr(lhs.getType(), 0));
      auto lnz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                          lhs, zero);
      auto rnz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                          rhs, zero);
      auto both = b.create<arith::AndIOp>(loc, lnz, rnz);
      return b.create<arith::ExtUIOp>(loc, lhs.getType(), both);
    }
    break;
  }
  case ReductionKind::LOr: {
    if (isIntLike) {
      auto zero = b.create<arith::ConstantOp>(
          loc, lhs.getType(), b.getIntegerAttr(lhs.getType(), 0));
      auto lnz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                          lhs, zero);
      auto rnz = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                          rhs, zero);
      auto either = b.create<arith::OrIOp>(loc, lnz, rnz);
      return b.create<arith::ExtUIOp>(loc, lhs.getType(), either);
    }
    break;
  }
  case ReductionKind::Min:
    if (isFloat) return b.create<arith::MinimumFOp>(loc, lhs, rhs);
    if (isIntLike)
      return isSigned ? (Value)b.create<arith::MinSIOp>(loc, lhs, rhs)
                      : (Value)b.create<arith::MinUIOp>(loc, lhs, rhs);
    break;
  case ReductionKind::Max:
    if (isFloat) return b.create<arith::MaximumFOp>(loc, lhs, rhs);
    if (isIntLike)
      return isSigned ? (Value)b.create<arith::MaxSIOp>(loc, lhs, rhs)
                      : (Value)b.create<arith::MaxUIOp>(loc, lhs, rhs);
    break;
  }
  llvm::report_fatal_error("createCombinerOp: unsupported type/kind combination");
}

/// Create or fetch an OpenMP private/firstprivate recipe for the given type.
/// Returns the PrivateClauseOp symbol. Inserted at module scope.
static mlir::omp::PrivateClauseOp
getOrCreatePrivatizer(mlir::OpBuilder &builder, mlir::Operation *anchor,
                      mlir::Type varTy, bool isFirstPrivate) {
  auto *ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();
  const char *prefix = isFirstPrivate ? "firstprivate" : "private";

  std::string symName;
  {
    llvm::raw_string_ostream os(symName);
    os << prefix << "_";
    varTy.print(os);
  }
  // Sanitize name (replace special chars).
  for (auto &c : symName)
    if (c == '<' || c == '>' || c == ',' || c == ' ' || c == '?')
      c = '_';

  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  if (auto existing = module.lookupSymbol<omp::PrivateClauseOp>(symName))
    return existing;

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());

  // Use !llvm.ptr for IsolatedFromAbove compatibility (same as DeclareReductionOp).
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto dsType = isFirstPrivate
      ? omp::DataSharingClauseType::FirstPrivate
      : omp::DataSharingClauseType::Private;
  auto privOp = omp::PrivateClauseOp::create(
      builder, loc, builder.getStringAttr(symName),
      mlir::TypeAttr::get(ptrTy),
      omp::DataSharingClauseTypeAttr::get(ctx, dsType));

  // Determine the LLVM type for alloca.
  mlir::Type allocaTy;
  if (auto mt = dyn_cast<MemRefType>(varTy)) {
    if (mt.hasStaticShape())
      allocaTy = LLVM::LLVMArrayType::get(mt.getElementType(),
                                            mt.getNumElements());
    else
      allocaTy = mt.getElementType(); // dynamic: single element
  } else {
    allocaTy = varTy;
  }

  // Init region: alloca + yield.
  {
    Region &init = privOp.getInitRegion();
    Block *initBlock = new Block();
    init.push_back(initBlock);
    initBlock->addArgument(ptrTy, loc);
    initBlock->addArgument(ptrTy, loc);
    OpBuilder ib(ctx);
    ib.setInsertionPointToStart(initBlock);
    ib.create<omp::YieldOp>(loc, ValueRange{initBlock->getArgument(1)});
  }

  // Copy region: only for firstprivate. Element-wise copy via LLVM ops.
  if (isFirstPrivate) {
    Region &copy = privOp.getCopyRegion();
    Block *copyBlock = new Block();
    copy.push_back(copyBlock);
    copyBlock->addArgument(ptrTy, loc);
    copyBlock->addArgument(ptrTy, loc);
    OpBuilder cb(ctx);
    cb.setInsertionPointToStart(copyBlock);
    Value orig = copyBlock->getArgument(0);
    Value priv = copyBlock->getArgument(1);

    if (auto mt = dyn_cast<MemRefType>(varTy)) {
      if (mt.hasStaticShape()) {
        auto arrTy = LLVM::LLVMArrayType::get(mt.getElementType(),
                                                mt.getNumElements());
        auto zero = cb.create<LLVM::ConstantOp>(loc, cb.getI32Type(),
                                                 cb.getI32IntegerAttr(0));
        for (int64_t i = 0; i < mt.getNumElements(); i++) {
          auto idx = cb.create<LLVM::ConstantOp>(loc, cb.getI32Type(),
                                                  cb.getI32IntegerAttr(i));
          SmallVector<LLVM::GEPArg> args = {LLVM::GEPArg(zero),
                                             LLVM::GEPArg(idx)};
          auto origPtr = cb.create<LLVM::GEPOp>(loc, ptrTy, arrTy,
                                                 orig, args);
          auto privPtr = cb.create<LLVM::GEPOp>(loc, ptrTy, arrTy,
                                                 priv, args);
          auto val = cb.create<LLVM::LoadOp>(loc, mt.getElementType(),
                                              origPtr);
          cb.create<LLVM::StoreOp>(loc, val, privPtr);
        }
      } else {
        // Dynamic: copy single element.
        auto val = cb.create<LLVM::LoadOp>(loc, mt.getElementType(), orig);
        cb.create<LLVM::StoreOp>(loc, val, priv);
      }
    } else {
      auto val = cb.create<LLVM::LoadOp>(loc, varTy, orig);
      cb.create<LLVM::StoreOp>(loc, val, priv);
    }
    cb.create<omp::YieldOp>(loc, ValueRange{priv});
  }

  return privOp;
}

/// Create or fetch an OpenMP reduction declaration for the given operator and
/// type. For memref types, creates a byref array reduction with element-wise
/// init and combiner loops. The declaration is inserted at module scope.
static mlir::omp::DeclareReductionOp
getOrCreateReductionDecl(mlir::OpBuilder &builder, mlir::Operation *anchor,
                         mlir::Type elementTy, ReductionKind kind,
                         bool isSigned = true) {
  auto *ctx = builder.getContext();
  auto loc = builder.getUnknownLoc();

  // Determine scalar type and whether this is an array (memref) reduction.
  mlir::Type scalarTy = elementTy;
  auto memrefTy = dyn_cast<mlir::MemRefType>(elementTy);
  if (memrefTy)
    scalarTy = memrefTy.getElementType();

  // Build a stable symbol name: <kind>_<type>[_<shape>]
  const char *prefix = "add";
  switch (kind) {
  case ReductionKind::Add:  prefix = "add"; break;
  case ReductionKind::Mul:  prefix = "mul"; break;
  case ReductionKind::And:  prefix = "and"; break;
  case ReductionKind::Or:   prefix = "or";  break;
  case ReductionKind::Xor:  prefix = "xor"; break;
  case ReductionKind::LAnd: prefix = "land"; break;
  case ReductionKind::LOr:  prefix = "lor";  break;
  case ReductionKind::Min:  prefix = isSigned ? "min" : "umin"; break;
  case ReductionKind::Max:  prefix = isSigned ? "max" : "umax"; break;
  }

  std::string symName;
  if (auto it = dyn_cast<IntegerType>(scalarTy))
    symName = (Twine(prefix) + "_i" + Twine(it.getWidth())).str();
  else if (isa<IndexType>(scalarTy))
    symName = (Twine(prefix) + "_index").str();
  else if (auto ft = dyn_cast<FloatType>(scalarTy))
    symName = (Twine(prefix) + "_f" + Twine(ft.getWidth())).str();
  else {
    std::string tyStr;
    { llvm::raw_string_ostream os(tyStr); scalarTy.print(os); }
    symName = (Twine(prefix) + "_" + tyStr).str();
  }
  if (memrefTy) {
    symName += "_memref";
    for (auto dim : memrefTy.getShape())
      symName += "_" + std::to_string(dim);
  }

  // Find the containing module.
  ModuleOp module = anchor->getParentOfType<ModuleOp>();
  assert(module && "expected to be inside a ModuleOp");

  if (auto existing =
          module.lookupSymbol<mlir::omp::DeclareReductionOp>(symName))
    return existing;

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(module.getBody());

  // For array reductions, use !llvm.ptr with byref. DeclareReductionOp is
  // IsolatedFromAbove, so its regions use LLVM dialect ops directly.
  // This avoids ConvertPolygeistToLLVM needing to convert memref/scf ops
  // inside an IsolatedFromAbove context (which crashes).
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  mlir::Type declType = memrefTy ? (mlir::Type)ptrTy : elementTy;
  mlir::TypeAttr byrefAttr = memrefTy
      ? mlir::TypeAttr::get(scalarTy)
      : mlir::TypeAttr{};
  auto decl = builder.create<mlir::omp::DeclareReductionOp>(
      loc, builder.getStringAttr(symName),
      mlir::TypeAttr::get(declType), byrefAttr);

  if (memrefTy) {
    int64_t numElems = memrefTy.getNumElements();
    auto arrayTy = LLVM::LLVMArrayType::get(scalarTy, numElems);

    // --- Array reduction: alloc region (LLVM dialect) ---
    {
      Region &alloc = decl.getAllocRegion();
      Block *allocBlock = new Block();
      alloc.push_back(allocBlock);
      allocBlock->addArgument(ptrTy, loc);
      OpBuilder ab(ctx);
      ab.setInsertionPointToStart(allocBlock);
      auto one64 = ab.create<LLVM::ConstantOp>(loc, ab.getI64Type(),
                                                ab.getI64IntegerAttr(1));
      auto privAlloc = ab.create<LLVM::AllocaOp>(loc, ptrTy, arrayTy, one64);
      ab.create<mlir::omp::YieldOp>(loc, ValueRange{privAlloc});
    }

    // --- Array reduction: init region (LLVM dialect) ---
    // Two block args: mold and alloc result.
    {
      Region &init = decl.getInitializerRegion();
      Block *initBlock = new Block();
      init.push_back(initBlock);
      initBlock->addArgument(ptrTy, loc); // mold
      initBlock->addArgument(ptrTy, loc); // alloc result
      OpBuilder ib(ctx);
      ib.setInsertionPointToStart(initBlock);
      Value privCopy = initBlock->getArgument(1);
      Value identity = createIdentityValue(ib, loc, kind, scalarTy, isSigned);
      auto zero32 = ib.create<LLVM::ConstantOp>(loc, ib.getI32Type(),
                                                  ib.getI32IntegerAttr(0));
      for (int64_t i = 0; i < numElems; i++) {
        auto idx = ib.create<LLVM::ConstantOp>(loc, ib.getI32Type(),
                                                ib.getI32IntegerAttr(i));
        auto elemPtr = ib.create<LLVM::GEPOp>(
            loc, ptrTy, arrayTy, privCopy,
            SmallVector<LLVM::GEPArg>{LLVM::GEPArg(zero32),
                                       LLVM::GEPArg(idx)});
        ib.create<LLVM::StoreOp>(loc, identity, elemPtr);
      }
      ib.create<mlir::omp::YieldOp>(loc, ValueRange{privCopy});
    }

    // --- Array reduction: combiner region (LLVM dialect) ---
    {
      Region &comb = decl.getReductionRegion();
      Block *combBlock = new Block();
      comb.push_back(combBlock);
      combBlock->addArgument(ptrTy, loc);
      combBlock->addArgument(ptrTy, loc);
      OpBuilder cb(ctx);
      cb.setInsertionPointToStart(combBlock);
      Value lhsPtr = combBlock->getArgument(0);
      Value rhsPtr = combBlock->getArgument(1);
      auto cZero32 = cb.create<LLVM::ConstantOp>(loc, cb.getI32Type(),
                                                   cb.getI32IntegerAttr(0));
      for (int64_t i = 0; i < numElems; i++) {
        auto idx = cb.create<LLVM::ConstantOp>(loc, cb.getI32Type(),
                                                cb.getI32IntegerAttr(i));
        SmallVector<LLVM::GEPArg> gepArgs = {
            LLVM::GEPArg(cZero32), LLVM::GEPArg(idx)};
        auto lhsElemPtr = cb.create<LLVM::GEPOp>(loc, ptrTy, arrayTy,
                                                   lhsPtr, gepArgs);
        auto rhsElemPtr = cb.create<LLVM::GEPOp>(loc, ptrTy, arrayTy,
                                                   rhsPtr, gepArgs);
        auto lhsElem = cb.create<LLVM::LoadOp>(loc, scalarTy, lhsElemPtr);
        auto rhsElem = cb.create<LLVM::LoadOp>(loc, scalarTy, rhsElemPtr);
        auto combined = createCombinerOp(cb, loc, kind, lhsElem, rhsElem,
                                         scalarTy, isSigned);
        cb.create<LLVM::StoreOp>(loc, combined, lhsElemPtr);
      }
      cb.create<mlir::omp::YieldOp>(loc, ValueRange{lhsPtr});
    }

    // --- Array reduction: cleanup region ---
    // No-op for stack allocation (llvm.alloca is auto-freed).
    // If we used malloc, we'd need free here.
  } else {
    // --- Scalar reduction: init region ---
    Region &init = decl.getInitializerRegion();
    Block *initBlock = new Block();
    init.push_back(initBlock);
    initBlock->addArgument(elementTy, loc);
    OpBuilder ib(ctx);
    ib.setInsertionPointToStart(initBlock);
    Value identity = createIdentityValue(ib, loc, kind, scalarTy, isSigned);
    ib.create<mlir::omp::YieldOp>(loc, identity);

    // --- Scalar reduction: combiner region ---
    Region &comb = decl.getReductionRegion();
    Block *combBlock = new Block();
    comb.push_back(combBlock);
    combBlock->addArgument(elementTy, loc);
    combBlock->addArgument(elementTy, loc);
    OpBuilder cb(ctx);
    cb.setInsertionPointToStart(combBlock);
    Value lhs = combBlock->getArgument(0);
    Value rhs = combBlock->getArgument(1);
    Value result = createCombinerOp(cb, loc, kind, lhs, rhs, elementTy,
                                    isSigned);
    cb.create<mlir::omp::YieldOp>(loc, result);
  }

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
      loc, ValueRange{lb}, builder.getSymbolIdentityMap(), ValueRange{ub},
      builder.getSymbolIdentityMap(), descr.getStep(),
      /*iterArgs=*/ValueRange{});

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
            LLVM::LLVMPointerType::get(mt.getContext(),
                                       mt.getMemorySpaceAsInt()),
            cond);
      }
      if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
        auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
        cond = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cast<mlir::IntegerType>(cond.getType());
      if (ty.getWidth() != 1) {
        cond = builder.create<arith::CmpIOp>(
            loc, CmpIPredicate::ne, cond,
            builder.create<ConstantIntOp>(loc, ty, 0));
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
    auto ty = cast<mlir::IntegerType>(cond.getType());
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, ty, 0));
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

  // Check for nowait clause.
  bool nowait = false;
  for (auto *cl : par->clauses())
    if (cl->getClauseKind() == llvm::omp::OMPC_nowait)
      nowait = true;

  omp::SingleOperands singleClauses;
  if (nowait)
    singleClauses.nowait = builder.getUnitAttr();
  auto singleOp = builder.create<omp::SingleOp>(loc, singleClauses);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  singleOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&singleOp.getRegion().front());

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


  SmallVector<Attribute, 4> dependKindAttrs;
  SmallVector<Value, 4> dependVars;

  // Handle other clauses like allocate_vars, allocators_vars if present
  SmallVector<Value, 4> allocateVars;
  SmallVector<Value, 4> allocatorsVars;

  // Iterate over clauses in the OMPTaskDirective
  for (auto *f : task->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_if: {
      auto *ifClause = cast<OMPIfClause>(f);
      ifExprVal = Visit(ifClause->getCondition()).getValue(loc, builder);
      // OMP spec: condition is boolean; C gives i32. Truncate to i1.
      if (ifExprVal && !ifExprVal.getType().isInteger(1)) {
        auto i1Ty = builder.getIntegerType(1);
        auto zero = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(ifExprVal.getType(), 0));
        ifExprVal = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ifExprVal, zero);
      }
    } break;
    case llvm::omp::OMPC_final: {
      auto *finalClause = cast<OMPFinalClause>(f);
      finalExprVal = Visit(finalClause->getCondition()).getValue(loc, builder);
      if (finalExprVal && !finalExprVal.getType().isInteger(1)) {
        auto zero = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(finalExprVal.getType(), 0));
        finalExprVal = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, finalExprVal, zero);
      }
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

      for (auto *depExpr : depClause->varlist()) {
        if (auto *arraySection =
                dyn_cast<clang::ArraySectionExpr>(depExpr)) {
          // Collect all dimensions from nested array sections
          SmallVector<clang::ArraySectionExpr *> sections;
          clang::Expr *currentExpr = depExpr;

          // Traverse nested array sections to collect all dimensions
          while (auto *ase =
                     dyn_cast<clang::ArraySectionExpr>(currentExpr)) {
            sections.push_back(ase);
            currentExpr = ase->getBase();
          }

          // Get the base array (innermost expression)
          auto baseVC = Visit(currentExpr);
          mlir::Value base = baseVC.getValue(loc, builder);
          auto memrefType = cast<mlir::MemRefType>(base.getType());
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
              memrefType = cast<mlir::MemRefType>(base.getType());
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
          // Handle regular variable dependency.
          // Depend vars need the ADDRESS (not value) for runtime tracking.
          auto vc = Visit(depExpr);
          mlir::Value varVal = vc.val;

          // If the value is a reference (memref/ptr), use it directly.
          // If it's a scalar, create a temporary alloca.
          if (!vc.isReference) {
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
        if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                                 name->getType())))) {
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

  // Create the omp.task operation using TaskOperands
  omp::TaskOperands taskClauses;
  if (ifExprVal)
    taskClauses.ifExpr = ifExprVal;
  if (finalExprVal)
    taskClauses.final = finalExprVal;
  if (untiedAttr)
    taskClauses.untied = untiedAttr;
  if (mergeableAttr)
    taskClauses.mergeable = mergeableAttr;
  if (priorityVal)
    taskClauses.priority = priorityVal;
  if (dependsAttr) {
    for (auto attr : dependsAttr)
      taskClauses.dependKinds.push_back(attr);
  }
  taskClauses.dependVars.assign(dependVars.begin(), dependVars.end());
  taskClauses.allocateVars.assign(allocateVars.begin(), allocateVars.end());
  taskClauses.allocatorVars.assign(allocatorsVars.begin(),
                                   allocatorsVars.end());
  auto taskOp = builder.create<omp::TaskOp>(loc, taskClauses);

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
  DenseMap<VarDecl *, ReductionKind> reductionKindPerVar;
  DenseMap<VarDecl *, bool> reductionSignedPerVar;

  SmallVector<Value, 4> allocateVars;
  SmallVector<Value, 4> allocatorsVars;

  for (auto *f : taskloop->clauses()) {
    switch (f->getClauseKind()) {
    case llvm::omp::OMPC_if: {
      auto *clause = cast<OMPIfClause>(f);
      ifExprVal = Visit(clause->getCondition()).getValue(loc, builder);
      if (ifExprVal && !ifExprVal.getType().isInteger(1)) {
        auto zero = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(ifExprVal.getType(), 0));
        ifExprVal = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, ifExprVal, zero);
      }
    } break;
    case llvm::omp::OMPC_final: {
      auto *clause = cast<OMPFinalClause>(f);
      finalExprVal = Visit(clause->getCondition()).getValue(loc, builder);
      if (finalExprVal && !finalExprVal.getType().isInteger(1)) {
        auto zero = builder.create<arith::ConstantOp>(
            loc, builder.getIntegerAttr(finalExprVal.getType(), 0));
        finalExprVal = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, finalExprVal, zero);
      }
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
      ReductionKind rkind = getReductionKind(rc);
      for (auto *expr : rc->varlist()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl())) {
            reductionDecls.push_back(vd);
            reductionKindPerVar[vd] = rkind;
            reductionSignedPerVar[vd] =
                !vd->getType()->isUnsignedIntegerType();
          }
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
      if (auto mt = dyn_cast<mlir::MemRefType>(sharedAddr.getType())) {
        if (mt.hasStaticShape() && mt.getNumElements() > 1)
          elemTy = mt;
        else
          elemTy = mt.getElementType();
      }

      ReductionKind varKind = reductionKindPerVar.count(vd)
          ? reductionKindPerVar[vd] : ReductionKind::Add;
      bool varSigned = reductionSignedPerVar.count(vd)
          ? reductionSignedPerVar[vd] : true;
      auto decl = getOrCreateReductionDecl(builder, function.getOperation(),
                                           elemTy, varKind, varSigned);
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

      reductionAccumulators.push_back(sharedAddr);
      reductionAccumulatorForVar[vd] = sharedAddr;
      reductionOrder.push_back(vd);
    }
  }

  omp::TaskloopOperands clauses;
  clauses.allocateVars.assign(allocateVars.begin(), allocateVars.end());
  clauses.allocatorVars.assign(allocatorsVars.begin(), allocatorsVars.end());
  if (finalExprVal)
    clauses.final = finalExprVal;
  if (grainSizeValue)
    clauses.grainsize = grainSizeValue;
  if (ifExprVal)
    clauses.ifExpr = ifExprVal;

  if (mergeableFlag)
    clauses.mergeable = builder.getUnitAttr();
  if (nogroupFlag)
    clauses.nogroup = builder.getUnitAttr();
  if (numTasksValue)
    clauses.numTasks = numTasksValue;
  if (priorityVal)
    clauses.priority = priorityVal;
  clauses.reductionVars.assign(reductionAccumulators.begin(),
                               reductionAccumulators.end());
  for (auto sym : reductionDeclSymbols)
    clauses.reductionSyms.push_back(sym);
  for (auto acc : reductionAccumulators) {
    bool isByRef = false;
    if (auto mt = dyn_cast<MemRefType>(acc.getType())) {
      isByRef = mt.hasStaticShape() && mt.getNumElements() > 1;
    } else if (isa<LLVM::LLVMPointerType>(acc.getType())) {
      isByRef = true;
    }
    clauses.reductionByref.push_back(isByRef);
  }
  if (untiedFlag)
    clauses.untied = builder.getUnitAttr();

  // If reduction is present, fall back to sequential scf.for
  // because LLVM IR translation does not support taskloop+reduction.
  if (!reductionAccumulators.empty()) {
    assert(lowerBounds.size() == 1 && "taskloop reduction only supports 1D");
    auto forOp = builder.create<scf::ForOp>(loc, lowerBounds[0],
                                             upperBounds[0], steps[0]);
    builder.setInsertionPointToStart(forOp.getBody());
    // Map IV.
    auto *counter = taskloop->counters()[0];
    auto *counterExpr = cast<DeclRefExpr>(counter);
    VarDecl *indVar = cast<VarDecl>(counterExpr->getDecl());
    bool isArr = false;
    auto indTy = Glob.getMLIRType(indVar->getType(), &isArr);
    auto castIdx = builder.create<IndexCastOp>(loc, indTy,
                                                forOp.getInductionVar());
    params.erase(indVar);
    auto alloc = createAllocOp(indTy, indVar, 0, isArr, false);
    params[indVar] = ValueCategory(alloc, true);
    params[indVar].store(loc, builder, castIdx);
    // Map reduction vars to their accumulators directly.
    for (auto &pr : prevReduction)
      params[pr.first] = pr.second;
    Visit(taskloop->getBody());
    builder.setInsertionPointAfter(forOp);
    for (auto &pr : prevReduction)
      params[pr.first] = pr.second;
    return nullptr;
  }

  auto taskloopOp = builder.create<omp::TaskloopOp>(loc, clauses);

  auto oldPoint = builder.getInsertionPoint();
  auto *oldBlock = builder.getInsertionBlock();

  // LLVM 23: omp.taskloop is a wrapper (NoTerminator, SingleBlock);
  // the actual loop goes inside omp.loop_nest.
  taskloopOp.getRegion().push_back(new Block());
  // Add block args for reduction vars (BlockArgOpenMPOpInterface).
  auto &taskloopBlock = taskloopOp.getRegion().front();
  for (auto acc : reductionAccumulators)
    taskloopBlock.addArgument(acc.getType(), loc);
  builder.setInsertionPointToStart(&taskloopBlock);

  omp::LoopNestOperands taskLoopNestClauses;
  taskLoopNestClauses.loopLowerBounds.assign(lowerBounds.begin(),
                                             lowerBounds.end());
  taskLoopNestClauses.loopUpperBounds.assign(upperBounds.begin(),
                                             upperBounds.end());
  taskLoopNestClauses.loopSteps.assign(steps.begin(), steps.end());
  auto taskLoopNest =
      builder.create<omp::LoopNestOp>(loc, taskLoopNestClauses);
  // Set collapse attribute for multi-level collapsed loops.
  unsigned taskCollapseVal = taskloop->getLoopsNumber();
  if (taskCollapseVal > 1)
    taskLoopNest.setCollapseNumLoops(
        std::optional<uint64_t>(taskCollapseVal));
  // LoopNestOp::build creates an empty region; add a block with IV args.
  auto &taskLoopNestBlock = taskLoopNest.getRegion().emplaceBlock();
  for (auto lb : lowerBounds)
    taskLoopNestBlock.addArgument(lb.getType(), loc);
  auto loopArgs = taskLoopNestBlock.getArguments();

  builder.setInsertionPointToStart(&taskLoopNestBlock);
  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::YieldOp>(loc, ValueRange());
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
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(indVar->getType()))))
      llvmABI = true;

    auto castIdx = builder.create<IndexCastOp>(loc, indTy, std::get<0>(it));
    auto alloc = createAllocOp(indTy, indVar, /*memspace*/ 0, isArray, llvmABI);
    params[indVar] = ValueCategory(alloc, /*isRef*/ true);
    params[indVar].store(loc, builder, castIdx);
  }

  DenseMap<VarDecl *, ValueCategory> prevMappedReduction;
  // Map reduction variables to taskloop block args (thread-level accumulators).
  // Don't create per-iteration temps — use block args directly.
  {
    auto taskBlockArgs = taskloopOp.getRegion().front().getArguments();
    unsigned redIdx = 0;
    for (auto *name : reductionOrder) {
      if (params.count(name))
        prevMappedReduction[name] = params[name];
      if (redIdx < taskBlockArgs.size()) {
        params[name] = ValueCategory(taskBlockArgs[redIdx], /*isRef*/ true);
        redIdx++;
      }
    }
  }

  // Reset keepRunning at the start of each OMP iteration.
  if (!loops.empty() && loops.back().keepRunning) {
    auto vtrue =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), true, 1);
    builder.create<mlir::memref::StoreOp>(loc, vtrue,
                                           loops.back().keepRunning);
  }

  Visit(taskloop->getBody());

  // Body already accumulates directly into block args.
  // Restore original param mappings.
  if (!prevReduction.empty()) {
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
  builder.create<omp::TaskwaitOp>(loc, omp::TaskwaitOperands{});
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPMasterDirective(clang::OMPMasterDirective *master) {
  auto loc = getMLIRLocation(master->getBeginLoc());
  IfScope scope(*this);

  auto masterOp = builder.create<omp::MasterOp>(loc);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  masterOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&masterOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  {
    auto *assoc = master->getAssociatedStmt();
    if (auto *cs = dyn_cast<CapturedStmt>(assoc))
      Visit(cs->getCapturedDecl()->getBody());
    else
      Visit(assoc);
  }

  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPBarrierDirective(clang::OMPBarrierDirective *barrier) {
  auto loc = getMLIRLocation(barrier->getBeginLoc());
  builder.create<omp::BarrierOp>(loc);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPCriticalDirective(clang::OMPCriticalDirective *dir) {
  auto loc = getMLIRLocation(dir->getBeginLoc());
  IfScope scope(*this);

  // Get optional critical section name.
  auto nameInfo = dir->getDirectiveName();
  FlatSymbolRefAttr nameAttr;
  if (nameInfo.getName().getAsString().size() > 0) {
    auto nameStr = nameInfo.getName().getAsString();
    // Ensure a CriticalDeclareOp exists at module scope for named critical.
    auto module = function->getParentOfType<ModuleOp>();
    if (!module.lookupSymbol<omp::CriticalDeclareOp>(nameStr)) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      omp::CriticalDeclareOperands declClauses;
      declClauses.symName = builder.getStringAttr(nameStr);
      builder.create<omp::CriticalDeclareOp>(loc, declClauses);
    }
    nameAttr = FlatSymbolRefAttr::get(builder.getContext(), nameStr);
  }

  auto critOp = builder.create<omp::CriticalOp>(loc, nameAttr);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  critOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&critOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  {
    auto *assoc = dir->getAssociatedStmt();
    if (auto *cs = dyn_cast<CapturedStmt>(assoc))
      Visit(cs->getCapturedDecl()->getBody());
    else
      Visit(assoc);
  }

  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

/// Create the binary operation for an atomic update from the Clang update
/// expression. Returns null if the opcode is unsupported.
static mlir::Value createAtomicBinOp(mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::Value lhs, mlir::Value rhs,
                                     clang::BinaryOperatorKind opcode) {
  bool isFloat = isa<FloatType>(lhs.getType());
  switch (opcode) {
  case clang::BO_Add: case clang::BO_AddAssign:
    return isFloat ? (Value)b.create<arith::AddFOp>(loc, lhs, rhs)
                   : (Value)b.create<arith::AddIOp>(loc, lhs, rhs);
  case clang::BO_Sub: case clang::BO_SubAssign:
    return isFloat ? (Value)b.create<arith::SubFOp>(loc, lhs, rhs)
                   : (Value)b.create<arith::SubIOp>(loc, lhs, rhs);
  case clang::BO_Mul: case clang::BO_MulAssign:
    return isFloat ? (Value)b.create<arith::MulFOp>(loc, lhs, rhs)
                   : (Value)b.create<arith::MulIOp>(loc, lhs, rhs);
  case clang::BO_Div: case clang::BO_DivAssign:
    return isFloat ? (Value)b.create<arith::DivFOp>(loc, lhs, rhs)
                   : (Value)b.create<arith::DivSIOp>(loc, lhs, rhs);
  case clang::BO_And: case clang::BO_AndAssign:
    return b.create<arith::AndIOp>(loc, lhs, rhs);
  case clang::BO_Or: case clang::BO_OrAssign:
    return b.create<arith::OrIOp>(loc, lhs, rhs);
  case clang::BO_Xor: case clang::BO_XorAssign:
    return b.create<arith::XOrIOp>(loc, lhs, rhs);
  case clang::BO_Shl: case clang::BO_ShlAssign:
    return b.create<arith::ShLIOp>(loc, lhs, rhs);
  case clang::BO_Shr: case clang::BO_ShrAssign:
    return b.create<arith::ShRSIOp>(loc, lhs, rhs);
  default:
    return nullptr;
  }
}

/// Extract the binary opcode from an atomic update expression.
static clang::BinaryOperatorKind getAtomicOpcode(clang::Expr *updateExpr) {
  // CompoundAssignOperator is a subclass of BinaryOperator, so this
  // covers both regular binary ops and compound assignments.
  if (auto *bo = dyn_cast<clang::BinaryOperator>(updateExpr))
    return bo->getOpcode();
  return clang::BO_Add;
}

ValueCategory
MLIRScanner::VisitOMPAtomicDirective(clang::OMPAtomicDirective *dir) {
  auto loc = getMLIRLocation(dir->getBeginLoc());

  // Determine atomic kind from clauses (default = update).
  bool isRead = false, isWrite = false, isCapture = false;
  for (auto *cl : dir->clauses()) {
    switch (cl->getClauseKind()) {
    case llvm::omp::OMPC_read:    isRead = true; break;
    case llvm::omp::OMPC_write:   isWrite = true; break;
    case llvm::omp::OMPC_capture: isCapture = true; break;
    default: break;
    }
  }

  auto xExpr = dir->getX();
  auto xVC = Visit(xExpr);
  mlir::Value xAddr = xVC.val;

  // Extract the binary opcode from the update expression.
  auto opcode = dir->getUpdateExpr()
                    ? getAtomicOpcode(dir->getUpdateExpr())
                    : clang::BO_Add;

  if (isRead) {
    // atomic read: v = x
    auto vExpr = dir->getV();
    auto vVC = Visit(vExpr);
    mlir::Value vAddr = vVC.val;
    auto elemTy = cast<MemRefType>(xAddr.getType()).getElementType();
    omp::AtomicReadOp::create(builder, loc, xAddr, vAddr,
          mlir::TypeAttr::get(elemTy), mlir::IntegerAttr{},
          omp::ClauseMemoryOrderKindAttr{});
  } else if (isWrite) {
    // atomic write: x = expr
    auto eExpr = dir->getExpr();
    auto eVal = Visit(eExpr).getValue(loc, builder);
    omp::AtomicWriteOp::create(builder, loc, xAddr, eVal,
          mlir::IntegerAttr{}, omp::ClauseMemoryOrderKindAttr{});
  } else if (isCapture) {
    // atomic capture: combine read + update or read + write
    auto captureOp = omp::AtomicCaptureOp::create(builder, loc,
          mlir::IntegerAttr{}, omp::ClauseMemoryOrderKindAttr{});
    captureOp.getRegion().push_back(new Block());
    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&captureOp.getRegion().front());

    auto vExpr = dir->getV();
    auto vVC = Visit(vExpr);
    mlir::Value vAddr = vVC.val;
    auto elemTy = cast<MemRefType>(xAddr.getType()).getElementType();

    if (dir->isPostfixUpdate()) {
      // v = x; x = x op expr  (read then update)
      omp::AtomicReadOp::create(builder, loc, xAddr, vAddr,
          mlir::TypeAttr::get(elemTy), mlir::IntegerAttr{},
          omp::ClauseMemoryOrderKindAttr{});
      auto updateOp = omp::AtomicUpdateOp::create(builder, loc, xAddr,
          /*atomic_control=*/omp::AtomicControlAttr{},
          /*hint=*/mlir::IntegerAttr{},
          /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});
      auto &updateBlock = updateOp.getRegion().emplaceBlock();
      updateBlock.addArgument(elemTy, loc);
      builder.setInsertionPointToStart(&updateBlock);
      auto curVal = updateBlock.getArgument(0);
      auto eVal = Visit(dir->getExpr()).getValue(loc, builder);
      auto newVal = dir->isXLHSInRHSPart()
          ? createAtomicBinOp(builder, loc, curVal, eVal, opcode)
          : createAtomicBinOp(builder, loc, eVal, curVal, opcode);
      builder.create<omp::YieldOp>(loc, ValueRange{newVal});
    } else {
      // x = x op expr; v = x  (update then read)
      auto updateOp = omp::AtomicUpdateOp::create(builder, loc, xAddr,
          /*atomic_control=*/omp::AtomicControlAttr{},
          /*hint=*/mlir::IntegerAttr{},
          /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});
      auto &updateBlock = updateOp.getRegion().emplaceBlock();
      updateBlock.addArgument(elemTy, loc);
      builder.setInsertionPointToStart(&updateBlock);
      auto curVal = updateBlock.getArgument(0);
      auto eVal = Visit(dir->getExpr()).getValue(loc, builder);
      auto newVal = dir->isXLHSInRHSPart()
          ? createAtomicBinOp(builder, loc, curVal, eVal, opcode)
          : createAtomicBinOp(builder, loc, eVal, curVal, opcode);
      builder.create<omp::YieldOp>(loc, ValueRange{newVal});
      builder.setInsertionPointAfter(updateOp);
      omp::AtomicReadOp::create(builder, loc, xAddr, vAddr,
          mlir::TypeAttr::get(elemTy), mlir::IntegerAttr{},
          omp::ClauseMemoryOrderKindAttr{});
    }
    builder.setInsertionPoint(oldblock, oldpoint);
  } else {
    // atomic update (default): x = x op expr
    auto updateOp = omp::AtomicUpdateOp::create(builder, loc, xAddr,
          /*atomic_control=*/omp::AtomicControlAttr{},
          /*hint=*/mlir::IntegerAttr{},
          /*memory_order=*/omp::ClauseMemoryOrderKindAttr{});
    auto elemTy = cast<MemRefType>(xAddr.getType()).getElementType();
    auto &updateBlock = updateOp.getRegion().emplaceBlock();
    updateBlock.addArgument(elemTy, loc);

    auto oldpoint = builder.getInsertionPoint();
    auto *oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&updateBlock);

    auto curVal = updateBlock.getArgument(0);
    auto eVal = Visit(dir->getExpr()).getValue(loc, builder);
    auto newVal = dir->isXLHSInRHSPart()
        ? createAtomicBinOp(builder, loc, curVal, eVal, opcode)
        : createAtomicBinOp(builder, loc, eVal, curVal, opcode);
    builder.create<omp::YieldOp>(loc, ValueRange{newVal});

    builder.setInsertionPoint(oldblock, oldpoint);
  }

  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPOrderedDirective(clang::OMPOrderedDirective *ordered) {
  auto loc = getMLIRLocation(ordered->getBeginLoc());
  IfScope scope(*this);

  if (!ordered->getAssociatedStmt())
    return nullptr;

  auto orderedOp = builder.create<omp::OrderedRegionOp>(loc);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  orderedOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&orderedOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  {
    auto *assoc = ordered->getAssociatedStmt();
    if (auto *cs = dyn_cast<CapturedStmt>(assoc))
      Visit(cs->getCapturedDecl()->getBody());
    else
      Visit(assoc);
  }

  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitOMPTaskgroupDirective(clang::OMPTaskgroupDirective *tg) {
  auto loc = getMLIRLocation(tg->getBeginLoc());
  IfScope scope(*this);

  auto tgOp = builder.create<omp::TaskgroupOp>(loc, omp::TaskgroupOperands{});

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  tgOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&tgOp.getRegion().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.getRegion().push_back(new Block());
  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  {
    auto *assoc = tg->getAssociatedStmt();
    if (auto *cs = dyn_cast<CapturedStmt>(assoc))
      Visit(cs->getCapturedDecl()->getBody());
    else
      Visit(assoc);
  }

  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPParallelSectionsDirective(
    clang::OMPParallelSectionsDirective *par) {
  auto loc = getMLIRLocation(par->getBeginLoc());
  IfScope scope(*this);

  // Create omp.parallel wrapping omp.sections with omp.section children.
  // Parse num_threads clause.
  omp::ParallelOperands parallelClauses;
  for (auto *cl : par->clauses()) {
    if (cl->getClauseKind() == llvm::omp::OMPC_num_threads) {
      auto *ntc = cast<OMPNumThreadsClause>(cl);
      parallelClauses.numThreadsVars.push_back(
          Visit(ntc->getNumThreads()).getValue(loc, builder));
    }
  }
  auto parallelOp = builder.create<omp::ParallelOp>(loc, parallelClauses);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  parallelOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&parallelOp.getRegion().front());

  // Create omp.sections inside the parallel region.
  omp::SectionsOperands sectionsClauses;
  auto sectionsOp = builder.create<omp::SectionsOp>(loc, sectionsClauses);
  builder.create<omp::TerminatorOp>(loc);

  sectionsOp.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&sectionsOp.getRegion().front());

  // Walk the body to find OMPSectionDirective children.
  Stmt *body;
  if (auto *cs = dyn_cast<CapturedStmt>(par->getAssociatedStmt()))
    body = cs->getCapturedDecl()->getBody();
  else
    body = par->getAssociatedStmt();
  for (auto *child : body->children()) {
    if (isa<OMPSectionDirective>(child)) {
      auto sectionOp = builder.create<omp::SectionOp>(loc);
      sectionOp.getRegion().push_back(new Block());
      auto sectionInsert = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&sectionOp.getRegion().front());

      auto executeRegion =
          builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
      executeRegion.getRegion().push_back(new Block());
      builder.create<omp::TerminatorOp>(loc);
      builder.setInsertionPointToStart(&executeRegion.getRegion().back());

      {
        auto *assoc = cast<OMPSectionDirective>(child)->getAssociatedStmt();
        if (auto *cs = dyn_cast<CapturedStmt>(assoc))
          Visit(cs->getCapturedDecl()->getBody());
        else
          Visit(assoc);
      }

      builder.create<scf::YieldOp>(loc);
      builder.restoreInsertionPoint(sectionInsert);
    }
  }

  builder.create<omp::TerminatorOp>(loc);
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPForDirective(clang::OMPForDirective *fors) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(fors->getBeginLoc());

  // Collect reduction variables with per-variable kind and signedness.
  SmallVector<VarDecl *, 4> reductionVars;
  DenseMap<VarDecl *, ReductionKind> reductionKindPerVar;
  DenseMap<VarDecl *, bool> reductionSignedPerVar;
  SmallVector<VarDecl *, 4> lastprivateVars;
  DenseMap<VarDecl *, ValueCategory> lastprivateOriginal;
  // Detect ordered, nowait, and schedule clauses.
  bool hasOrdered = false;
  bool hasNowait = false;
  mlir::omp::ClauseScheduleKindAttr scheduleValAttr = nullptr;
  mlir::Value scheduleChunkVar = nullptr;
  for (auto *cl : fors->clauses()) {
    if (cl->getClauseKind() == llvm::omp::OMPC_lastprivate) {
      for (auto *expr : cl->children()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl()))
            lastprivateVars.push_back(vd);
      }
    }
    if (cl->getClauseKind() == llvm::omp::OMPC_reduction) {
      auto *rc = cast<OMPReductionClause>(cl);
      ReductionKind rkind = getReductionKind(rc);
      for (auto *expr : rc->varlist()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr)) {
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl())) {
            reductionVars.push_back(vd);
            reductionKindPerVar[vd] = rkind;
            reductionSignedPerVar[vd] =
                !vd->getType()->isUnsignedIntegerType();
          }
        }
      }
    }
    if (cl->getClauseKind() == llvm::omp::OMPC_ordered)
      hasOrdered = true;
    if (cl->getClauseKind() == llvm::omp::OMPC_nowait)
      hasNowait = true;
    if (cl->getClauseKind() == llvm::omp::OMPC_schedule) {
      auto *sc = cast<OMPScheduleClause>(cl);
      auto kind = sc->getScheduleKind();
      mlir::omp::ClauseScheduleKind ompKind;
      switch (kind) {
      case OMPC_SCHEDULE_static:
        ompKind = mlir::omp::ClauseScheduleKind::Static; break;
      case OMPC_SCHEDULE_dynamic:
        ompKind = mlir::omp::ClauseScheduleKind::Dynamic; break;
      case OMPC_SCHEDULE_guided:
        ompKind = mlir::omp::ClauseScheduleKind::Guided; break;
      case OMPC_SCHEDULE_auto:
        ompKind = mlir::omp::ClauseScheduleKind::Auto; break;
      case OMPC_SCHEDULE_runtime:
        ompKind = mlir::omp::ClauseScheduleKind::Runtime; break;
      default:
        ompKind = mlir::omp::ClauseScheduleKind::Static; break;
      }
      scheduleValAttr = mlir::omp::ClauseScheduleKindAttr::get(
          builder.getContext(), ompKind);
      if (auto *chunk = sc->getChunkSize()) {
        scheduleChunkVar = Visit(chunk).getValue(loc, builder);
        scheduleChunkVar = builder.create<IndexCastOp>(
            loc, builder.getIndexType(), scheduleChunkVar);
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
    bool negateStep = false;
    if (bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Sub)
      negateStep = true;
    else
      assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    mlir::Value stepVal = builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder));
    if (negateStep) {
      auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      stepVal = builder.create<arith::SubIOp>(loc, zero, stepVal);
    }
    incs.push_back(stepVal);
  }

  // Prepare reduction metadata: keep shared lvalues and build declare ops.
  std::map<VarDecl *, ValueCategory> prevReduction;
  SmallVector<Attribute, 4> reductionDeclSymbols;
  SmallVector<Value, 4> reductionAccumulators; // accumulator addresses
  DenseMap<VarDecl *, mlir::Value> reductionAccumulatorForVar;
  SmallVector<VarDecl *, 4> reductionOrder;
  if (!reductionVars.empty()) {
    for (auto *name : reductionVars) {
      if (params.find(name) == params.end())
        continue;

      // Save shared mapping and keep shared lvalue as accumulator.
      prevReduction[name] = params[name];

      // Determine type of the reduction variable.
      bool isArray = false;
      mlir::Type elemTy = Glob.getMLIRType(name->getType(), &isArray);
      // For array reductions, use the full memref type (not element type).
      mlir::Value sharedAddr = prevReduction[name].val;
      if (auto mt = dyn_cast<mlir::MemRefType>(sharedAddr.getType())) {
        if (mt.hasStaticShape() && mt.getNumElements() > 1)
          elemTy = mt;
        else
          elemTy = mt.getElementType();
      }

      ReductionKind varKind = reductionKindPerVar.count(name)
          ? reductionKindPerVar[name] : ReductionKind::Add;
      bool varSigned = reductionSignedPerVar.count(name)
          ? reductionSignedPerVar[name] : true;
      auto decl = getOrCreateReductionDecl(builder, function.getOperation(),
                                           elemTy, varKind, varSigned);
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

      Value acc = prevReduction[name].val;
      // For array reduction, convert memref to !llvm.ptr for wsloop operand.
      if (auto mt = dyn_cast<MemRefType>(acc.getType()))
        if (mt.hasStaticShape() && mt.getNumElements() > 1)
          acc = builder.create<polygeist::Memref2PointerOp>(
              loc, LLVM::LLVMPointerType::get(builder.getContext()), acc);
      reductionAccumulators.push_back(acc);
      reductionAccumulatorForVar[name] = acc;
      reductionOrder.push_back(name);
    }
  }

  // Array reduction: fall back to sequential scf.for (no parallelism).
  // LLVM IR translation doesn't support memref-typed reduction.

  omp::WsloopOperands wsloopClauses;
  if (!reductionDeclSymbols.empty()) {
    wsloopClauses.reductionVars.assign(reductionAccumulators.begin(),
                                       reductionAccumulators.end());
    for (auto sym : reductionDeclSymbols)
      wsloopClauses.reductionSyms.push_back(sym);
    // LLVM 23: must specify byref for each reduction var.
    // Scalar reductions: byref=false (value-based, runtime does alloca).
    // Array reductions: byref=true (pointer-based, custom alloc).
    for (auto acc : reductionAccumulators) {
      bool isByRef = false;
      if (auto mt = dyn_cast<MemRefType>(acc.getType())) {
        isByRef = mt.hasStaticShape() && mt.getNumElements() > 1;
      } else if (isa<LLVM::LLVMPointerType>(acc.getType())) {
        isByRef = true;
      }
      wsloopClauses.reductionByref.push_back(isByRef);
    }
  }
  if (hasNowait)
    wsloopClauses.nowait = builder.getUnitAttr();
  if (scheduleValAttr)
    wsloopClauses.scheduleKind = scheduleValAttr;
  if (scheduleChunkVar)
    wsloopClauses.scheduleChunk = scheduleChunkVar;
  if (hasOrdered)
    wsloopClauses.ordered = builder.getI64IntegerAttr(0);
  auto affineOp = builder.create<omp::WsloopOp>(loc, wsloopClauses);
  affineOp.getRegion().push_back(new Block());
  // LLVM 23: add block args for reduction vars (BlockArgOpenMPOpInterface).
  auto &wsloopBlock = affineOp.getRegion().front();
  for (auto acc : reductionAccumulators)
    wsloopBlock.addArgument(acc.getType(), loc);

  auto oldpoint = builder.getInsertionPoint();
  auto *oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&wsloopBlock);

  // LLVM 23: omp.wsloop is a wrapper (NoTerminator, SingleBlock);
  // the actual loop goes inside omp.loop_nest.
  omp::LoopNestOperands loopNestClauses;
  loopNestClauses.loopLowerBounds = inits;
  loopNestClauses.loopUpperBounds = finals;
  loopNestClauses.loopSteps = incs;
  auto loopNest = builder.create<omp::LoopNestOp>(loc, loopNestClauses);
  // Set collapse attribute for multi-level collapsed loops.
  unsigned collapseVal = fors->getLoopsNumber();
  if (collapseVal > 1)
    loopNest.setCollapseNumLoops(std::optional<uint64_t>(collapseVal));
  // LoopNestOp::build creates an empty region; add a block with IV args.
  auto &loopNestBlock = loopNest.getRegion().emplaceBlock();
  for (auto lb : inits)
    loopNestBlock.addArgument(lb.getType(), loc);
  auto inds = loopNestBlock.getArguments();

  builder.setInsertionPointToStart(&loopNestBlock);

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  builder.create<omp::YieldOp>(loc, ValueRange());
  executeRegion.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&executeRegion.getRegion().back());

  auto *oldScope = allocationScope;
  allocationScope = &executeRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto zp : zip(inds, fors->counters())) {
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());
    // Cast index IV to the counter's own type (not the flattened IV type)
    // to avoid type mismatch in collapsed loop body arithmetic.
    auto idx = builder.create<IndexCastOp>(
        loc, getMLIRType(name->getType()), std::get<0>(zp));

    if (params.find(name) != params.end()) {
      prevInduction[name] = params[name];
      params.erase(name);
    }

    bool LLVMABI = false;
    bool isArray = false;
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(name->getType()))))
      LLVMABI = true;
    else
      Glob.getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    params[name] = ValueCategory(allocop, true);
    params[name].store(loc, builder, idx);
  }



  // Lastprivate: save original addresses, create private copies inside body.
  for (auto *vd : lastprivateVars) {
    if (params.count(vd))
      lastprivateOriginal[vd] = params[vd];
    // Create thread-private alloca for lastprivate variable.
    bool lpLLVMABI = false;
    bool lpIsArray = false;
    mlir::Type lpTy;
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
            Glob.CGM.getContext().getLValueReferenceType(vd->getType())))) {
      lpLLVMABI = true;
      bool undef;
      lpTy = Glob.getMLIRType(vd->getType(), &undef);
    } else
      lpTy = Glob.getMLIRType(vd->getType(), &lpIsArray);
    params.erase(vd);
    auto lpAlloc = createAllocOp(lpTy, vd, /*memtype*/ 0,
                                 /*isArray*/ lpIsArray, /*LLVMABI*/ lpLLVMABI);
    params[vd] = ValueCategory(lpAlloc, true);
  }

  // Map reduction variables to wsloop block args (thread-level accumulators).
  {
    auto wsBlockArgs = affineOp.getRegion().front().getArguments();
    unsigned redIdx = 0;
    for (auto *name : reductionOrder) {
      if (redIdx < wsBlockArgs.size()) {
        mlir::Value blockArg = wsBlockArgs[redIdx];
        // For array reduction (block arg is !llvm.ptr), cast back to memref
        // so the body code can use memref.load/store.
        if (isa<LLVM::LLVMPointerType>(blockArg.getType())) {
          auto origTy = prevReduction[name].val.getType();
          blockArg = builder.create<polygeist::Pointer2MemrefOp>(
              loc, origTy, blockArg);
        }
        params[name] = ValueCategory(blockArg, /*isRef*/ true);
        redIdx++;
      }
    }
  }

  // Reset keepRunning at the start of each OMP iteration.
  // In OMP parallel for, 'continue' only affects the current iteration.
  // The shared keepRunning flag must be reset per-iteration to prevent
  // one thread's continue from affecting other threads/iterations.
  if (!loops.empty() && loops.back().keepRunning) {
    auto vtrue =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), true, 1);
    builder.create<mlir::memref::StoreOp>(loc, vtrue,
                                           loops.back().keepRunning);
  }

  Visit(fors->getBody());

  // Lastprivate write-back: copy from last iteration to original.
  if (!lastprivateVars.empty() && !inds.empty()) {
    auto iv = inds[0];
    auto step = incs[0];
    // last IV value = ub - step
    auto lastIV = builder.create<arith::SubIOp>(loc, finals[0], step);
    auto isLast = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, iv, lastIV);
    auto ifOp = builder.create<scf::IfOp>(loc, isLast, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    for (auto *vd : lastprivateVars) {
      if (lastprivateOriginal.count(vd) && params.count(vd)) {
        auto origAddr = lastprivateOriginal[vd];
        auto privateVal = params[vd].getValue(loc, builder);
        origAddr.store(loc, builder, privateVal);
      }
    }
    builder.setInsertionPointAfter(ifOp);
  }

  // Restore original mappings for lastprivate variables.
  for (auto &lp : lastprivateOriginal)
    params[lp.first] = lp.second;

  // Reduction body directly accumulates into wsloop block args.
  // No per-iteration temp → block arg copy needed.

  builder.create<scf::YieldOp>(loc, ValueRange());

  allocationScope = oldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);

  // Reset keepRunning flag after OMP parallel for.
  // 'continue' inside the parallel body may have set it to false, but
  // 'continue' only affects the current iteration, not post-loop code.
  if (!loops.empty() && loops.back().keepRunning) {
    auto vtrue =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), true, 1);
    builder.create<mlir::memref::StoreOp>(loc, vtrue,
                                           loops.back().keepRunning);
  }

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
      // Save original mappings. Alloca + init will happen inside parallel body
      // so each thread gets its own private copy.
      for (auto *stmt : f->children()) {
        VarDecl *name = cast<VarDecl>(cast<DeclRefExpr>(stmt)->getDecl());
        prevInduction[name] = params[name];
      }
    }

    break;
    case llvm::omp::OMPC_num_threads: {
      auto *numThreadsClause = cast<OMPNumThreadsClause>(f);
      numThreads =
          Visit(numThreadsClause->getNumThreads()).getValue(loc, builder);
      break;
    }
    case llvm::omp::OMPC_shared:
      // Shared variables are accessible by default in MLIR's OpenMP dialect.
      // No special handling needed - variables from outer scope remain shared.
      break;
    default:
      llvm::errs() << "may not handle omp clause " << (int)f->getClauseKind()
                   << "\n";
    }
  }
  omp::ParallelOperands parallelClauses;
  if (numThreads)
    parallelClauses.numThreadsVars.push_back(numThreads);
  auto affineOp = builder.create<omp::ParallelOp>(loc, parallelClauses);

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

  // Create private/firstprivate copies inside the parallel body
  // so each thread gets its own alloca.
  for (auto *f : par->clauses()) {
    if (f->getClauseKind() != llvm::omp::OMPC_private &&
        f->getClauseKind() != llvm::omp::OMPC_firstprivate)
      continue;
    for (auto *stmt : f->children()) {
      VarDecl *name = cast<VarDecl>(cast<DeclRefExpr>(stmt)->getDecl());

      bool LLVMABI = false;
      bool isArray = false;
      mlir::Type ty;
      if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
              Glob.CGM.getContext().getLValueReferenceType(name->getType())))) {
        LLVMABI = true;
        bool undef;
        ty = Glob.getMLIRType(name->getType(), &undef);
      } else
        ty = Glob.getMLIRType(name->getType(), &isArray);

      params.erase(name);
      auto allocop = createAllocOp(ty, name, /*memtype*/ 0,
                                   /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
      params[name] = ValueCategory(allocop, true);

      // Firstprivate: copy original value into the thread-local alloca.
      if (f->getClauseKind() == llvm::omp::OMPC_firstprivate) {
        if (prevInduction.count(name))
          params[name].store(loc, builder, prevInduction[name], isArray);
      }
    }
  }

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

  // Collect reduction variables with per-variable kind and signedness.
  SmallVector<VarDecl *, 4> reductionVars;
  DenseMap<VarDecl *, ReductionKind> reductionKindPerVar;
  DenseMap<VarDecl *, bool> reductionSignedPerVar;
  SmallVector<VarDecl *, 4> privateVars;
  SmallVector<VarDecl *, 4> firstprivateVars;
  SmallVector<VarDecl *, 4> lastprivateVars;
  DenseMap<VarDecl *, ValueCategory> lastprivateOriginal;
  std::map<VarDecl *, ValueCategory> prevPrivate;
  bool hasOrdered = false;

  // Handle schedule clause
  mlir::omp::ClauseScheduleKindAttr scheduleValAttr = nullptr;
  mlir::Value scheduleChunkVar = nullptr;
  mlir::omp::ScheduleModifierAttr scheduleModifierAttr = nullptr;

  for (auto *cl : fors->clauses()) {
    if (cl->getClauseKind() == llvm::omp::OMPC_private) {
      for (auto *expr : cl->children())
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl())) {
            privateVars.push_back(vd);
            prevPrivate[vd] = params[vd];
          }
    }
    if (cl->getClauseKind() == llvm::omp::OMPC_firstprivate) {
      for (auto *expr : cl->children())
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl())) {
            firstprivateVars.push_back(vd);
            prevPrivate[vd] = params[vd];
          }
    }
    if (cl->getClauseKind() == llvm::omp::OMPC_lastprivate) {
      for (auto *expr : cl->children()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr))
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl()))
            lastprivateVars.push_back(vd);
      }
    }
    if (cl->getClauseKind() == llvm::omp::OMPC_reduction) {
      auto *rc = cast<OMPReductionClause>(cl);
      ReductionKind rkind = getReductionKind(rc);
      for (auto *expr : rc->varlist()) {
        if (auto *dref = dyn_cast<DeclRefExpr>(expr)) {
          if (auto *vd = dyn_cast<VarDecl>(dref->getDecl())) {
            reductionVars.push_back(vd);
            reductionKindPerVar[vd] = rkind;
            reductionSignedPerVar[vd] =
                !vd->getType()->isUnsignedIntegerType();
          }
        }
      }
    } else if (cl->getClauseKind() == llvm::omp::OMPC_ordered) {
      hasOrdered = true;
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
    bool negateStep = false;
    if (bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Sub)
      negateStep = true;
    else
      assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Add);
    f = bo->getRHS();
    while (auto *ce = dyn_cast<clang::CastExpr>(f))
      f = ce->getSubExpr();
    bo = cast<clang::BinaryOperator>(f);
    assert(bo->getOpcode() == clang::BinaryOperator::Opcode::BO_Mul);
    f = bo->getRHS();
    mlir::Value stepVal = builder.create<IndexCastOp>(
        loc, builder.getIndexType(), Visit(f).getValue(loc, builder));
    if (negateStep) {
      auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
      stepVal = builder.create<arith::SubIOp>(loc, zero, stepVal);
    }
    incs.push_back(stepVal);
  }

  SmallVector<mlir::Value> inds;
  /// Create OpenMP parallel + wsloop
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
      if (auto mt = dyn_cast<mlir::MemRefType>(sharedAddr.getType())) {
        if (mt.hasStaticShape() && mt.getNumElements() > 1)
          elemTy = mt;
        else
          elemTy = mt.getElementType();
      }

      ReductionKind varKind = reductionKindPerVar.count(name)
          ? reductionKindPerVar[name] : ReductionKind::Add;
      bool varSigned = reductionSignedPerVar.count(name)
          ? reductionSignedPerVar[name] : true;
      auto decl = getOrCreateReductionDecl(builder, function.getOperation(),
                                           elemTy, varKind, varSigned);
      reductionDeclSymbols.push_back(
          SymbolRefAttr::get(builder.getContext(), decl.getSymName()));

      // For array reduction, convert memref to !llvm.ptr.
      if (auto mt = dyn_cast<MemRefType>(sharedAddr.getType()))
        if (mt.hasStaticShape() && mt.getNumElements() > 1)
          sharedAddr = builder.create<polygeist::Memref2PointerOp>(
              loc, LLVM::LLVMPointerType::get(builder.getContext()),
              sharedAddr);
      reductionAccumulators.push_back(sharedAddr);
      reductionAccumulatorForVar[name] = sharedAddr;
      reductionOrder.push_back(name);
    }
  }



  omp::ParallelOperands parallelClauses2;

  auto parallelOp = builder.create<omp::ParallelOp>(loc, parallelClauses2);

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

  // Create thread-private allocas for private/firstprivate vars.
  // Allocas are inside executeRegion → per-thread.
  for (auto *vd : privateVars) {
    bool isArr = false;
    auto ty = Glob.getMLIRType(vd->getType(), &isArr);
    bool LLVMABI = isa<LLVM::LLVMPointerType>(Glob.getMLIRType(
        Glob.CGM.getContext().getLValueReferenceType(vd->getType())));
    params.erase(vd);
    auto alloc = createAllocOp(ty, vd, 0, isArr, LLVMABI);
    params[vd] = ValueCategory(alloc, true);
  }
  for (auto *vd : firstprivateVars) {
    bool isArr = false;
    auto ty = Glob.getMLIRType(vd->getType(), &isArr);
    bool LLVMABI = isa<LLVM::LLVMPointerType>(Glob.getMLIRType(
        Glob.CGM.getContext().getLValueReferenceType(vd->getType())));
    params.erase(vd);
    auto alloc = createAllocOp(ty, vd, 0, isArr, LLVMABI);
    params[vd] = ValueCategory(alloc, true);
    if (prevPrivate.count(vd))
      params[vd].store(loc, builder, prevPrivate[vd], isArr);
  }
  omp::WsloopOperands wsloopClauses2;
  wsloopClauses2.scheduleKind = scheduleValAttr;
  wsloopClauses2.scheduleChunk = scheduleChunkVar;
  wsloopClauses2.scheduleMod = scheduleModifierAttr;
  if (!reductionDeclSymbols.empty()) {
    wsloopClauses2.reductionVars.assign(reductionAccumulators.begin(),
                                        reductionAccumulators.end());
    for (auto sym : reductionDeclSymbols)
      wsloopClauses2.reductionSyms.push_back(sym);
    for (auto acc : reductionAccumulators) {
      bool isByRef = false;
      if (auto mt = dyn_cast<MemRefType>(acc.getType())) {
        isByRef = mt.hasStaticShape() && mt.getNumElements() > 1;
      } else if (isa<LLVM::LLVMPointerType>(acc.getType())) {
        isByRef = true;
      }
      wsloopClauses2.reductionByref.push_back(isByRef);
    }
  }
  if (hasOrdered)
    wsloopClauses2.ordered = builder.getI64IntegerAttr(0);

  auto wsLoopOp = builder.create<omp::WsloopOp>(loc, wsloopClauses2);

  wsLoopOp.getRegion().push_back(new Block());
  // LLVM 23: add block args for reduction vars (BlockArgOpenMPOpInterface).
  auto &wsLoopBlock = wsLoopOp.getRegion().front();
  for (auto acc : reductionAccumulators)
    wsLoopBlock.addArgument(acc.getType(), loc);

  auto wsLoopOldpoint = builder.getInsertionPoint();
  auto *wsLoopOldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&wsLoopBlock);

  // LLVM 23: omp.wsloop is a wrapper (NoTerminator, SingleBlock);
  // the actual loop goes inside omp.loop_nest.
  omp::LoopNestOperands loopNestClauses2;
  loopNestClauses2.loopLowerBounds = inits;
  loopNestClauses2.loopUpperBounds = finals;
  loopNestClauses2.loopSteps = incs;
  auto wsLoopNest = builder.create<omp::LoopNestOp>(loc, loopNestClauses2);
  // Set collapse attribute for multi-level collapsed loops.
  unsigned collapseVal2 = fors->getLoopsNumber();
  if (collapseVal2 > 1)
    wsLoopNest.setCollapseNumLoops(std::optional<uint64_t>(collapseVal2));
  // LoopNestOp::build creates an empty region; add a block with IV args.
  auto &wsLoopNestBlock = wsLoopNest.getRegion().emplaceBlock();
  for (auto lb : inits)
    wsLoopNestBlock.addArgument(lb.getType(), loc);
  auto wsLoopInds = wsLoopNestBlock.getArguments();
  inds.assign(wsLoopInds.begin(), wsLoopInds.end());

  builder.setInsertionPointToStart(&wsLoopNestBlock);

  auto wsLoopExecuteRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  builder.create<omp::YieldOp>(loc, ValueRange());
  wsLoopExecuteRegion.getRegion().push_back(new Block());
  builder.setInsertionPointToStart(&wsLoopExecuteRegion.getRegion().back());

  auto *wsLoopOldScope = allocationScope;
  allocationScope = &wsLoopExecuteRegion.getRegion().back();

  std::map<VarDecl *, ValueCategory> prevInduction;
  for (auto zp : zip(inds, fors->counters())) {
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());
    // Cast index IV to the counter's own type (not the flattened IV type)
    // to avoid type mismatch in collapsed loop body arithmetic.
    auto idx = builder.create<IndexCastOp>(
        loc, getMLIRType(name->getType()), std::get<0>(zp));

    if (params.find(name) != params.end()) {
      prevInduction[name] = params[name];
      params.erase(name);
    }

    bool LLVMABI = false;
    bool isArray = false;
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(name->getType()))))
      LLVMABI = true;
    else
      Glob.getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    params[name] = ValueCategory(allocop, true);
    params[name].store(loc, builder, idx);
  }

  if (!prevReduction.empty()) {
    for (auto &pr : prevReduction) {
      VarDecl *name = pr.first;

    }
  }

  // Lastprivate: save ORIGINAL (pre-parallel) addresses for write-back.
  for (auto *vd : lastprivateVars) {
    if (prevPrivate.count(vd)) lastprivateOriginal[vd] = prevPrivate[vd]; else if (params.count(vd))
      lastprivateOriginal[vd] = params[vd];
    // Create thread-private alloca for lastprivate variable.
    bool lpLLVMABI = false;
    bool lpIsArray = false;
    mlir::Type lpTy;
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
            Glob.CGM.getContext().getLValueReferenceType(vd->getType())))) {
      lpLLVMABI = true;
      bool undef;
      lpTy = Glob.getMLIRType(vd->getType(), &undef);
    } else
      lpTy = Glob.getMLIRType(vd->getType(), &lpIsArray);
    params.erase(vd);
    auto lpAlloc = createAllocOp(lpTy, vd, /*memtype*/ 0,
                                 /*isArray*/ lpIsArray, /*LLVMABI*/ lpLLVMABI);
    params[vd] = ValueCategory(lpAlloc, true);
  }

  // Map reduction variables to wsloop block args (thread-level accumulators).
  {
    auto wsBlockArgs = wsLoopOp.getRegion().front().getArguments();
    unsigned redIdx = 0;
    for (auto *name : reductionOrder) {
      if (redIdx < wsBlockArgs.size()) {
        mlir::Value blockArg = wsBlockArgs[redIdx];
        if (isa<LLVM::LLVMPointerType>(blockArg.getType())) {
          auto origTy = prevReduction[name].val.getType();
          blockArg = builder.create<polygeist::Pointer2MemrefOp>(
              loc, origTy, blockArg);
        }
        params[name] = ValueCategory(blockArg, /*isRef*/ true);
        redIdx++;
      }
    }
  }

  // Reset keepRunning at the start of each OMP iteration.
  if (!loops.empty() && loops.back().keepRunning) {
    auto vtrue =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), true, 1);
    builder.create<mlir::memref::StoreOp>(loc, vtrue,
                                           loops.back().keepRunning);
  }

  Visit(fors->getBody());

  // Lastprivate write-back: copy from last iteration to original.
  if (!lastprivateVars.empty() && !wsLoopInds.empty()) {
    auto iv = wsLoopInds[0];
    auto step = incs[0];
    auto lastIV = builder.create<arith::SubIOp>(loc, finals[0], step);
    auto isLast = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, iv, lastIV);
    auto ifOp = builder.create<scf::IfOp>(loc, isLast, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    for (auto *vd : lastprivateVars) {
      if (lastprivateOriginal.count(vd) && params.count(vd)) {
        auto origAddr = lastprivateOriginal[vd];
        auto privateVal = params[vd].getValue(loc, builder);
        origAddr.store(loc, builder, privateVal);
      }
    }
    builder.setInsertionPointAfter(ifOp);
  }

  // Restore original mappings for lastprivate variables.
  for (auto &lp : lastprivateOriginal)
    params[lp.first] = lp.second;


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
  for (auto &pp : prevPrivate)
    params[pp.first] = pp.second;

  // Reset keepRunning flag after OMP parallel for.
  if (!loops.empty() && loops.back().keepRunning) {
    auto vtrue =
        builder.create<ConstantIntOp>(builder.getUnknownLoc(), true, 1);
    builder.create<mlir::memref::StoreOp>(loc, vtrue,
                                           loops.back().keepRunning);
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
    auto ty = cast<mlir::IntegerType>(cond.getType());
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, ty, 0));
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
    auto ty = cast<mlir::IntegerType>(cond.getType());
    if (ty.getWidth() != 1) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, ty, 0));
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
        loc, LLVM::LLVMPointerType::get(builder.getContext()), cond);
  }
  if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
    auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  if (!isa<mlir::IntegerType>(cond.getType())) {
    stmt->dump();
    llvm::errs() << " cond: " << cond << " ct: " << cond.getType() << "\n";
  }
  auto prevTy = cast<mlir::IntegerType>(cond.getType());
  if (!prevTy.isInteger(1)) {
    cond = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, cond,
        builder.create<ConstantIntOp>(loc, prevTy, 0));
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
  auto ity = cast<mlir::IntegerType>(cond.getType());
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
  auto loc = getMLIRLocation(stmt->getBeginLoc());
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
  auto loc = getMLIRLocation(stmt->getBeginLoc());
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
    assert(cast<MemRefType>(rv.val.getType()).getElementType() ==
               cast<MemRefType>(op.getType()).getElementType() &&
           "type mismatch");
    assert(cast<MemRefType>(op.getType()).getShape().size() == 2 &&
           "expect 2d memref");
    assert(cast<MemRefType>(rv.val.getType()).getShape().size() == 2 &&
           "expect 2d memref");
    assert(cast<MemRefType>(rv.val.getType()).getShape()[1] ==
           cast<MemRefType>(op.getType()).getShape()[1]);

    for (int i = 0; i < cast<MemRefType>(op.getType()).getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(cast<MemRefType>(rv.val.getType()).getShape().size() == 2);
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

      auto postTy = cast<MemRefType>(returnVal.getType()).getElementType();
      if (auto prevTy = dyn_cast<mlir::IntegerType>(val.getType())) {
        auto ipostTy = cast<mlir::IntegerType>(postTy);
        if (prevTy != ipostTy) {
          val = builder.create<arith::TruncIOp>(loc, ipostTy, val);
        }
      } else if (isa<MemRefType>(val.getType()) &&
                 isa<LLVM::LLVMPointerType>(postTy))
        val = builder.create<polygeist::Memref2PointerOp>(loc, postTy, val);
      else if (isa<LLVM::LLVMPointerType>(val.getType()) &&
               isa<MemRefType>(postTy))
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
