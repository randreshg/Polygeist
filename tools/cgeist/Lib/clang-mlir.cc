//===- clang-mlir.cc - Emit MLIR IRs by walking clang AST--------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-mlir.h"
#include "../ArgumentList.h"
#include "TypeUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "utils.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"
#include "clang/Basic/LangStandard.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"
#include "clang/Options/OptionUtils.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace std;
using namespace clang;
using namespace llvm;
using namespace clang::driver;
using namespace llvm::opt;
using namespace mlir;
using namespace mlir::arith;
using namespace mlirclang;

/// Helper to deduce the pointee type of an LLVM pointer value.
/// With opaque pointers (LLVM 23+), LLVMPointerType no longer carries element
/// type info. This helper tries to recover it from the defining operation.
static mlir::Type deducePointeeType(mlir::Value ptr) {
  if (auto m2p = ptr.getDefiningOp<polygeist::Memref2PointerOp>()) {
    if (auto mt = dyn_cast<MemRefType>(m2p.getSource().getType()))
      return mt.getElementType();
  }
  if (auto alloca = ptr.getDefiningOp<LLVM::AllocaOp>())
    return alloca.getElemType();
  if (auto gep = ptr.getDefiningOp<LLVM::GEPOp>())
    return gep.getElemType();
  if (auto p2m = ptr.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
    if (auto mt = dyn_cast<MemRefType>(p2m.getResult().getType()))
      return mt.getElementType();
  }
  return nullptr;
}

#define DEBUG_TYPE "clang-mlir"

static cl::opt<bool>
    memRefFullRank("memref-fullrank", cl::init(false),
                   cl::desc("Get the full rank of the memref."));

static cl::opt<bool> memRefABI("memref-abi", cl::init(true),
                               cl::desc("Use memrefs when possible"));

cl::opt<std::string> PrefixABI("prefix-abi", cl::init(""),
                               cl::desc("Prefix for emitted symbols"));

cl::opt<bool> CStyleMemRef("c-style-memref", cl::init(true),
                           cl::desc("Use c style memrefs when possible"));

static cl::opt<bool>
    CombinedStructABI("struct-abi", cl::init(true),
                      cl::desc("Use literal LLVM ABI for structs"));

ValueCategory MLIRScanner::createComplexFloat(mlir::Location loc,
                                              mlir::Value real,
                                              mlir::Value imag,
                                              clang::QualType cty) {
  bool ref;
  mlir::Type elty = getMLIRType(cty);
  mlir::Type mt;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  mlir::Value result;
  if (auto MT = dyn_cast<MemRefType>(elty)) {
    mt = elty;
    elty = MT.getElementType();
    ref = true;
    result = abuilder.create<mlir::memref::AllocaOp>(loc, MT);
  } else {
    ref = false;
  }

  if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
    mlir::Value str = builder.create<polygeist::UndefOp>(loc, ST);
    str = builder.create<LLVM::InsertValueOp>(loc, ST, str, real,
                                              builder.getDenseI64ArrayAttr(0));
    str = builder.create<LLVM::InsertValueOp>(loc, ST, str, imag,
                                              builder.getDenseI64ArrayAttr(1));
    if (ref) {
      builder.create<mlir::memref::StoreOp>(loc, str, result,
                                            getConstantIndex(0));
    } else {
      result = str;
    }
  } else {
    builder.create<mlir::memref::StoreOp>(loc, real, result,
                                          getConstantIndex(0));
    builder.create<mlir::memref::StoreOp>(loc, imag, result,
                                          getConstantIndex(1));
  }
  return ValueCategory(result, /*isReference*/ ref);
}

ValueCategory MLIRScanner::getComplexPartRef(mlir::Location loc,
                                             mlir::Value val, int fnum,
                                             mlir::Type *pointeeType) {
  // Track the known pointee type for the LLVM pointer path (opaque pointers).
  mlir::Type knownPointeeType;
  if (auto MT = dyn_cast<MemRefType>(val.getType())) {
    if (isa<LLVM::LLVMStructType>(MT.getElementType()) &&
        MT.getShape().size() == 1) {
      knownPointeeType = MT.getElementType();
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(MT.getContext(),
                                     MT.getMemorySpaceAsInt()),
          val);
    }
  }
  if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);
    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    shape[0] = ShapedType::kDynamic;
    auto mt1 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    mlir::Value si = builder.create<polygeist::SubIndexOp>(loc, mt0, val,
                                                           getConstantIndex(0));
    si = builder.create<polygeist::SubIndexOp>(loc, mt1, si,
                                               getConstantIndex(fnum));
    if (pointeeType)
      *pointeeType = mt.getElementType();
    return ValueCategory(si, /*isReference*/ true);
  } else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
    // With opaque pointers, get the pointee type from the tracked conversion
    // or from the original pointer's defining op.
    mlir::Type srcType = knownPointeeType;
    if (!srcType) {
      // Try to get from Memref2PointerOp source
      if (auto m2p = val.getDefiningOp<polygeist::Memref2PointerOp>()) {
        if (auto srcMT = dyn_cast<MemRefType>(m2p.getSource().getType()))
          srcType = srcMT.getElementType();
      }
    }
    assert(srcType && "cannot determine pointee type for opaque pointer");
    mlir::Type ET;
    if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(srcType)) {
      ET = ST.getBody()[fnum];
    } else {
      ET = cast<mlir::LLVM::LLVMArrayType>(srcType).getElementType();
    }
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, fnum, 32)};
    if (pointeeType)
      *pointeeType = ET;
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(val.getContext(),
                                                   PT.getAddressSpace()),
            srcType, val, vec),
        /*isReference*/ true);
  } else {
    llvm_unreachable("unexpected complex type");
  }
}

/// Get real (fnum = 0) or imaginary (fnum = 1) part of a complex float
mlir::Value MLIRScanner::getComplexPart(mlir::Location loc, mlir::Value complex,
                                        int fnum) {
  if (auto ft = dyn_cast<mlir::FloatType>(complex.getType())) {
    if (fnum == 0)
      return complex;
    else
      return builder.create<ConstantFloatOp>(
          loc, ft, APFloat::getZero(ft.getFloatSemantics()));
  } else if (auto ST =
                 dyn_cast<mlir::LLVM::LLVMStructType>(complex.getType())) {
    return builder.create<LLVM::ExtractValueOp>(loc, complex, fnum);
  }
  mlir::Type elemTy;
  auto ref = getComplexPartRef(loc, complex, fnum, &elemTy).val;
  if (auto mt = dyn_cast<mlir::MemRefType>(ref.getType())) {
    return builder.create<mlir::memref::LoadOp>(loc, ref);
  } else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(ref.getType())) {
    return builder.create<mlir::LLVM::LoadOp>(loc, elemTy, ref);
  } else {
    assert(0);
  }
}

bool isLLVMStructABI(const RecordDecl *RD, llvm::StructType *ST) {
  if (!CombinedStructABI)
    return true;
  if (RD->isUnion())
    return true;
  if (auto CXRD = dyn_cast<CXXRecordDecl>(RD)) {
    if (!CXRD->hasDefinition())
      return true;
    if (CXRD->getNumVBases())
      return true;
    for (auto m : CXRD->methods()) {
      if (m->isVirtualAsWritten() || m->isPureVirtual())
        return true;
    }
  }
  if (ST) {
    if (!ST->isLiteral() && (ST->getName() == "struct._IO_FILE" ||
                             ST->getName() == "class.std::basic_ifstream" ||
                             ST->getName() == "class.std::basic_istream" ||
                             ST->getName() == "class.std::basic_ostream" ||
                             ST->getName() == "class.std::basic_ofstream"))
      return true;
  }
  return false;
}

mlir::Attribute wrapIntegerMemorySpace(unsigned memorySpace, MLIRContext *ctx) {
  if (memorySpace == 0)
    return nullptr;

  return mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), memorySpace);
}

MLIRScanner::MLIRScanner(MLIRASTConsumer &Glob,
                         mlir::OwningOpRef<mlir::ModuleOp> &module,
                         LowerToInfo &LTInfo)
    : Glob(Glob), module(module), builder(module->getContext()),
      ThisCapture(nullptr), LTInfo(LTInfo) {}

void MLIRScanner::init(mlir::func::FuncOp function, const FunctionDecl *fd) {
  this->function = function;
  this->EmittingFunctionDecl = fd;

  if (ShowAST) {
    llvm::errs() << "Emitting fn: " << function.getName() << "\n";
    llvm::errs() << *fd << "\n";
  }

  setEntryAndAllocBlock(function.addEntryBlock());

  unsigned i = 0;
  if (auto CM = dyn_cast<CXXMethodDecl>(fd)) {
    if (CM->getParent()->isLambda()) {
      for (auto C : CM->getParent()->captures()) {
        if (C.capturesVariable()) {
          CaptureKinds[C.getCapturedVar()] = C.getCaptureKind();
        }
      }
      CM->getParent()->getCaptureFields(Captures, ThisCapture);
      if (ThisCapture) {
        llvm::errs() << " thiscapture:\n";
        ThisCapture->dump();
      }
    }

    if (CM->isInstance()) {
      mlir::Value val = function.getArgument(i);
      ThisVal = ValueCategory(val, /*isReference*/ false);
      i++;
    }
  }

  for (auto parm : fd->parameters()) {
    assert(i != function.getNumArguments());
    // function.getArgument(i).setName(name);
    bool isArray = false;
    bool LLVMABI = false;

    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(Glob.CGM.getContext().getPointerType(parm->getType()))))
      LLVMABI = true;

    if (!LLVMABI) {
      Glob.getMLIRType(parm->getType(), &isArray);
    }
    if (!isArray && isa<clang::ReferenceType>(
                        parm->getType()->getUnqualifiedDesugaredType()))
      isArray = true;
    mlir::Value val = function.getArgument(i);
    assert(val);
    if (isArray) {
      params.emplace(parm, ValueCategory(val, /*isReference*/ true));
    } else {
      auto alloc = createAllocOp(val.getType(), parm, /*memspace*/ 0, isArray,
                                 /*LLVMABI*/ LLVMABI);
      ValueCategory(alloc, /*isReference*/ true)
          .store(getMLIRLocation(parm->getBeginLoc()), builder, val);
    }
    i++;
  }

  auto loc = getMLIRLocation(fd->getBeginLoc());
  if (fd->hasAttr<CUDAGlobalAttr>() && Glob.CGM.getLangOpts().CUDA &&
      !Glob.CGM.getLangOpts().CUDAIsDevice) {
    auto deviceStub =
        Glob.GetOrCreateMLIRFunction(fd, /* getDeviceStub */ true);
    builder.create<func::CallOp>(loc, deviceStub, function.getArguments());
    builder.create<func::ReturnOp>(loc);
    return;
  }

  if (auto CC = dyn_cast<CXXConstructorDecl>(fd)) {
    const CXXRecordDecl *ClassDecl = CC->getParent();
    for (auto expr : CC->inits()) {
      if (ShowAST) {
        llvm::errs() << " init: - baseInit:" << (int)expr->isBaseInitializer()
                     << " memberInit:" << (int)expr->isMemberInitializer()
                     << " anyMember:" << (int)expr->isAnyMemberInitializer()
                     << " indirectMember:"
                     << (int)expr->isIndirectMemberInitializer()
                     << " isinClass:" << (int)expr->isInClassMemberInitializer()
                     << " delegating:" << (int)expr->isDelegatingInitializer()
                     << " isPack:" << (int)expr->isPackExpansion() << "\n";
        if (expr->getMember())
          expr->getMember()->dump();
        if (expr->getInit())
          expr->getInit()->dump();
      }
      assert(ThisVal.val);
      FieldDecl *field = expr->getMember();
      if (!field) {
        if (expr->isBaseInitializer()) {
          bool BaseIsVirtual = expr->isBaseVirtual();

          auto BaseType = expr->getBaseClass();

          // Shift and cast down to the base type.
          // TODO: for complete types, this should be possible with a GEP.
          mlir::Value V = ThisVal.val;

          const clang::Type *BaseTypes[] = {BaseType};
          bool BaseVirtual[] = {BaseIsVirtual};

          V = GetAddressOfBaseClass(loc, V, /*derived*/ ClassDecl, BaseTypes,
                                    BaseVirtual);

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0, /*mem*/ V);
          continue;
        }
        if (expr->isDelegatingInitializer()) {

          Expr *init = expr->getInit();
          if (auto clean = dyn_cast<ExprWithCleanups>(init)) {
            llvm::errs() << "TODO: cleanup\n";
            init = clean->getSubExpr();
          }

          VisitConstructCommon(cast<clang::CXXConstructExpr>(init),
                               /*name*/ nullptr, /*space*/ 0,
                               /*mem*/ ThisVal.val);
          continue;
        }
      }
      assert(field && "initialiation expression must apply to a field");
      if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getInit())) {
        VisitArrayInitLoop(AILE, CommonFieldLookup(loc, CC->getFunctionObjectParameterType(),
                                                   field, ThisVal.val,
                                                   /*isLValue*/ false));
        continue;
      }
      if (auto cons = dyn_cast<CXXConstructExpr>(expr->getInit())) {
        VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0,
                             CommonFieldLookup(loc, CC->getFunctionObjectParameterType(),
                                               field, ThisVal.val,
                                               /*isLValue*/ false)
                                 .val);
        continue;
      }
      auto initexpr = Visit(expr->getInit());
      if (!initexpr.val) {
        expr->getInit()->dump();
        assert(initexpr.val);
      }
      bool isArray = false;
      Glob.getMLIRType(expr->getInit()->getType(), &isArray);

      if (field->getType()->isReferenceType()) {
        assert(initexpr.isReference);
        initexpr.isReference = false;
        isArray = false;
      }

      auto cfl =
          CommonFieldLookup(loc, CC->getFunctionObjectParameterType(), field, ThisVal.val,
                            /*isLValue*/ false);
      assert(cfl.val);
      cfl.store(loc, builder, initexpr, isArray);
    }
  }
  if (auto CC = dyn_cast<CXXDestructorDecl>(fd)) {
    CC->dump();
    llvm::errs() << " warning, destructor not fully handled yet\n";
  }

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back({builder.create<mlir::memref::AllocaOp>(loc, type),
                   builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().keepRunning);
  if (function.getFunctionType().getResults().size()) {
    auto type = mlir::MemRefType::get(
        {}, function.getFunctionType().getResult(0), {}, 0);
    returnVal = builder.create<mlir::memref::AllocaOp>(loc, type);
    if (isa<mlir::IntegerType, mlir::FloatType>(type.getElementType())) {
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<polygeist::UndefOp>(loc, type.getElementType()),
          returnVal, std::vector<mlir::Value>({}));
    }
  }

  if (auto D = dyn_cast<CXXMethodDecl>(fd)) {
    // ClangAST incorrectly does not contain the correct definition
    // of a union move operation and as such we _must_ emit a memcpy
    // for a defaulted union copy or move.
    if (D->getParent()->isUnion() && D->isDefaulted()) {
      mlir::Value V = ThisVal.val;
      assert(V);
      if (auto MT = dyn_cast<MemRefType>(V.getType())) {
        V = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getContext()), V);
      }
      mlir::Value src = function.getArgument(1);
      if (auto MT = dyn_cast<MemRefType>(src.getType())) {
        src = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getContext()), src);
      }
      // With opaque pointers, get the union type from clang QualType.
      auto unionTy = Glob.getMLIRType(
          Glob.CGM.getContext().getCanonicalTagType(D->getParent()));
      mlir::Value typeSize = builder.create<polygeist::TypeSizeOp>(
          loc, builder.getIndexType(), mlir::TypeAttr::get(unionTy));
      typeSize = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(),
                                                    typeSize);
      // With opaque pointers, ptr-to-ptr bitcasts are identity. Remove them.
      builder.create<LLVM::MemcpyOp>(loc, V, src, typeSize,
                                     /*isVolatile*/ false);
    }
  }

  Stmt *stmt = fd->getBody();
  if (stmt) {
    if (ShowAST) {
      stmt->dump();
    }
    Visit(stmt);

    loc = getMLIRLocation(stmt->getEndLoc());
  } else {
    loc = getMLIRLocation(fd->getEndLoc());
  }

  if (function.getFunctionType().getResults().size()) {
    mlir::Value vals[1] = {
        builder.create<mlir::memref::LoadOp>(loc, returnVal)};
    builder.create<func::ReturnOp>(loc, vals);
  } else
    builder.create<func::ReturnOp>(loc);

  assert(function->getParentOp() == Glob.module.get() &&
         "New function must be inserted into global module");
}

mlir::OpBuilder &MLIRScanner::getBuilder() { return builder; }

mlir::Value MLIRScanner::createAllocOp(mlir::Type t, VarDecl *name,
                                       uint64_t memspace, bool isArray = false,
                                       bool LLVMABI = false) {

  mlir::MemRefType mr;
  mlir::Value alloc = nullptr;
  OpBuilder abuilder(builder.getContext());
  abuilder.setInsertionPointToStart(allocationScope);
  auto varLoc =
      name ? getMLIRLocation(name->getBeginLoc()) : builder.getUnknownLoc();
  if (!isArray) {
    if (LLVMABI) {
      if (name)
        if (auto var = dyn_cast<VariableArrayType>(
                name->getType()->getUnqualifiedDesugaredType())) {
          auto len = Visit(var->getSizeExpr()).getValue(varLoc, builder);
          alloc = builder.create<mlir::LLVM::AllocaOp>(
              varLoc, LLVM::LLVMPointerType::get(t.getContext(), memspace),
              t, len);
          builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
          // With opaque pointers, ptr-to-ptr bitcast is identity.
        }

      if (!alloc) {
        alloc = abuilder.create<mlir::LLVM::AllocaOp>(
            varLoc, mlir::LLVM::LLVMPointerType::get(t.getContext(), memspace),
            t, abuilder.create<arith::ConstantIntOp>(varLoc, 1, 64), 0);
        if (isa<mlir::IntegerType, mlir::FloatType>(t) && memspace == 0) {
          abuilder.create<LLVM::StoreOp>(
              varLoc, abuilder.create<polygeist::UndefOp>(varLoc, t), alloc);
        }
        // alloc = builder.create<mlir::LLVM::BitcastOp>(varLoc,
        // LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(t, 1)), alloc);
      }
    } else {
      mr = mlir::MemRefType::get(1, t, {}, memspace);
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc,
            mlir::MemRefType::get(ShapedType::kDynamic, t, {}, memspace),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(t.getContext(), 0), alloc));
      }
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(ShapedType::kDynamic, t, {}, 0), alloc);
      if (isa<mlir::IntegerType, mlir::FloatType>(t) && memspace == 0) {
        mlir::Value idxs[] = {abuilder.create<ConstantIndexOp>(varLoc, 0)};
        abuilder.create<mlir::memref::StoreOp>(
            varLoc, abuilder.create<polygeist::UndefOp>(varLoc, t), alloc,
            idxs);
      }
    }
  } else {
    auto mt = cast<mlir::MemRefType>(t);
    auto shape = std::vector<int64_t>(mt.getShape());
    auto pshape = shape[0];

    if (name)
      if (auto var = dyn_cast<VariableArrayType>(
              name->getType()->getUnqualifiedDesugaredType())) {
        assert(shape[0] == ShapedType::kDynamic);

        mr = mlir::MemRefType::get(
            shape, mt.getElementType(), MemRefLayoutAttrInterface(),
            wrapIntegerMemorySpace(memspace, mt.getContext()));

        auto sizeExpr = var->getSizeExpr();
        SmallVector<mlir::Value, 4> lens;
        auto *VAT = var;
        mlir::Value len = Visit(sizeExpr).getValue(varLoc, builder);
        len = builder.create<IndexCastOp>(varLoc, builder.getIndexType(), len);
        lens.push_back(len);

        while (isa<VariableArrayType>(VAT->getElementType())) {
          VAT = dyn_cast<VariableArrayType>(VAT->getElementType());
          VAT->dump();
          len = Visit(VAT->getSizeExpr()).getValue(varLoc, builder);
          len =
              builder.create<IndexCastOp>(varLoc, builder.getIndexType(), len);
          lens.push_back(len);
        }

        alloc = builder.create<mlir::memref::AllocaOp>(varLoc, mr, lens);
        builder.create<polygeist::TrivialUseOp>(varLoc, alloc);
        if (memspace != 0) {
          alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
              varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
              abuilder.create<polygeist::Memref2PointerOp>(
                  varLoc, LLVM::LLVMPointerType::get(mt.getContext(), 0),
                  alloc));
        }
      }

    if (!alloc) {
      if (pshape == ShapedType::kDynamic)
        shape[0] = 1;
      mr = mlir::MemRefType::get(
          shape, mt.getElementType(), MemRefLayoutAttrInterface(),
          wrapIntegerMemorySpace(memspace, mt.getContext()));
      alloc = abuilder.create<mlir::memref::AllocaOp>(varLoc, mr);
      if (memspace != 0) {
        alloc = abuilder.create<polygeist::Pointer2MemrefOp>(
            varLoc, mlir::MemRefType::get(shape, mt.getElementType()),
            abuilder.create<polygeist::Memref2PointerOp>(
                varLoc, LLVM::LLVMPointerType::get(mt.getContext(), 0),
                alloc));
      }
      shape[0] = pshape;
      alloc = abuilder.create<mlir::memref::CastOp>(
          varLoc, mlir::MemRefType::get(shape, mt.getElementType()), alloc);
    }
  }
  assert(alloc);
  // NamedAttribute attrs[] = {NamedAttribute("name", name)};
  if (name) {
    // if (name->getName() == "i")
    //  llvm_unreachable(" not i");
    if (params.find(name) != params.end()) {
      name->dump();
    }
    assert(params.find(name) == params.end());
    params[name] = ValueCategory(alloc, /*isReference*/ true);
  }
  return alloc;
}

ValueCategory
MLIRScanner::VisitExtVectorElementExpr(clang::ExtVectorElementExpr *expr) {
  auto base = Visit(expr->getBase());

  SmallVector<uint32_t, 4> indices;
  expr->getEncodedElementAccess(indices);
  assert(indices.size() == 1 &&
         "The support for higher dimensions to be implemented.");

  assert(base.isReference);
  base.isReference = false;

  const auto et = base.val.getType();
  assert(isa<LLVM::LLVMPointerType>(et) || isa<MemRefType>(et));

  ValueCategory result = nullptr;
  const auto exprLoc = getMLIRLocation(expr->getExprLoc());
  const auto accLoc = getMLIRLocation(expr->getAccessorLoc());
  const mlir::Value idxs[2] = {
      builder.create<ConstantIntOp>(exprLoc, 0, 32),
      builder.create<ConstantIntOp>(exprLoc, indices[0], 32),
  };

  if (const auto pt = dyn_cast<LLVM::LLVMPointerType>(et)) {
    // With opaque pointers, get the pointee type from the clang base type.
    auto baseTy = getMLIRType(expr->getBase()->getType()->getPointeeType());
    auto arrTy = cast<mlir::LLVM::LLVMArrayType>(baseTy);
    auto pt0 = arrTy.getElementType();
    base.val = builder.create<mlir::LLVM::GEPOp>(
        exprLoc, mlir::LLVM::LLVMPointerType::get(pt0.getContext(), pt.getAddressSpace()),
        arrTy, base.val, idxs);

    result = ValueCategory(base.val, true);
  } else if (const auto mt = dyn_cast<MemRefType>(et)) {
    auto shape = std::vector<int64_t>(mt.getShape());

    if (shape.size() == 1) {
      shape[0] = ShapedType::kDynamic;
    } else {
      shape.erase(shape.begin());
    }

    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    base.val = builder.create<polygeist::SubIndexOp>(
        exprLoc, mt0, base.val, castToIndex(accLoc, idxs[0]));

    result = CommonArrayLookup(exprLoc, base, castToIndex(accLoc, idxs[1]),
                               base.isReference);
  } else {
    llvm_unreachable("Unexpected MLIR type received");
  }

  return result;
}

ValueCategory MLIRScanner::VisitConstantExpr(clang::ConstantExpr *expr) {
  auto sv = Visit(expr->getSubExpr());
  if (auto ty = dyn_cast<mlir::IntegerType>(getMLIRType(expr->getType()))) {
    if (expr->hasAPValueResult()) {
      return ValueCategory(builder.create<arith::ConstantIntOp>(
                               getMLIRLocation(expr->getExprLoc()),
                               ty, expr->getResultAsAPSInt().getExtValue()),
                           /*isReference*/ false);
    }
  }
  assert(sv.val);
  return sv;
}

ValueCategory MLIRScanner::VisitTypeTraitExpr(clang::TypeTraitExpr *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           ty, (int64_t)expr->getBoolValue()),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitGNUNullExpr(clang::GNUNullExpr *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(builder.create<arith::ConstantIntOp>(
                           getMLIRLocation(expr->getExprLoc()), ty, 0),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitIntegerLiteral(clang::IntegerLiteral *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           ty, expr->getValue().getSExtValue()),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitCharacterLiteral(clang::CharacterLiteral *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<arith::ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                           ty, expr->getValue()),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitFloatingLiteral(clang::FloatingLiteral *expr) {
  auto ty = cast<mlir::FloatType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<ConstantFloatOp>(getMLIRLocation(expr->getExprLoc()),
                                      ty, expr->getValue()),
      /*isReference*/ false);
}

ValueCategory
MLIRScanner::VisitImaginaryLiteral(clang::ImaginaryLiteral *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto convertedType = getMLIRType(expr->getType());
  mlir::FloatType fty;
  if (auto mt = dyn_cast<MemRefType>(convertedType)) {
    auto elty = mt.getElementType();
    if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
      fty = cast<FloatType>(ST.getBody()[0]);
    } else {
      fty = cast<FloatType>(elty);
    }
  } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(convertedType)) {
    fty = cast<FloatType>(ST.getBody()[0]);
  } else {
    llvm_unreachable("unexpected complex type\n");
  }

  auto zero = builder.create<ConstantFloatOp>(
      loc, fty, APFloat(fty.getFloatSemantics(), "0"));
  auto imag = Visit(expr->getSubExpr()).getValue(loc, builder);
  return createComplexFloat(loc, zero, imag, expr->getType());
}

ValueCategory
MLIRScanner::VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    ty, expr->getValue()),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStringLiteral(clang::StringLiteral *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  return ValueCategory(
      Glob.GetOrCreateGlobalLLVMString(loc, builder, expr->getString()),
      /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitParenExpr(clang::ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory
MLIRScanner::VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *decl) {
  auto Mty = getMLIRType(decl->getType());
  auto loc = getMLIRLocation(decl->getExprLoc());

  if (auto FT = dyn_cast<mlir::FloatType>(Mty))
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, FT, APFloat(FT.getFloatSemantics(), "0")),
                         /*isReference*/ false);
  if (auto IT = dyn_cast<mlir::IntegerType>(Mty))
    return ValueCategory(builder.create<ConstantIntOp>(loc, IT, 0),
                         /*isReference*/ false);
  if (auto MT = dyn_cast<mlir::MemRefType>(Mty))
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::ZeroOp>(
                loc, LLVM::LLVMPointerType::get(builder.getContext()))),
        false);
  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(Mty))
    return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, PT), false);
  for (auto child : decl->children()) {
    child->dump();
  }
  decl->dump();
  llvm::errs() << " mty: " << Mty << "\n";
  llvm_unreachable("bad");
}

/// Construct corresponding MLIR operations to initialize the given value by a
/// provided InitListExpr.
mlir::Attribute MLIRScanner::InitializeValueByInitListExpr(mlir::Value toInit,
                                                           clang::Expr *expr) {
  // Struct initializan requires an extra 0, since the first index
  // is the pointer index, and then the struct index.
  auto PTT = expr->getType()->getUnqualifiedDesugaredType();

  bool inner = false;
  if (isa<RecordType>(PTT) || isa<clang::ComplexType>(PTT)) {
    if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
      inner = true;
    }
  }

  while (auto CO = toInit.getDefiningOp<memref::CastOp>())
    toInit = CO.getSource();

  // Recursively visit the initialization expression following the linear
  // increment of the memory address.
  std::function<mlir::DenseElementsAttr(Expr *, mlir::Value, bool)> helper =
      [&](Expr *expr, mlir::Value toInit,
          bool inner) -> mlir::DenseElementsAttr {
    Location loc = toInit.getLoc();
    if (InitListExpr *initListExpr = dyn_cast<InitListExpr>(expr)) {

      if (inner) {
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
          auto shape = std::vector<int64_t>(mt.getShape());
          shape.erase(shape.begin());
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          toInit = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                         getConstantIndex(0));
        }
      }

      unsigned num = 0;
      if (initListExpr->hasArrayFiller()) {
        if (auto MT = dyn_cast<MemRefType>(toInit.getType())) {
          auto shape = MT.getShape();
          assert(shape.size() > 0);
          assert(shape[0] != ShapedType::kDynamic);
          num = shape[0];
        } else if (auto PT =
                       dyn_cast<LLVM::LLVMPointerType>(toInit.getType())) {
          // With opaque pointers, get pointee type from clang type.
          auto pointeeTy = getMLIRType(expr->getType());
          if (auto AT = dyn_cast<LLVM::LLVMArrayType>(pointeeTy)) {
            num = AT.getNumElements();
          } else if (auto AT =
                         dyn_cast<LLVM::LLVMStructType>(pointeeTy)) {
            num = AT.getBody().size();
          } else {
            toInit.getType().dump();
            llvm_unreachable(
                "TODO get number of values in array filler expression");
          }
        } else {
          toInit.getType().dump();
          llvm_unreachable(
              "TODO get number of values in array filler expression");
        }
      } else {
        num = initListExpr->getNumInits();
      }

      SmallVector<char> attrs;
      bool allSub = true;

      if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        if (shape.size() == 0) {
          auto ET = mt.getElementType();
          if (isa<mlir::LLVM::LLVMStructType>(ET) ||
              isa<mlir::LLVM::LLVMArrayType>(ET))
            toInit = builder.create<polygeist::Memref2PointerOp>(
                loc, LLVM::LLVMPointerType::get(ET.getContext(), mt.getMemorySpaceAsInt()),
                toInit);
          // else if (isa<MemRefType>(ET))
          //   toInit = builder.create<memref::LoadOp>(loc, toInit);
          else
            assert(0);
        }
      }

      for (unsigned i = 0, e = num; i < e; ++i) {

        mlir::Value next;
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
          auto shape = std::vector<int64_t>(mt.getShape());
          assert(shape.size() > 0);
          shape[0] = ShapedType::kDynamic;
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          next = builder.create<polygeist::SubIndexOp>(loc, mt0, toInit,
                                                       getConstantIndex(i));
        } else {
          auto PT = cast<LLVM::LLVMPointerType>(toInit.getType());
          // With opaque pointers, get pointee type from clang type.
          auto ET = getMLIRType(expr->getType());
          mlir::Type nextType;
          if (auto ST = dyn_cast<LLVM::LLVMStructType>(ET))
            nextType = ST.getBody()[i];
          else if (auto AT = dyn_cast<LLVM::LLVMArrayType>(ET))
            nextType = AT.getElementType();
          else
            llvm_unreachable("unknown inner type");

          mlir::Value idxs[] = {
              builder.create<ConstantIntOp>(loc, 0, 32),
              builder.create<ConstantIntOp>(loc, i, 32),
          };
          next = builder.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(toInit.getContext(), PT.getAddressSpace()),
              ET, toInit, idxs);
        }

        auto expr =
            (initListExpr->hasArrayFiller() && i >= initListExpr->getNumInits())
                ? initListExpr->getArrayFiller()
                : initListExpr->getInit(i);
        auto sub = helper(expr, next, true);
        if (sub) {
          size_t n = 1;
          if (sub.isSplat())
            n = sub.size();
          for (size_t i = 0; i < n; i++)
            for (auto ea : sub.getRawData())
              attrs.push_back(ea);
        } else {
          allSub = false;
        }
      }
      if (!allSub)
        return mlir::DenseElementsAttr();
      if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
        std::vector<int64_t> shape(mt.getShape());
        assert(shape.size() > 0);
        shape[0] = num;
        return DenseElementsAttr::getFromRawBuffer(
            RankedTensorType::get(shape, mt.getElementType()), attrs);
      }
      return mlir::DenseElementsAttr();
    } else {
      bool isArray = false;
      Glob.getMLIRType(expr->getType(), &isArray);
      ValueCategory sub = Visit(expr);
      ValueCategory(toInit, /*isReference*/ true)
          .store(loc, builder, sub, isArray);
      if (!sub.isReference)
        if (auto mt = dyn_cast<MemRefType>(toInit.getType())) {
          if (auto cop = sub.val.getDefiningOp<ConstantIntOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
          if (auto cop = sub.val.getDefiningOp<ConstantFloatOp>())
            return DenseElementsAttr::get(
                RankedTensorType::get(std::vector<int64_t>({1}),
                                      mt.getElementType()),
                cop.getValue());
        }
      return mlir::DenseElementsAttr();
    }
  };

  return helper(expr, toInit, inner);
}

ValueCategory MLIRScanner::VisitVarDecl(clang::VarDecl *decl) {
  decl = decl->getCanonicalDecl();
  auto varLoc = getMLIRLocation(decl->getBeginLoc());
  mlir::Type subType = getMLIRType(decl->getType());
  ValueCategory inite = nullptr;
  unsigned memtype = decl->hasAttr<CUDASharedAttr>() ? 5 : 0;
  bool LLVMABI = false;
  bool isArray = false;

  if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
              Glob.CGM.getContext().getLValueReferenceType(decl->getType()))))
    LLVMABI = true;
  else
    Glob.getMLIRType(decl->getType(), &isArray);

  if (!LLVMABI && isArray) {
    subType = Glob.getMLIRType(
        Glob.CGM.getContext().getLValueReferenceType(decl->getType()));
  }

  if (auto init = decl->getInit()) {
    if (!isa<InitListExpr>(init) && !isa<CXXConstructExpr>(init)) {
      auto visit = Visit(init);
      if (!visit.val) {
        decl->dump();
        assert(visit.val);
      }
      bool isReference = init->isLValue() || init->isXValue();
      if (isReference) {
        assert(visit.isReference);
        builder.create<polygeist::TrivialUseOp>(varLoc, visit.val);
        return params[decl] = visit;
      }
      if (isArray) {
        assert(visit.isReference);
        inite = visit;
      } else {
        inite = ValueCategory(visit.getValue(varLoc, builder), /*isRef*/ false);
        if (!inite.val) {
          init->dump();
          llvm_unreachable("?");
        }
        subType = inite.val.getType();
      }
    }
  } else if (auto ava = decl->getAttr<AlignValueAttr>()) {
    if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
      for (auto a : algn->children()) {
        if (auto IL = dyn_cast<IntegerLiteral>(a)) {
          if (IL->getValue() == 8192) {
            llvm::Type *T = Glob.CGM.getTypes().ConvertType(decl->getType());
            subType = Glob.typeTranslator.translateType(T);
            LLVMABI = true;
            break;
          }
        }
      }
    }
  } else if (auto ava = decl->getAttr<InitPriorityAttr>()) {
    if (ava->getPriority() == 8192) {
      llvm::Type *T = Glob.CGM.getTypes().ConvertType(decl->getType());
      subType = Glob.typeTranslator.translateType(T);
      LLVMABI = true;
    }
  }

  mlir::Value op;

  Block *block = nullptr;
  Block::iterator iter;

  if (decl->isStaticLocal() && memtype == 0) {
    OpBuilder abuilder(builder.getContext());
    abuilder.setInsertionPointToStart(allocationScope);

    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(Glob.CGM.getContext().getPointerType(decl->getType())))) {
      op = abuilder.create<mlir::LLVM::AddressOfOp>(
          varLoc, Glob.GetOrCreateLLVMGlobal(
                      decl, (function.getName() + "@static@").str()));
    } else {
      auto gv = Glob.GetOrCreateGlobal(
          decl, (function.getName() + "@static@").str(), /*tryInit*/ false);
      op = abuilder.create<memref::GetGlobalOp>(varLoc, gv.first.getType(),
                                                gv.first.getName());
      auto mt = cast<MemRefType>(gv.first.getType());
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      op = abuilder.create<memref::CastOp>(
          varLoc,
          MemRefType::get(shape, mt.getElementType(),
                          MemRefLayoutAttrInterface(), mt.getMemorySpace()),
          op);
    }
    params[decl] = ValueCategory(op, /*isReference*/ true);
    if (decl->getInit()) {
      auto mr = MemRefType::get({1}, builder.getI1Type());
      bool inits[1] = {true};
      auto rtt = RankedTensorType::get({1}, builder.getI1Type());
      auto init_value = DenseIntElementsAttr::get(rtt, inits);
      OpBuilder gbuilder(builder.getContext());
      gbuilder.setInsertionPointToStart(module->getBody());
      auto name = Glob.CGM.getMangledName(decl);
      auto globalOp = gbuilder.create<mlir::memref::GlobalOp>(
          module->getLoc(),
          builder.getStringAttr(function.getName() + "@static@" + name +
                                "@init"),
          /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
          init_value, mlir::UnitAttr(), /*alignment*/ nullptr);
      SymbolTable::setSymbolVisibility(globalOp,
                                       mlir::SymbolTable::Visibility::Private);

      auto boolop =
          builder.create<memref::GetGlobalOp>(varLoc, mr, globalOp.getName());
      auto cond = builder.create<memref::LoadOp>(
          varLoc, boolop, std::vector<mlir::Value>({getConstantIndex(0)}));

      auto ifOp = builder.create<scf::IfOp>(varLoc, cond, /*hasElse*/ false);
      block = builder.getInsertionBlock();
      iter = builder.getInsertionPoint();
      builder.setInsertionPointToStart(&ifOp.getThenRegion().back());
      builder.create<memref::StoreOp>(
          varLoc, builder.create<ConstantIntOp>(varLoc, false, 1), boolop,
          std::vector<mlir::Value>({getConstantIndex(0)}));
    }
  } else
    op = createAllocOp(subType, decl, memtype, isArray, LLVMABI);

  if (inite.val) {
    ValueCategory(op, /*isReference*/ true)
        .store(varLoc, builder, inite, isArray);
  } else if (auto init = decl->getInit()) {
    if (isa<InitListExpr>(init)) {
      InitializeValueByInitListExpr(op, init);
    } else if (auto CE = dyn_cast<CXXConstructExpr>(init)) {
      VisitConstructCommon(CE, decl, memtype, op);
    } else
      llvm_unreachable("unknown init list");
  }
  if (block)
    builder.setInsertionPoint(block, iter);
  return ValueCategory(op, /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr *expr) {
  return Visit(expr->getExpr());
}

ValueCategory MLIRScanner::VisitCXXThisExpr(clang::CXXThisExpr *expr) {
  return ThisVal;
}

ValueCategory MLIRScanner::VisitPredefinedExpr(clang::PredefinedExpr *expr) {
  return VisitStringLiteral(expr->getFunctionName());
}

ValueCategory MLIRScanner::VisitInitListExpr(clang::InitListExpr *expr) {
  mlir::Type subType = getMLIRType(expr->getType());
  bool isArray = false;
  bool LLVMABI = false;

  if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(
              Glob.CGM.getContext().getLValueReferenceType(expr->getType()))))
    LLVMABI = true;
  else {
    Glob.getMLIRType(expr->getType(), &isArray);
    if (isArray)
      subType = Glob.getMLIRType(
          Glob.CGM.getContext().getLValueReferenceType(expr->getType()));
  }
  auto op = createAllocOp(subType, nullptr, /*memtype*/ 0, isArray, LLVMABI);
  InitializeValueByInitListExpr(op, expr);
  return ValueCategory(op, true);
}

ValueCategory MLIRScanner::VisitCXXStdInitializerListExpr(
    clang::CXXStdInitializerListExpr *expr) {

  auto loc = getMLIRLocation(expr->getExprLoc());

  auto ArrayPtr = Visit(expr->getSubExpr());

  const ConstantArrayType *ArrayType =
      Glob.CGM.getContext().getAsConstantArrayType(
          expr->getSubExpr()->getType());
  assert(ArrayType && "std::initializer_list constructed from non-array");

  // FIXME: Perform the checks on the field types in SemaInit.
  RecordDecl *Record = expr->getType()->castAs<RecordType>()->getDecl();
  auto Field = Record->field_begin();

  mlir::Type subType = getMLIRType(expr->getType());

  mlir::Value res = builder.create<polygeist::UndefOp>(loc, subType);

  ArrayPtr = CommonArrayToPointer(loc, ArrayPtr);

  res = builder.create<LLVM::InsertValueOp>(loc, res.getType(), res,
                                            ArrayPtr.getValue(loc, builder),
                                            builder.getDenseI64ArrayAttr(0));
  Field++;
  auto iTy = cast<mlir::IntegerType>(getMLIRType(Field->getType()));
  res = builder.create<LLVM::InsertValueOp>(
      loc, res.getType(), res,
      builder.create<arith::ConstantIntOp>(
          loc, ArrayType->getSize().getZExtValue(), iTy.getWidth()),
      builder.getDenseI64ArrayAttr(1));
  return ValueCategory(res, /*isRef*/ false);
}

ValueCategory
MLIRScanner::VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr *expr) {
  assert(arrayinit.size());
  auto loc = getMLIRLocation(expr->getExprLoc());
  return ValueCategory(builder.create<IndexCastOp>(
                           loc, getMLIRType(expr->getType()), arrayinit.back()),
                       /*isReference*/ false);
}

static const clang::ConstantArrayType *getCAT(const clang::Type *T) {
  const clang::Type *Child;
  if (auto CAT = dyn_cast<clang::ConstantArrayType>(T)) {
    return CAT;
  } else if (auto TypeDefT = dyn_cast<clang::TypedefType>(T)) {
    Child = TypeDefT->getUnqualifiedDesugaredType();
  } else {
    llvm_unreachable("Unhandled case\n");
  }
  return getCAT(Child);
}

ValueCategory MLIRScanner::VisitArrayInitLoop(clang::ArrayInitLoopExpr *expr,
                                              ValueCategory tostore) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  const clang::ConstantArrayType *CAT = getCAT(expr->getType().getTypePtr());
  llvm::errs() << "warning recomputing common in arrayinitloopexpr\n";
  std::vector<mlir::Value> start = {getConstantIndex(0)};
  std::vector<mlir::Value> sizes = {
      getConstantIndex(CAT->getSize().getLimitedValue())};
  AffineMap map = builder.getSymbolIdentityMap();
  auto affineOp =
      builder.create<affine::AffineForOp>(loc, start, map, sizes, map);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.getRegion().getBlocks().front());

  arrayinit.push_back(affineOp.getInductionVar());

  auto alu =
      CommonArrayLookup(loc, CommonArrayToPointer(loc, tostore),
                        affineOp.getInductionVar(), /*isImplicitRef*/ false);

  if (auto AILE = dyn_cast<ArrayInitLoopExpr>(expr->getSubExpr())) {
    VisitArrayInitLoop(AILE, alu);
  } else {
    auto val = Visit(expr->getSubExpr());
    if (!val.val) {
      expr->dump();
      expr->getSubExpr()->dump();
    }
    assert(val.val);
    assert(tostore.isReference);
    bool isArray = false;
    Glob.getMLIRType(expr->getSubExpr()->getType(), &isArray);
    alu.store(loc, builder, val, isArray);
  }

  arrayinit.pop_back();

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *expr) {
  if (expr->getType()->isVoidType()) {
    Visit(expr->getSubExpr());
    return nullptr;
  }
  if (expr->getCastKind() == clang::CastKind::CK_NoOp)
    return Visit(expr->getSubExpr());
  if (expr->getCastKind() == clang::CastKind::CK_ConstructorConversion)
    return Visit(expr->getSubExpr());
  return VisitCastExpr(expr);
}

ValueCategory
MLIRScanner::VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr) {
  return Visit(expr->getSubExpr());
}

ValueCategory MLIRScanner::VisitLambdaExpr(clang::LambdaExpr *expr) {

  // llvm::DenseMap<const VarDecl *, FieldDecl *> InnerCaptures;
  // FieldDecl *ThisCapture = nullptr;

  // expr->getLambdaClass()->getCaptureFields(InnerCaptures, ThisCapture);
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool LLVMABI = false;
  mlir::Type t = Glob.getMLIRType(expr->getCallOperator()->getThisType());

  bool isArray = false;
  Glob.getMLIRType(expr->getCallOperator()->getFunctionObjectParameterType(), &isArray);

  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(t)) {
    LLVMABI = true;
    // With opaque pointers, get the pointee type from the clang type.
    t = Glob.getMLIRType(expr->getCallOperator()->getFunctionObjectParameterType());
  }
  if (auto mt = dyn_cast<MemRefType>(t)) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (!isArray) {
      t = mt.getElementType();
    }
  }
  auto op = createAllocOp(t, nullptr, /*memtype*/ 0, isArray, LLVMABI);

  for (auto tup : llvm::zip(expr->getLambdaClass()->captures(),
                            expr->getLambdaClass()->fields())) {
    auto C = std::get<0>(tup);
    auto field = std::get<1>(tup);
    if (C.capturesThis())
      continue;
    else if (!C.capturesVariable())
      continue;

    auto CK = C.getCaptureKind();
    auto var = C.getCapturedVar();

    ValueCategory result;

    if (params.find(var) != params.end()) {
      result = params[var];
    } else {
      if (auto VD = dyn_cast<VarDecl>(var)) {
        if (Captures.find(VD) != Captures.end()) {
          FieldDecl *field = Captures[VD];
          result = CommonFieldLookup(
              loc,
              cast<CXXMethodDecl>(EmittingFunctionDecl)->getFunctionObjectParameterType(),
              field, ThisVal.val, /*isLValue*/ false);
          assert(CaptureKinds.find(VD) != CaptureKinds.end());
          if (CaptureKinds[VD] == LambdaCaptureKind::LCK_ByRef)
            result = result.dereference(loc, builder);
          goto endp;
        }
      }
      EmittingFunctionDecl->dump();
      expr->dump();
      function.dump();
      llvm::errs() << "<pairs>\n";
      for (auto p : params)
        p.first->dump();
      llvm::errs() << "</pairs>";
      var->dump();
    }
  endp:

    bool isArray = false;
    Glob.getMLIRType(field->getType(), &isArray);

    if (CK == LambdaCaptureKind::LCK_ByCopy)
      CommonFieldLookup(loc, expr->getCallOperator()->getFunctionObjectParameterType(),
                        field, op,
                        /*isLValue*/ false)
          .store(loc, builder, result, isArray);
    else {
      assert(CK == LambdaCaptureKind::LCK_ByRef);
      assert(result.isReference);

      auto val = result.val;

      if (auto mt = dyn_cast<MemRefType>(val.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        shape[0] = ShapedType::kDynamic;
        val = builder.create<memref::CastOp>(
            loc,
            MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace()),
            val);
      }

      CommonFieldLookup(loc, expr->getCallOperator()->getFunctionObjectParameterType(),
                        field, op,
                        /*isLValue*/ false)
          .store(loc, builder, val);
    }
  }
  return ValueCategory(op, /*isReference*/ true);
}

// TODO actually deallocate
ValueCategory MLIRScanner::VisitMaterializeTemporaryExpr(
    clang::MaterializeTemporaryExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto v = Visit(expr->getSubExpr());
  if (!v.val) {
    expr->dump();
  }
  assert(v.val);

  bool isArray = false;
  bool LLVMABI = false;
  if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
                           expr->getSubExpr()->getType()))))
    LLVMABI = true;
  else {
    Glob.getMLIRType(expr->getSubExpr()->getType(), &isArray);
  }
  if (isArray)
    return v;

  llvm::errs() << "cleanup of materialized not handled";
  auto op = createAllocOp(getMLIRType(expr->getSubExpr()->getType()), nullptr,
                          0, /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);

  ValueCategory(op, /*isRefererence*/ true).store(loc, builder, v, isArray);
  return ValueCategory(op, /*isRefererence*/ true);
}

ValueCategory MLIRScanner::VisitCXXDeleteExpr(clang::CXXDeleteExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  expr->dump();
  llvm::errs() << "warning not calling destructor on delete\n";

  mlir::Value toDelete = Visit(expr->getArgument()).getValue(loc, builder);

  if (isa<mlir::MemRefType>(toDelete.getType())) {
    builder.create<mlir::memref::DeallocOp>(loc, toDelete);
  } else {
    mlir::Value args[1] = {toDelete};
    builder.create<mlir::LLVM::CallOp>(loc, Glob.GetOrCreateFreeFunction(),
                                       args);
  }

  return nullptr;
}
ValueCategory MLIRScanner::VisitCXXNewExpr(clang::CXXNewExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  mlir::Value count;

  if (expr->isArray()) {
    count = Visit(*expr->raw_arg_begin()).getValue(loc, builder);
    count = builder.create<IndexCastOp>(
        loc, mlir::IndexType::get(builder.getContext()), count);
  } else {
    count = getConstantIndex(1);
  }
  assert(count);

  auto ty = getMLIRType(expr->getType());

  mlir::Value alloc;
  mlir::Value arrayCons;
  if (!expr->placement_arguments().empty()) {
    mlir::Value val =
        Visit(*expr->placement_arg_begin()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(ty)) {
      if (auto mtin = dyn_cast<mlir::MemRefType>(val.getType())) {
        val = builder.create<polygeist::Memref2PointerOp>(
            loc,
            LLVM::LLVMPointerType::get(mtin.getContext(), mtin.getMemorySpaceAsInt()),
            val);
      }
      arrayCons = alloc =
          builder.create<polygeist::Pointer2MemrefOp>(loc, mt, val);
    } else {
      // With opaque pointers, ptr-to-ptr bitcasts are identity.
      arrayCons = alloc = val;
    }
  } else if (auto mt = dyn_cast<mlir::MemRefType>(ty)) {
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value args[1] = {count};
    arrayCons = alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
  } else {
    auto typeSize = getTypeSize(loc, expr->getAllocatedType());
    mlir::Value arg = builder.create<arith::MulIOp>(loc, typeSize, count);
    // With opaque pointers, ptr-to-ptr bitcast is identity.
    arrayCons = alloc = Glob.CallMalloc(builder, loc, arg);
  }
  assert(alloc);
  if (expr->hasInitializer()) {
    auto init = expr->getInitializer();
    if (isa<InitListExpr>(init)) {
      (void)InitializeValueByInitListExpr(alloc, init);
    } else if (isa<CXXConstructExpr>(init)) {
      assert(arrayCons);
      VisitConstructCommon(
          const_cast<CXXConstructExpr *>(expr->getConstructExpr()),
          /*name*/ nullptr, /*memtype*/ 0, arrayCons, count);
    } else {
      ValueCategory val = Visit(init);
      ValueCategory(alloc, /* isReference */ true)
          .store(loc, builder, val, /* isArray */ false);
    }
  }
  return ValueCategory(alloc, /*isRefererence*/ false);
}

mlir::Value add(MLIRScanner &sc, mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value lhs, mlir::Value rhs) {
  assert(lhs);
  assert(rhs);
  if (auto op = lhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = lhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return rhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIntOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }

  if (auto op = rhs.getDefiningOp<ConstantIndexOp>()) {
    if (op.value() == 0) {
      return lhs;
    }
  }
  return builder.create<AddIOp>(loc, lhs, rhs);
}

mlir::Value MLIRScanner::castToIndex(mlir::Location loc, mlir::Value val) {
  assert(val && "Expect non-null value");

  if (auto op = val.getDefiningOp<ConstantIntOp>())
    return getConstantIndex(op.value());

  return builder.create<arith::IndexCastOp>(
      loc, mlir::IndexType::get(val.getContext()), val);
}

ValueCategory
MLIRScanner::VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());

  bool isArray = false;
  mlir::Type melem = Glob.getMLIRType(expr->getType(), &isArray);
  assert(!isArray);

  if (isa<mlir::IntegerType>(melem))
    return ValueCategory(builder.create<ConstantIntOp>(loc, melem, 0), false);
  else if (auto MT = dyn_cast<mlir::MemRefType>(melem))
    return ValueCategory(
        builder.create<polygeist::Pointer2MemrefOp>(
            loc, MT,
            builder.create<mlir::LLVM::ZeroOp>(
                loc, LLVM::LLVMPointerType::get(builder.getContext()))),
        false);
  else if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(melem))
    return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, PT), false);
  else {
    if (!isa<FloatType>(melem))
      expr->dump();
    auto ft = cast<FloatType>(melem);
    return ValueCategory(builder.create<ConstantFloatOp>(
                             loc, ft, APFloat(ft.getFloatSemantics(), "0")),
                         false);
  }
}

ValueCategory MLIRScanner::VisitCXXPseudoDestructorExpr(
    clang::CXXPseudoDestructorExpr *expr) {
  Visit(expr->getBase());
  llvm::errs() << "not running pseudo destructor\n";
  return nullptr;
}

ValueCategory
MLIRScanner::VisitCXXConstructExpr(clang::CXXConstructExpr *cons) {
  return VisitConstructCommon(cons, /*name*/ nullptr, /*space*/ 0);
}

ValueCategory MLIRScanner::VisitConstructCommon(clang::CXXConstructExpr *cons,
                                                VarDecl *name, unsigned memtype,
                                                mlir::Value op,
                                                mlir::Value count) {
  auto loc = getMLIRLocation(cons->getExprLoc());

  bool isArray = false;
  mlir::Type subType = Glob.getMLIRType(cons->getType(), &isArray);

  bool LLVMABI = false;
  auto ptrty = Glob.getMLIRType(
      Glob.CGM.getContext().getLValueReferenceType(cons->getType()));
  if (isa<mlir::LLVM::LLVMPointerType>(ptrty))
    LLVMABI = true;
  else if (isArray) {
    subType = ptrty;
    isArray = true;
  }
  if (op == nullptr)
    op = createAllocOp(subType, name, memtype, isArray, LLVMABI);

  auto decl = cons->getConstructor();
  if (cons->requiresZeroInitialization()) {
    mlir::Value val = op;
    if (isa<MemRefType>(val.getType())) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), val);
    }
    // With opaque pointers, ptr-to-ptr bitcast is identity.
    mlir::Value size = getTypeSize(loc, cons->getType());

    auto i8_0 = builder.create<ConstantIntOp>(loc, 0, 8);
    auto sizev =
        builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), size);

    builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev,
                                    /*isVolatile=*/false);
  }

  if (decl->isTrivial() && decl->isDefaultConstructor())
    return ValueCategory(op, /*isReference*/ true);

  mlir::Block::iterator oldpoint;
  mlir::Block *oldblock;
  ValueCategory endobj(op, /*isReference*/ true);

  ValueCategory obj(op, /*isReference*/ true);
  QualType innerType = cons->getType();
  if (auto arrayType = Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    innerType = arrayType->getElementType();
    mlir::Value size;
    if (count)
      size = count;
    else {
      auto CAT = cast<clang::ConstantArrayType>(arrayType);
      size = getConstantIndex(CAT->getSize().getLimitedValue());
    }
    auto forOp = builder.create<scf::ForOp>(loc, getConstantIndex(0), size,
                                            getConstantIndex(1));
    oldpoint = builder.getInsertionPoint();
    oldblock = builder.getInsertionBlock();

    builder.setInsertionPointToStart(&forOp.getRegion().getBlocks().front());
    assert(obj.isReference);
    obj = CommonArrayToPointer(loc, obj);
    obj = CommonArrayLookup(loc, obj, forOp.getInductionVar(),
                            /*isImplicitRef*/ false, /*removeIndex*/ false);
    assert(obj.isReference);
  }

  auto tocall = Glob.GetOrCreateMLIRFunction(cons->getConstructor());

  SmallVector<std::pair<ValueCategory, clang::Expr *>> args;
  args.emplace_back(make_pair(obj, (clang::Expr *)nullptr));
  for (auto a : cons->arguments())
    args.push_back(make_pair(Visit(a), a));
  CallHelper(tocall, innerType, args,
             /*retType*/ Glob.CGM.getContext().VoidTy, false, cons);

  if (Glob.CGM.getContext().getAsArrayType(cons->getType())) {
    builder.setInsertionPoint(oldblock, oldpoint);
  }
  return endobj;
}

ValueCategory MLIRScanner::CommonArrayToPointer(mlir::Location loc,
                                                ValueCategory scalar) {
  assert(scalar.val);
  assert(scalar.isReference);
  if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.val.getType())) {
    // With opaque pointers, deduce pointee type from defining op.
    auto pointeeTy = deducePointeeType(scalar.val);
    if (isa_and_nonnull<mlir::LLVM::LLVMPointerType>(pointeeTy))
      return ValueCategory(scalar.val, /*isRef*/ false);
    mlir::Value vec[2] = {builder.create<ConstantIntOp>(loc, 0, 32),
                          builder.create<ConstantIntOp>(loc, 0, 32)};
    if (!isa_and_nonnull<mlir::LLVM::LLVMArrayType>(pointeeTy)) {
      EmittingFunctionDecl->dump();
      function.dump();
      llvm::errs() << " sval: " << scalar.val << "\n";
      llvm::errs() << PT << "\n";
    }
    auto arrTy = cast<mlir::LLVM::LLVMArrayType>(pointeeTy);
    auto ET = arrTy.getElementType();
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(scalar.val.getContext(), PT.getAddressSpace()),
            arrTy, scalar.val, vec),
        /*isReference*/ false);
  }

  auto mt = cast<MemRefType>(scalar.val.getType());
  auto shape = std::vector<int64_t>(mt.getShape());
  LLVM_DEBUG(llvm::dbgs() << "scalar in CommonArrayToPointer: " << scalar.val
                          << "\n");
  // if (shape.size() > 1) {
  //   shape.erase(shape.begin());
  // } else {
  //   shape[0] = ShapedType::kDynamic;
  // }
  shape[0] = ShapedType::kDynamic;
  // LLVM_DEBUG(llvm::dbgs() << "scalar: " << scalar.val << "\n");
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());

  auto post = builder.create<memref::CastOp>(loc, mt0, scalar.val);
  return ValueCategory(post, /*isReference*/ false);
}

ValueCategory MLIRScanner::CommonArrayLookup(mlir::Location loc,
                                             ValueCategory array,
                                             mlir::Value idx,
                                             bool isImplicitRefResult,
                                             bool removeIndex) {
  mlir::Value val = array.getValue(loc, builder);
  assert(val);
  if (isa<LLVM::LLVMPointerType>(val.getType())) {

    mlir::Value vals[] = {
        builder.create<IndexCastOp>(loc, builder.getIntegerType(64), idx)};
    // TODO sub
    auto elemTy = deducePointeeType(val);
    if (!elemTy)
      elemTy = builder.getI8Type();
    return ValueCategory(
        builder.create<mlir::LLVM::GEPOp>(loc, val.getType(), elemTy, val,
                                           vals),
        /*isReference*/ true);
  }
  if (!isa<MemRefType>(val.getType())) {
    EmittingFunctionDecl->dump();
    builder.getInsertionBlock()->dump();
    function.dump();
    llvm::errs() << "value: " << val << "\n";
  }

  ValueCategory dref;
  {
    auto mt = cast<MemRefType>(val.getType());
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = ShapedType::kDynamic;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, val, idx);
    // TODO sub
    dref = ValueCategory(post, /*isReference*/ true);
  }
  assert(dref.isReference);
  if (!removeIndex)
    return dref;

  auto mt = cast<MemRefType>(dref.val.getType());
  auto shape = std::vector<int64_t>(mt.getShape());
  if (shape.size() == 1 || (shape.size() == 2 && isImplicitRefResult)) {
    shape[0] = ShapedType::kDynamic;
  } else {
    shape.erase(shape.begin());
  }
  auto mt0 =
      mlir::MemRefType::get(shape, mt.getElementType(),
                            MemRefLayoutAttrInterface(), mt.getMemorySpace());
  auto post = builder.create<polygeist::SubIndexOp>(loc, mt0, dref.val,
                                                    getConstantIndex(0));
  return ValueCategory(post, /*isReference*/ true);
}

ValueCategory
MLIRScanner::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  auto moo = Visit(expr->getLHS());

  auto rhs = Visit(expr->getRHS()).getValue(loc, builder);
  // Check the RHS has been successfully emitted
  assert(rhs);
  auto idx = castToIndex(getMLIRLocation(expr->getRBracketLoc()), rhs);
  if (isa<clang::VectorType>(
          expr->getLHS()->getType()->getUnqualifiedDesugaredType())) {
    assert(moo.isReference);
    moo.isReference = false;
    auto mt = cast<MemRefType>(moo.val.getType());
    auto shape = std::vector<int64_t>(mt.getShape());
    shape.erase(shape.begin());
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    moo.val = builder.create<polygeist::SubIndexOp>(loc, mt0, moo.val,
                                                    getConstantIndex(0));
  }
  bool isArray = false;
  if (!Glob.CGM.getContext().getAsArrayType(expr->getType()))
    Glob.getMLIRType(expr->getType(), &isArray);
  return CommonArrayLookup(loc, moo, idx, isArray);
}

const clang::FunctionDecl *MLIRScanner::EmitCallee(const Expr *E) {
  E = E->IgnoreParens();
  // Look through function-to-pointer decay.
  if (auto ICE = dyn_cast<ImplicitCastExpr>(E)) {
    if (ICE->getCastKind() == CK_FunctionToPointerDecay ||
        ICE->getCastKind() == CK_BuiltinFnToFnPtr) {
      return EmitCallee(ICE->getSubExpr());
    }

    // Resolve direct calls.
  } else if (auto DRE = dyn_cast<DeclRefExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      return FD;
    }

  } else if (auto ME = dyn_cast<MemberExpr>(E)) {
    if (auto FD = dyn_cast<FunctionDecl>(ME->getMemberDecl())) {
      // TODO EmitIgnoredExpr(ME->getBase());
      return FD;
    }

    // Look through template substitutions.
  } else if (auto NTTP = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) {
    return EmitCallee(NTTP->getReplacement());
  } else if (auto UOp = dyn_cast<clang::UnaryOperator>(E)) {
    if (UOp->getOpcode() == UnaryOperatorKind::UO_AddrOf) {
      return EmitCallee(UOp->getSubExpr());
    }
  }

  return nullptr;
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitBuiltinOps(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__log2f") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::Log2Op>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "log") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::LogOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "ceil")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<math::CeilOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() &&
          (sr->getDecl()->getName() == "expf" ||
           sr->getDecl()->getName() == "exp")) {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::ExpOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "sin") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::SinOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }

      if (sr->getDecl()->getIdentifier() && sr->getDecl()->getName() == "cos") {
        std::vector<mlir::Value> args;
        for (auto a : expr->arguments()) {
          args.push_back(Visit(a).getValue(loc, builder));
        }
        return make_pair(
            ValueCategory(builder.create<mlir::math::CosOp>(loc, args[0]),
                          /*isReference*/ false),
            true);
      }
    }
  }

  return make_pair(ValueCategory(), false);
}

std::pair<ValueCategory, bool>
MLIRScanner::EmitGPUCallExpr(clang::CallExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  if (auto ic = dyn_cast<ImplicitCastExpr>(expr->getCallee())) {
    if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
      if (sr->getDecl()->getIdentifier() &&
          sr->getDecl()->getName() == "__syncthreads") {
        builder.create<mlir::NVVM::Barrier0Op>(loc);
        return make_pair(ValueCategory(), true);
      }
      if (sr->getDecl()->getIdentifier() && CudaLower &&
          sr->getDecl()->getName() == "cudaFuncSetCacheConfig") {
        llvm::errs() << " Not emitting GPU option: cudaFuncSetCacheConfig\n";
        return make_pair(ValueCategory(), true);
      }
      // TODO move free out.
      std::string str;
      llvm::raw_string_ostream ss(str);
      ss.str();
      sr->getDecl()->printQualifiedName(ss);
      if (str == "free" || ((CudaLower && ToCPU.size() > 0) &&
                            (str == "cudaFree" || str == "cudaFreeHost"))) {

        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        mlir::Value arg = Visit(sub).getValue(loc, builder);

        if (isa<mlir::LLVM::LLVMPointerType>(arg.getType())) {
          auto callee = EmitCallee(expr->getCallee());
          auto strcmpF = Glob.GetOrCreateLLVMFunction(callee);
          mlir::Value args[] = {arg};
          builder.create<mlir::LLVM::CallOp>(loc, strcmpF, args);
        } else {
          assert(isa<MemRefType>(arg.getType()));
          builder.create<mlir::memref::DeallocOp>(loc, arg);
        }
        if (sr->getDecl()->getName() == "cudaFree" ||
            sr->getDecl()->getName() == "cudaFreeHost") {
          auto ty = getMLIRType(expr->getType());
          auto op = builder.create<ConstantIntOp>(loc, ty, 0);
          return make_pair(ValueCategory(op, /*isReference*/ false), true);
        }
        // TODO remove me when the free is removed.
        return make_pair(ValueCategory(), true);
      }
      if ((CudaLower && ToCPU.size() > 0) &&
          (str == "cudaMalloc" || str == "cudaMallocHost" ||
           str == "cudaMallocPitch")) {
        auto sub = expr->getArg(0);
        while (auto BC = dyn_cast<clang::CastExpr>(sub))
          sub = BC->getSubExpr();
        {
          auto dst = Visit(sub).getValue(loc, builder);
          if (auto omt = dyn_cast<MemRefType>(dst.getType())) {
            if (auto mt = dyn_cast<MemRefType>(omt.getElementType())) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  loc, cast<clang::PointerType>(
                           cast<clang::PointerType>(
                               sub->getType()->getUnqualifiedDesugaredType())
                               ->getPointeeType())
                           ->getPointeeType());
              mlir::Value allocSize;
              if (str == "cudaMallocPitch") {
                mlir::Value width =
                    Visit(expr->getArg(2)).getValue(loc, builder);
                mlir::Value height =
                    Visit(expr->getArg(3)).getValue(loc, builder);
                // Not changing pitch from provided width here
                // TODO can consider addition alignment considerations
                Visit(expr->getArg(1))
                    .dereference(loc, builder)
                    .store(loc, builder, width);
                allocSize = builder.create<MulIOp>(loc, width, height);
              } else
                allocSize = Visit(expr->getArg(1)).getValue(loc, builder);
              auto idxType = mlir::IndexType::get(builder.getContext());
              mlir::Value args[1] = {builder.create<DivUIOp>(
                  loc, builder.create<IndexCastOp>(loc, idxType, allocSize),
                  elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(
                  loc,
                  (str != "cudaMallocHost" && !CudaLower)
                      ? mlir::MemRefType::get(shape, mt.getElementType(),
                                              MemRefLayoutAttrInterface())
                      : mt,
                  args);
              mlir::Value allocv = alloc;
              allocv = builder.create<polygeist::Memref2PointerOp>(
                  loc, LLVM::LLVMPointerType::get(mt.getContext(), 1),
                  allocv);
              allocv =
                  builder.create<polygeist::Pointer2MemrefOp>(loc, mt, allocv);
              ValueCategory(dst, /*isReference*/ true)
                  .store(loc, builder, allocv);
              auto retTy = getMLIRType(expr->getType());
              return make_pair(
                  ValueCategory(builder.create<ConstantIntOp>(loc, retTy, 0),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }

    auto createBlockIdOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createBlockDimOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::BlockDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createThreadIdOp = [&](gpu::Dimension str,
                                mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::ThreadIdOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    auto createGridDimOp = [&](gpu::Dimension str,
                               mlir::Type mlirType) -> mlir::Value {
      return builder.create<IndexCastOp>(
          loc, mlirType,
          builder.create<mlir::gpu::GridDimOp>(
              loc, mlir::IndexType::get(builder.getContext()), str));
    };

    if (auto ME = dyn_cast<MemberExpr>(ic->getSubExpr())) {
      auto memberName = ME->getMemberDecl()->getName();

      if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
        if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
          if (sr->getDecl()->getName() == "blockIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createBlockIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "blockDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createBlockDimOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "threadIdx") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createThreadIdOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
          if (sr->getDecl()->getName() == "gridDim") {
            auto mlirType = getMLIRType(expr->getType());
            if (memberName == "__fetch_builtin_x") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::x, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_y") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::y, mlirType),
                                /*isReference*/ false),
                  true);
            }
            if (memberName == "__fetch_builtin_z") {
              return make_pair(
                  ValueCategory(createGridDimOp(gpu::Dimension::z, mlirType),
                                /*isReference*/ false),
                  true);
            }
          }
        }
      }
    }
  }
  return make_pair(ValueCategory(), false);
}

mlir::Value MLIRScanner::getConstantIndex(int x) {
  if (constants.find(x) != constants.end()) {
    return constants[x];
  }
  mlir::OpBuilder subbuilder(builder.getContext());
  subbuilder.setInsertionPointToStart(entryBlock);
  return constants[x] =
             subbuilder.create<ConstantIndexOp>(subbuilder.getUnknownLoc(), x);
}

ValueCategory MLIRScanner::VisitMSPropertyRefExpr(MSPropertyRefExpr *expr) {
  llvm_unreachable("unhandled ms propertyref");
  // TODO obviously fake
  return nullptr;
}

ValueCategory
MLIRScanner::VisitPseudoObjectExpr(clang::PseudoObjectExpr *expr) {
  return Visit(expr->getResultExpr());
}

ValueCategory MLIRScanner::VisitUnaryOperator(clang::UnaryOperator *U) {
  auto loc = getMLIRLocation(U->getExprLoc());
  auto sub = Visit(U->getSubExpr());

  switch (U->getOpcode()) {
  case clang::UnaryOperator::Opcode::UO_Extension: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_LNot: {
    assert(sub.val);
    mlir::Value val = sub.getValue(loc, builder);

    if (auto MT = dyn_cast<mlir::MemRefType>(val.getType())) {
      val = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(MT.getContext(), MT.getMemorySpaceAsInt()),
          val);
    }
    auto postTy = cast<mlir::IntegerType>(getMLIRType(U->getType()));

    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      mlir::Value ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::eq, val, nullptr_llvm);
      if (postTy.getWidth() > 1)
        ne = builder.create<arith::ExtUIOp>(loc, postTy, ne);
      return ValueCategory(ne, /*isReference*/ false);
    }

    if (!isa<mlir::IntegerType>(val.getType())) {
      U->dump();
      val.dump();
    }
    auto ty = cast<mlir::IntegerType>(val.getType());
    if (ty.getWidth() != 1) {
      val = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, val,
          builder.create<ConstantIntOp>(loc, ty, 0));
    }
    auto c1 = builder.create<ConstantIntOp>(loc, val.getType(), 1);
    mlir::Value res = builder.create<XOrIOp>(loc, val, c1);

    if (postTy.getWidth() > 1)
      res = builder.create<arith::ExtUIOp>(loc, postTy, res);
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Not: {
    assert(sub.val);
    mlir::Value val = sub.getValue(loc, builder);

    if (!isa<mlir::IntegerType>(val.getType())) {
      U->dump();
      val.dump();
    }
    auto ty = cast<mlir::IntegerType>(val.getType());
    auto c1 = builder.create<ConstantIntOp>(
        loc, ty, APInt::getAllOnes(ty.getWidth()).getSExtValue());
    return ValueCategory(builder.create<XOrIOp>(loc, val, c1),
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Deref: {
    auto dref = sub.dereference(loc, builder);
    return dref;
  }
  case clang::UnaryOperator::Opcode::UO_AddrOf: {
    assert(sub.isReference);
    if (isa<mlir::LLVM::LLVMPointerType>(sub.val.getType())) {
      return ValueCategory(sub.val, /*isReference*/ false);
    }

    bool isArray = false;
    Glob.getMLIRType(U->getSubExpr()->getType(), &isArray);
    auto mt = cast<MemRefType>(sub.val.getType());
    auto shape = std::vector<int64_t>(mt.getShape());
    mlir::Value res;
    shape[0] = ShapedType::kDynamic;
    auto mt0 =
        mlir::MemRefType::get(shape, mt.getElementType(),
                              MemRefLayoutAttrInterface(), mt.getMemorySpace());
    res = builder.create<memref::CastOp>(loc, mt0, sub.val);
    return ValueCategory(res,
                         /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Plus: {
    return sub;
  }
  case clang::UnaryOperator::Opcode::UO_Minus: {
    mlir::Value val = sub.getValue(loc, builder);
    auto ty = val.getType();
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
      if (auto CI = val.getDefiningOp<ConstantFloatOp>()) {
        auto api = cast<FloatAttr>(CI.getValue()).getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::FloatAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(builder.create<NegFOp>(loc, val),
                           /*isReference*/ false);
    } else {
      if (auto CI = val.getDefiningOp<ConstantIntOp>()) {
        auto api = cast<IntegerAttr>(CI.getValue()).getValue();
        return ValueCategory(builder.create<arith::ConstantOp>(
                                 loc, ty, mlir::IntegerAttr::get(ty, -api)),
                             /*isReference*/ false);
      }
      return ValueCategory(
          builder.create<SubIOp>(loc,
                                 builder.create<ConstantIntOp>(
                                     loc, cast<mlir::IntegerType>(ty), 0),
                                 val),
          /*isReference*/ false);
    }
  }
  case clang::UnaryOperator::Opcode::UO_PreInc:
  case clang::UnaryOperator::Opcode::UO_PostInc: {
    assert(sub.isReference);
    auto prev = sub.getValue(loc, builder);
    auto ty = prev.getType();

    mlir::Value next;
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, ft, APFloat(ft.getFloatSemantics(), "1")));
    } else if (auto mt = dyn_cast<MemRefType>(ty)) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(1));
    } else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(ty)) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      auto gepElemTy = getMLIRType(U->getSubExpr()->getType()->getPointeeType());
      next = builder.create<LLVM::GEPOp>(
          loc, pt, gepElemTy, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, ity, 1)}));
    } else {
      if (!isa<mlir::IntegerType>(ty)) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      if (prev.getType() != ty) {
        U->dump();
        llvm::errs() << " ty: " << ty << "prev: " << prev << "\n";
      }
      assert(prev.getType() == ty);
      next = builder.create<AddIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, cast<mlir::IntegerType>(ty), 1));
    }
    sub.store(loc, builder, next);

    if (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc)
      return sub;
    else
      return ValueCategory(prev, /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_PreDec:
  case clang::UnaryOperator::Opcode::UO_PostDec: {
    auto ty = getMLIRType(U->getType());
    assert(sub.isReference);
    auto prev = sub.getValue(loc, builder);

    mlir::Value next;
    if (auto ft = dyn_cast<mlir::FloatType>(ty)) {
      next = builder.create<SubFOp>(
          loc, prev,
          builder.create<ConstantFloatOp>(
              loc, ft, APFloat(ft.getFloatSemantics(), "1")));
    } else if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(ty)) {
      auto ity = mlir::IntegerType::get(builder.getContext(), 64);
      auto gepElemTy = getMLIRType(U->getSubExpr()->getType()->getPointeeType());
      next = builder.create<LLVM::GEPOp>(
          loc, pt, gepElemTy, prev,
          std::vector<mlir::Value>(
              {builder.create<ConstantIntOp>(loc, ity, ShapedType::kDynamic)}));
    } else if (auto mt = dyn_cast<MemRefType>(ty)) {
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      next = builder.create<polygeist::SubIndexOp>(loc, mt0, prev,
                                                   getConstantIndex(-1));
    } else {
      if (!isa<mlir::IntegerType>(ty)) {
        llvm::errs() << ty << " - " << prev << "\n";
        U->dump();
      }
      next = builder.create<SubIOp>(
          loc, prev,
          builder.create<ConstantIntOp>(loc, cast<mlir::IntegerType>(ty), 1));
    }
    sub.store(loc, builder, next);
    return ValueCategory(
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_PostDec) ? prev
                                                                     : next,
        /*isReference*/ false);
  }
  case clang::UnaryOperator::Opcode::UO_Real:
  case clang::UnaryOperator::Opcode::UO_Imag: {
    int fnum =
        (U->getOpcode() == clang::UnaryOperator::Opcode::UO_Real) ? 0 : 1;
    assert(sub.isReference);
    return getComplexPartRef(loc, sub.val, fnum);
  }
  default: {
    U->dump();
    llvm_unreachable("unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::VisitSubstNonTypeTemplateParmExpr(
    SubstNonTypeTemplateParmExpr *expr) {
  return Visit(expr->getReplacement());
}

ValueCategory
MLIRScanner::VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *Uop) {
  auto loc = getMLIRLocation(Uop->getExprLoc());
  switch (Uop->getKind()) {
  case UETT_SizeOf: {
    auto value = getTypeSize(loc, Uop->getTypeOfArgument());
    auto retTy = cast<mlir::IntegerType>(getMLIRType(Uop->getType()));
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  case UETT_AlignOf: {
    auto value = getTypeAlign(loc, Uop->getTypeOfArgument());
    auto retTy = cast<mlir::IntegerType>(getMLIRType(Uop->getType()));
    return ValueCategory(builder.create<arith::IndexCastOp>(loc, retTy, value),
                         /*isReference*/ false);
  }
  default:
    Uop->dump();
    llvm_unreachable("unhandled VisitUnaryExprOrTypeTraitExpr");
  }
}

bool hasAffineArith(Operation *op, AffineExpr &expr,
                    mlir::Value &affineForIndVar) {
  // skip IndexCastOp
  if (isa<IndexCastOp>(op))
    return hasAffineArith(op->getOperand(0).getDefiningOp(), expr,
                          affineForIndVar);

  // induction variable are modelled as memref<1xType>
  // %1 = index_cast %induction : index to i32
  // %2 = alloca() : memref<1xi32>
  // store %1, %2[0] : memref<1xi32>
  // ...
  // %5 = load %2[0] : memref<1xf32>
  if (isa<mlir::memref::LoadOp>(op)) {
    auto load = cast<mlir::memref::LoadOp>(op);
    auto loadOperand = load.getOperand(0);
    if (cast<MemRefType>(loadOperand.getType()).getShape().size() != 1)
      return false;
    auto maybeAllocaOp = loadOperand.getDefiningOp();
    if (!isa<mlir::memref::AllocaOp>(maybeAllocaOp))
      return false;
    auto allocaUsers = maybeAllocaOp->getUsers();
    if (llvm::none_of(allocaUsers, [](mlir::Operation *op) {
          if (isa<mlir::memref::StoreOp>(op))
            return true;
          return false;
        }))
      return false;
    for (auto user : allocaUsers)
      if (auto storeOp = dyn_cast<mlir::memref::StoreOp>(user)) {
        auto storeOperand = storeOp.getOperand(0);
        auto maybeIndexCast = storeOperand.getDefiningOp();
        if (!isa<IndexCastOp>(maybeIndexCast))
          return false;
        auto indexCastOperand = maybeIndexCast->getOperand(0);
        if (auto blockArg = dyn_cast<mlir::BlockArgument>(indexCastOperand)) {
          if (auto affineForOp = dyn_cast<mlir::affine::AffineForOp>(
                  blockArg.getOwner()->getParentOp()))
            affineForIndVar = affineForOp.getInductionVar();
          else
            return false;
        }
      }
    return true;
  }

  // at this point we expect only AddIOp or MulIOp
  if ((!isa<AddIOp>(op)) && (!isa<MulIOp>(op))) {
    return false;
  }

  // make sure that the current op has at least one constant operand
  // (ConstantIndexOp or ConstantIntOp)
  if (llvm::none_of(op->getOperands(), [](mlir::Value operand) {
        return (isa<ConstantIndexOp>(operand.getDefiningOp()) ||
                isa<ConstantIntOp>(operand.getDefiningOp()));
      }))
    return false;

  // build affine expression by adding or multiplying constants.
  // and keep iterating on the non-constant index
  mlir::Value nonCstOperand = nullptr;
  for (auto operand : op->getOperands()) {
    if (auto constantIndexOp =
            dyn_cast<ConstantIndexOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIndexOp.value();
      else
        expr = expr * constantIndexOp.value();
    } else if (auto constantIntOp =
                   dyn_cast<ConstantIntOp>(operand.getDefiningOp())) {
      if (isa<AddIOp>(op))
        expr = expr + constantIntOp.value();
      else
        expr = expr * constantIntOp.value();
    } else
      nonCstOperand = operand;
  }
  return hasAffineArith(nonCstOperand.getDefiningOp(), expr, affineForIndVar);
}

ValueCategory MLIRScanner::VisitAtomicExpr(clang::AtomicExpr *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  switch (BO->getOp()) {
  case AtomicExpr::AtomicOp::AO__atomic_add_fetch: {
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto a1 = Visit(BO->getVal1()).getValue(loc, builder);
    auto ty = a1.getType();
    AtomicRMWKind op;
    LLVM::AtomicBinOp lop;
    if (isa<mlir::IntegerType>(ty)) {
      op = AtomicRMWKind::addi;
      lop = LLVM::AtomicBinOp::add;
    } else {
      op = AtomicRMWKind::addf;
      lop = LLVM::AtomicBinOp::fadd;
    }
    // TODO add atomic ordering
    mlir::Value v;
    if (isa<MemRefType>(a0.getType()))
      v = builder.create<memref::AtomicRMWOp>(
          loc, op, a1, a0, std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      v = builder.create<LLVM::AtomicRMWOp>(loc, lop, a0, a1,
                                            LLVM::AtomicOrdering::acq_rel);

    if (isa<mlir::IntegerType>(ty))
      v = builder.create<arith::AddIOp>(loc, v, a1);
    else
      v = builder.create<arith::AddFOp>(loc, v, a1);

    return ValueCategory(v, false);
  }
  case AtomicExpr::AtomicOp::AO__atomic_load: {
    // In the absence of an atomic load instruction, fall back to += 0.0
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto ret = Visit(BO->getVal1()).dereference(loc, builder);
    mlir::Type ty;
    if (auto MT = dyn_cast<MemRefType>(a0.getType())) {
      ty = MT.getElementType();
    } else {
      ty = getMLIRType(BO->getPtr()->getType()->getPointeeType());
    }
    AtomicRMWKind op;
    LLVM::AtomicBinOp lop;
    mlir::Value a1;
    if (isa<mlir::IntegerType>(ty)) {
      op = AtomicRMWKind::addi;
      lop = LLVM::AtomicBinOp::add;
      a1 = builder.create<ConstantIntOp>(loc, ty, 0);
    } else {
      op = AtomicRMWKind::addf;
      lop = LLVM::AtomicBinOp::fadd;
      a1 = builder.create<ConstantFloatOp>(
          loc, cast<mlir::FloatType>(ty),
          APFloat(cast<mlir::FloatType>(ty).getFloatSemantics(), "0"));
    }
    // TODO add atomic ordering
    mlir::Value v;
    if (isa<MemRefType>(a0.getType()))
      v = builder.create<memref::AtomicRMWOp>(
          loc, op, a1, a0, std::vector<mlir::Value>({getConstantIndex(0)}));
    else
      v = builder.create<LLVM::AtomicRMWOp>(loc, lop, a0, a1,
                                            LLVM::AtomicOrdering::acq_rel);
    ret.store(loc, builder, v);
    return ValueCategory(v, false);
  }
  case AtomicExpr::AtomicOp::AO__atomic_store: {
    llvm::errs() << " TODO: implement atomic store with atomics\n";
    auto a0 = Visit(BO->getPtr()).dereference(loc, builder);
    auto a1 =
        Visit(BO->getVal1()).dereference(loc, builder).getValue(loc, builder);
    a0.store(loc, builder, a1);
    return ValueCategory();
  }
  case AtomicExpr::AtomicOp::AO__atomic_compare_exchange: {
    // In the absence of an atomic load instruction, fall back to += 0.0
    auto a0 = Visit(BO->getPtr()).getValue(loc, builder);
    auto a1 =
        Visit(BO->getVal1()).dereference(loc, builder).getValue(loc, builder);
    auto a2 =
        Visit(BO->getVal2()).dereference(loc, builder).getValue(loc, builder);
    if (auto mt = dyn_cast<MemRefType>(a0.getType())) {
      a0 = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getContext(), mt.getMemorySpaceAsInt()),
          a0);
    }
    // TODO add atomic ordering
    mlir::Value v = builder.create<LLVM::AtomicCmpXchgOp>(
        loc, a0, a1, a2, LLVM::AtomicOrdering::seq_cst,
        LLVM::AtomicOrdering::seq_cst);
    v = builder.create<LLVM::ExtractValueOp>(loc, v, 1);
    auto postTy = cast<mlir::IntegerType>(getMLIRType(BO->getType()));
    if (postTy.getWidth() > 1)
      v = builder.create<arith::ExtUIOp>(loc, postTy, v);
    return ValueCategory(v, false);
  }
  default:
    llvm::errs() << "unhandled atomic:";
    BO->dump();
    assert(0);
  }
}

ValueCategory MLIRScanner::VisitBinaryOperator(clang::BinaryOperator *BO) {
  auto loc = getMLIRLocation(BO->getExprLoc());

  auto fixInteger = [&](mlir::Value res, bool forceUnsigned = false) {
    auto prevTy = cast<mlir::IntegerType>(res.getType());
    auto postTy = cast<mlir::IntegerType>(getMLIRType(BO->getType()));
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (forceUnsigned)
      signedType = false;
    if (postTy != prevTy) {
      if (signedType) {
        res = builder.create<mlir::arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<mlir::arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  };

  auto lhs = Visit(BO->getLHS());
  if (!lhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->dump();
    BO->getLHS()->dump();
    assert(lhs.val);
  }

  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_LAnd: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(cond.getType())) {
      cond = builder.create<polygeist::Memref2PointerOp>(
          loc,
          LLVM::LLVMPointerType::get(mt.getContext(), mt.getMemorySpaceAsInt()),
          cond);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    if (!isa<mlir::IntegerType>(cond.getType())) {
      BO->dump();
      BO->getType()->dump();
      llvm::errs() << "cond: " << cond << "\n";
    }
    auto prevTy = cast<mlir::IntegerType>(cond.getType());
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, prevTy, 0));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    auto rhs = Visit(BO->getRHS()).getValue(loc, builder);
    assert(rhs != nullptr);
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(rhs.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      rhs = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, rhs, nullptr_llvm);
    }
    if (!cast<mlir::IntegerType>(rhs.getType()).isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, rhs.getType(), 0));
    }
    mlir::Value truearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    mlir::Value falsearray[] = {
        builder.create<ConstantIntOp>(loc, types[0], 0)};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);
    return fixInteger(ifOp.getResult(0));
  }
  case clang::BinaryOperator::Opcode::BO_LOr: {
    mlir::Type types[] = {builder.getIntegerType(1)};
    auto cond = lhs.getValue(loc, builder);
    auto prevTy = cast<mlir::IntegerType>(cond.getType());
    if (!prevTy.isInteger(1)) {
      cond = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, cond,
          builder.create<ConstantIntOp>(loc, prevTy, 0));
    }
    auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                /*hasElseRegion*/ true);

    auto oldpoint = builder.getInsertionPoint();
    auto oldblock = builder.getInsertionBlock();
    builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

    mlir::Value truearray[] = {builder.create<ConstantIntOp>(loc, types[0], 1)};
    builder.create<mlir::scf::YieldOp>(loc, truearray);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().back());
    auto rhs = Visit(BO->getRHS()).getValue(loc, builder);
    if (!cast<mlir::IntegerType>(rhs.getType()).isInteger(1)) {
      rhs = builder.create<arith::CmpIOp>(
          loc, CmpIPredicate::ne, rhs,
          builder.create<ConstantIntOp>(loc, rhs.getType(), 0));
    }
    assert(rhs != nullptr);
    mlir::Value falsearray[] = {rhs};
    builder.create<mlir::scf::YieldOp>(loc, falsearray);

    builder.setInsertionPoint(oldblock, oldpoint);

    return fixInteger(ifOp.getResult(0));
  }
  default:
    break;
  }
  auto rhs = Visit(BO->getRHS());
  if (!rhs.val && BO->getOpcode() != clang::BinaryOperator::Opcode::BO_Comma) {
    BO->getRHS()->dump();
    assert(rhs.val);
  }
  // TODO note assumptions made here about unsigned / unordered
  bool signedType = true;
  if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
    if (bit->isUnsignedInteger())
      signedType = false;
    if (bit->isSignedInteger())
      signedType = true;
  }
  switch (BO->getOpcode()) {
  case clang::BinaryOperator::Opcode::BO_Shr: {
    auto lhsv = lhs.getValue(loc, builder);
    auto rhsv = rhs.getValue(loc, builder);
    auto prevTy = cast<mlir::IntegerType>(rhsv.getType());
    auto postTy = cast<mlir::IntegerType>(lhsv.getType());
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<mlir::arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<mlir::arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    if (signedType)
      return ValueCategory(builder.create<ShRSIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
    else
      return ValueCategory(builder.create<ShRUIOp>(loc, lhsv, rhsv),
                           /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Shl: {
    auto lhsv = lhs.getValue(loc, builder);
    auto rhsv = rhs.getValue(loc, builder);
    auto prevTy = cast<mlir::IntegerType>(rhsv.getType());
    auto postTy = cast<mlir::IntegerType>(lhsv.getType());
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());
    return ValueCategory(builder.create<ShLIOp>(loc, lhsv, rhsv),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_And: {
    return ValueCategory(builder.create<AndIOp>(loc, lhs.getValue(loc, builder),
                                                rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Xor: {
    return ValueCategory(builder.create<XOrIOp>(loc, lhs.getValue(loc, builder),
                                                rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
  case clang::BinaryOperator::Opcode::BO_Or: {
    // TODO short circuit
    return ValueCategory(builder.create<OrIOp>(loc, lhs.getValue(loc, builder),
                                               rhs.getValue(loc, builder)),
                         /*isReference*/ false);
  }
    {
      auto lhs_v = lhs.getValue(loc, builder);
      mlir::Value res;
      if (isa<mlir::FloatType>(lhs_v.getType())) {
        res = builder.create<CmpFOp>(loc, CmpFPredicate::UGT, lhs_v,
                                     rhs.getValue(loc, builder));
      } else {
        res = builder.create<CmpIOp>(
            loc, signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt, lhs_v,
            rhs.getValue(loc, builder));
      }
      return fixInteger(res);
    }
  case clang::BinaryOperator::Opcode::BO_GT:
  case clang::BinaryOperator::Opcode::BO_GE:
  case clang::BinaryOperator::Opcode::BO_LT:
  case clang::BinaryOperator::Opcode::BO_LE:
  case clang::BinaryOperator::Opcode::BO_EQ:
  case clang::BinaryOperator::Opcode::BO_NE: {
    signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getLHS()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    CmpFPredicate FPred;
    CmpIPredicate IPred;
    LLVM::ICmpPredicate LPred;
    switch (BO->getOpcode()) {
    case clang::BinaryOperator::Opcode::BO_GT:
      FPred = CmpFPredicate::OGT;
      IPred = signedType ? CmpIPredicate::sgt : CmpIPredicate::ugt,
      LPred = LLVM::ICmpPredicate::ugt;
      break;
    case clang::BinaryOperator::Opcode::BO_GE:
      FPred = CmpFPredicate::OGE;
      IPred = signedType ? CmpIPredicate::sge : CmpIPredicate::uge,
      LPred = LLVM::ICmpPredicate::uge;
      break;
    case clang::BinaryOperator::Opcode::BO_LT:
      FPred = CmpFPredicate::OLT;
      IPred = signedType ? CmpIPredicate::slt : CmpIPredicate::ult,
      LPred = LLVM::ICmpPredicate::ult;
      break;
    case clang::BinaryOperator::Opcode::BO_LE:
      FPred = CmpFPredicate::OLE;
      IPred = signedType ? CmpIPredicate::sle : CmpIPredicate::ule,
      LPred = LLVM::ICmpPredicate::ule;
      break;
    case clang::BinaryOperator::Opcode::BO_EQ:
      FPred = CmpFPredicate::OEQ;
      IPred = CmpIPredicate::eq;
      LPred = LLVM::ICmpPredicate::eq;
      break;
    case clang::BinaryOperator::Opcode::BO_NE:
      FPred = CmpFPredicate::UNE;
      IPred = CmpIPredicate::ne;
      LPred = LLVM::ICmpPredicate::ne;
      break;
    default:
      llvm_unreachable("Unknown op in binary comparision switch");
    }

    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(lhs_v.getType())) {
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getContext()), lhs_v);
    }
    if (auto mt = dyn_cast<mlir::MemRefType>(rhs_v.getType())) {
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getContext()), rhs_v);
    }
    mlir::Value res;
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      res = builder.create<arith::CmpFOp>(loc, FPred, lhs_v, rhs_v);
    } else if (isa<LLVM::LLVMPointerType>(lhs_v.getType())) {
      res = builder.create<LLVM::ICmpOp>(loc, LPred, lhs_v, rhs_v);
    } else {
      res = builder.create<arith::CmpIOp>(loc, IPred, lhs_v, rhs_v);
    }
    return fixInteger(res, /*forceUnsigned*/ true);
  }
  case clang::BinaryOperator::Opcode::BO_Mul: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex mult");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      return ValueCategory(
          builder.create<arith::MulFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
    } else {
      return ValueCategory(
          builder.create<arith::MulIOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Div: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex div");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      return ValueCategory(
          builder.create<arith::DivFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
      ;
    } else {
      if (signedType)
        return ValueCategory(builder.create<arith::DivSIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
      else
        return ValueCategory(builder.create<arith::DivUIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Rem: {
    auto lhs_v = lhs.getValue(loc, builder);
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      return ValueCategory(
          builder.create<arith::RemFOp>(loc, lhs_v, rhs.getValue(loc, builder)),
          /*isReference*/ false);
    } else {
      if (signedType)
        return ValueCategory(builder.create<arith::RemSIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
      else
        return ValueCategory(builder.create<arith::RemUIOp>(
                                 loc, lhs_v, rhs.getValue(loc, builder)),
                             /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Add: {
    auto emitSubindex = [&](auto mr, auto ptradd) {
      auto mt = dyn_cast<mlir::MemRefType>(mr.getType());
      auto shape = std::vector<int64_t>(mt.getShape());
      shape[0] = ShapedType::kDynamic;
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      ptradd = castToIndex(loc, ptradd);
      return ValueCategory(
          builder.create<polygeist::SubIndexOp>(loc, mt0, mr, ptradd),
          /*isReference*/ false);
    };
    if (auto cty = dyn_cast<clang::ComplexType>(BO->getType())) {
      mlir::Value real =
          builder.create<AddFOp>(loc, getComplexPart(loc, lhs.val, 0),
                                 getComplexPart(loc, rhs.val, 0));
      mlir::Value imag =
          builder.create<AddFOp>(loc, getComplexPart(loc, lhs.val, 1),
                                 getComplexPart(loc, rhs.val, 1));
      return createComplexFloat(loc, real, imag, BO->getType());
    }

    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      return ValueCategory(builder.create<AddFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (isa<mlir::MemRefType>(lhs_v.getType())) {
      return emitSubindex(lhs_v, rhs_v);
    } else if (isa<mlir::MemRefType>(rhs_v.getType())) {
      return emitSubindex(rhs_v, lhs_v);
    } else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(lhs_v.getType())) {
      auto gepElemTy = getMLIRType(BO->getLHS()->getType()->getPointeeType());
      return ValueCategory(
          builder.create<LLVM::GEPOp>(loc, pt, gepElemTy, lhs_v,
                                      std::vector<mlir::Value>({rhs_v})),
          /*isReference*/ false);
    } else {
      auto coerceIntegerOperand = [&](mlir::Value val) -> mlir::Value {
        auto resultType = getMLIRType(BO->getType());
        if (isa<mlir::IndexType>(resultType))
          return castToIndex(loc, val);

        auto targetIntTy = dyn_cast<mlir::IntegerType>(resultType);
        if (!targetIntTy)
          return val;

        if (val.getType() == resultType)
          return val;
        if (isa<mlir::IndexType>(val.getType()))
          return builder.create<arith::IndexCastOp>(loc, targetIntTy, val);

        auto fromIntTy = dyn_cast<mlir::IntegerType>(val.getType());
        if (!fromIntTy)
          return val;

        bool signedType = true;
        if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType()))
          signedType = !bit->isUnsignedInteger();

        if (fromIntTy.getWidth() < targetIntTy.getWidth()) {
          if (signedType)
            return builder.create<arith::ExtSIOp>(loc, targetIntTy, val);
          return builder.create<arith::ExtUIOp>(loc, targetIntTy, val);
        }
        if (fromIntTy.getWidth() > targetIntTy.getWidth())
          return builder.create<arith::TruncIOp>(loc, targetIntTy, val);
        return val;
      };

      if (auto lhs_c = lhs_v.getDefiningOp<ConstantIntOp>()) {
        if (auto rhs_c = rhs_v.getDefiningOp<ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantIntOp>(
                  loc, lhs_c.getType(), lhs_c.value() + rhs_c.value()),
              false);
        }
      }
      lhs_v = coerceIntegerOperand(lhs_v);
      rhs_v = coerceIntegerOperand(rhs_v);
      return ValueCategory(builder.create<AddIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Sub: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex sub");
    }
    auto lhs_v = lhs.getValue(loc, builder);
    auto rhs_v = rhs.getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(lhs_v.getType())) {
      mlir::Type innerType = mt.getElementType();
      auto shape = mt.getShape();
      for (size_t i = 1; i < shape.size(); i++)
        innerType = LLVM::LLVMArrayType::get(innerType, shape[i]);
      lhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(innerType.getContext()), lhs_v);
    }
    if (auto mt = dyn_cast<mlir::MemRefType>(rhs_v.getType())) {
      mlir::Type innerType = mt.getElementType();
      auto shape = mt.getShape();
      for (size_t i = 1; i < shape.size(); i++)
        innerType = LLVM::LLVMArrayType::get(innerType, shape[i]);
      rhs_v = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(innerType.getContext()), rhs_v);
    }
    if (isa<mlir::FloatType>(lhs_v.getType())) {
      assert(rhs_v.getType() == lhs_v.getType());
      return ValueCategory(builder.create<SubFOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    } else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(lhs_v.getType())) {
      if (auto IT = dyn_cast<mlir::IntegerType>(rhs_v.getType())) {
        mlir::Value vals[1] = {builder.create<SubIOp>(
            loc, builder.create<ConstantIntOp>(loc, 0, IT.getWidth()), rhs_v)};
        auto gepElemTy = getMLIRType(BO->getLHS()->getType()->getPointeeType());
        return ValueCategory(
            builder.create<LLVM::GEPOp>(loc, lhs_v.getType(), gepElemTy,
                                        lhs_v, ArrayRef<mlir::Value>(vals)),
            false);
      }
      mlir::Value val =
          builder.create<SubIOp>(loc,
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), lhs_v),
                                 builder.create<LLVM::PtrToIntOp>(
                                     loc, getMLIRType(BO->getType()), rhs_v));
      val = builder.create<DivSIOp>(
          loc, val,
          builder.create<IndexCastOp>(
              loc, val.getType(),
              builder.create<polygeist::TypeSizeOp>(
                  loc, builder.getIndexType(),
                  mlir::TypeAttr::get(getMLIRType(
                      BO->getLHS()->getType()->getPointeeType())))));
      return ValueCategory(val, /*isReference*/ false);
    } else {
      auto coerceIntegerOperand = [&](mlir::Value val) -> mlir::Value {
        auto resultType = getMLIRType(BO->getType());
        if (isa<mlir::IndexType>(resultType))
          return castToIndex(loc, val);

        auto targetIntTy = dyn_cast<mlir::IntegerType>(resultType);
        if (!targetIntTy)
          return val;

        if (val.getType() == resultType)
          return val;
        if (isa<mlir::IndexType>(val.getType()))
          return builder.create<arith::IndexCastOp>(loc, targetIntTy, val);

        auto fromIntTy = dyn_cast<mlir::IntegerType>(val.getType());
        if (!fromIntTy)
          return val;

        bool signedType = true;
        if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType()))
          signedType = !bit->isUnsignedInteger();

        if (fromIntTy.getWidth() < targetIntTy.getWidth()) {
          if (signedType)
            return builder.create<arith::ExtSIOp>(loc, targetIntTy, val);
          return builder.create<arith::ExtUIOp>(loc, targetIntTy, val);
        }
        if (fromIntTy.getWidth() > targetIntTy.getWidth())
          return builder.create<arith::TruncIOp>(loc, targetIntTy, val);
        return val;
      };

      lhs_v = coerceIntegerOperand(lhs_v);
      rhs_v = coerceIntegerOperand(rhs_v);
      return ValueCategory(builder.create<SubIOp>(loc, lhs_v, rhs_v),
                           /*isReference*/ false);
    }
  }
  case clang::BinaryOperator::Opcode::BO_Assign: {
    assert(lhs.isReference);
    mlir::Value tostore = rhs.getValue(loc, builder);
    mlir::Type subType;
    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(lhs.val.getType()))
      subType = getMLIRType(BO->getLHS()->getType()->getPointeeType());
    else
      subType = cast<MemRefType>(lhs.val.getType()).getElementType();
    if (tostore.getType() != subType) {
      if (auto prevTy = dyn_cast<mlir::IntegerType>(tostore.getType())) {
        if (auto postTy = dyn_cast<mlir::IntegerType>(subType)) {
          bool signedType = true;
          if (auto bit = dyn_cast<clang::BuiltinType>(&*BO->getType())) {
            if (bit->isUnsignedInteger())
              signedType = false;
            if (bit->isSignedInteger())
              signedType = true;
          }

          if (prevTy.getWidth() < postTy.getWidth()) {
            if (signedType) {
              tostore = builder.create<arith::ExtSIOp>(loc, postTy, tostore);
            } else {
              tostore = builder.create<arith::ExtUIOp>(loc, postTy, tostore);
            }
          } else if (prevTy.getWidth() > postTy.getWidth()) {
            tostore = builder.create<arith::TruncIOp>(loc, postTy, tostore);
          }
        }
      }
    }
    lhs.store(loc, builder, tostore);
    return lhs;
  }

  case clang::BinaryOperator::Opcode::BO_Comma: {
    return rhs;
  }

  case clang::BinaryOperator::Opcode::BO_AddAssign: {
    assert(lhs.isReference);

    mlir::Value result;
    if (isa<clang::ComplexType>(BO->getType())) {
      mlir::Value prev = lhs.val;
      mlir::Value rhsV = rhs.val;
      mlir::Value real = builder.create<AddFOp>(
          loc, getComplexPart(loc, prev, 0), getComplexPart(loc, rhsV, 0));
      mlir::Value imag = builder.create<AddFOp>(
          loc, getComplexPart(loc, prev, 1), getComplexPart(loc, rhsV, 1));
      result = createComplexFloat(loc, real, imag, BO->getType())
                   .getValue(loc, builder);
      lhs.store(loc, builder, result);
      return lhs;
    }
    auto prev = lhs.getValue(loc, builder);
    if (auto postTy = dyn_cast<mlir::FloatType>(prev.getType())) {
      mlir::Value rhsV = rhs.getValue(loc, builder);
      auto prevTy = cast<mlir::FloatType>(rhsV.getType());
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        rhsV = builder.create<mlir::arith::ExtFOp>(loc, postTy, rhsV);
      } else {
        rhsV = builder.create<mlir::arith::TruncFOp>(loc, postTy, rhsV);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddFOp>(loc, prev, rhsV);
    } else if (auto pt =
                   dyn_cast<mlir::LLVM::LLVMPointerType>(prev.getType())) {
      auto gepElemTy = getMLIRType(BO->getLHS()->getType()->getPointeeType());
      result = builder.create<LLVM::GEPOp>(
          loc, pt, gepElemTy, prev,
          std::vector<mlir::Value>({rhs.getValue(loc, builder)}));
    } else if (auto postTy = dyn_cast<mlir::IntegerType>(prev.getType())) {
      mlir::Value rhsV = rhs.getValue(loc, builder);
      auto prevTy = cast<mlir::IntegerType>(rhsV.getType());
      if (prevTy == postTy) {
      } else if (prevTy.getWidth() < postTy.getWidth()) {
        if (signedType) {
          rhsV = builder.create<arith::ExtSIOp>(loc, postTy, rhsV);
        } else {
          rhsV = builder.create<arith::ExtUIOp>(loc, postTy, rhsV);
        }
      } else {
        rhsV = builder.create<arith::TruncIOp>(loc, postTy, rhsV);
      }
      assert(rhsV.getType() == prev.getType());
      result = builder.create<AddIOp>(loc, prev, rhsV);
    } else if (auto postTy = dyn_cast<mlir::MemRefType>(prev.getType())) {
      mlir::Value rhsV = rhs.getValue(loc, builder);
      auto shape = std::vector<int64_t>(postTy.getShape());
      shape[0] = ShapedType::kDynamic;
      postTy = mlir::MemRefType::get(shape, postTy.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     postTy.getMemorySpace());
      auto ptradd = rhsV;
      ptradd = castToIndex(loc, ptradd);
      result = builder.create<polygeist::SubIndexOp>(loc, postTy, prev, ptradd);
    } else {
      assert(false && "Unsupported add assign type");
    }
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_SubAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex sub");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result;
    if (isa<mlir::FloatType>(prev.getType())) {
      auto right = rhs.getValue(loc, builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = cast<mlir::FloatType>(right.getType());
        auto postTy = cast<mlir::FloatType>(getMLIRType(BO->getType()));

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<arith::ExtFOp>(loc, postTy, right);
        } else {
          right = builder.create<arith::TruncFOp>(loc, postTy, right);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<SubFOp>(loc, prev, right);
    } else {
      result = builder.create<SubIOp>(loc, prev, rhs.getValue(loc, builder));
    }
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_MulAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex mult");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result;
    if (isa<mlir::FloatType>(prev.getType())) {
      auto right = rhs.getValue(loc, builder);
      if (right.getType() != prev.getType()) {
        auto prevTy = cast<mlir::FloatType>(right.getType());
        auto postTy = cast<mlir::FloatType>(getMLIRType(BO->getType()));

        if (prevTy.getWidth() < postTy.getWidth()) {
          right = builder.create<arith::ExtFOp>(loc, postTy, right);
        } else {
          right = builder.create<arith::TruncFOp>(loc, postTy, right);
        }
      }
      if (right.getType() != prev.getType()) {
        BO->dump();
        llvm::errs() << " p:" << prev << " r:" << right << "\n";
      }
      assert(right.getType() == prev.getType());
      result = builder.create<MulFOp>(loc, prev, right);
    } else {
      result = builder.create<MulIOp>(loc, prev, rhs.getValue(loc, builder));
    }
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_DivAssign: {
    if (isa<clang::ComplexType>(BO->getType())) {
      llvm_unreachable("Unhandled complex div");
    }
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result;
    if (isa<mlir::FloatType>(prev.getType())) {
      mlir::Value val = rhs.getValue(loc, builder);
      auto prevTy = cast<mlir::FloatType>(val.getType());
      auto postTy = cast<mlir::FloatType>(prev.getType());

      if (prevTy.getWidth() < postTy.getWidth()) {
        val = builder.create<arith::ExtFOp>(loc, postTy, val);
      } else if (prevTy.getWidth() > postTy.getWidth()) {
        val = builder.create<arith::TruncFOp>(loc, postTy, val);
      }
      result = builder.create<arith::DivFOp>(loc, prev, val);
    } else {
      if (signedType)
        result = builder.create<arith::DivSIOp>(loc, prev,
                                                rhs.getValue(loc, builder));
      else
        result = builder.create<arith::DivUIOp>(loc, prev,
                                                rhs.getValue(loc, builder));
    }
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShrAssign: {
    assert(lhs.isReference);
    auto lhsv = lhs.getValue(loc, builder);
    auto rhsv = rhs.getValue(loc, builder);

    mlir::Value result;

    auto prevTy = cast<mlir::IntegerType>(rhsv.getType());
    auto postTy = cast<mlir::IntegerType>(lhsv.getType());
    if (prevTy.getWidth() < postTy.getWidth())
      rhsv = builder.create<mlir::arith::ExtUIOp>(loc, postTy, rhsv);
    if (prevTy.getWidth() > postTy.getWidth())
      rhsv = builder.create<mlir::arith::TruncIOp>(loc, postTy, rhsv);
    assert(lhsv.getType() == rhsv.getType());

    if (signedType)
      result = builder.create<ShRSIOp>(loc, lhsv, rhsv);
    else
      result = builder.create<ShRUIOp>(loc, lhsv, rhsv);
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_ShlAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result =
        builder.create<ShLIOp>(loc, prev, rhs.getValue(loc, builder));
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_RemAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result;

    if (isa<mlir::FloatType>(prev.getType())) {
      result = builder.create<RemFOp>(loc, prev, rhs.getValue(loc, builder));
    } else {
      if (signedType)
        result = builder.create<RemSIOp>(loc, prev, rhs.getValue(loc, builder));
      else
        result = builder.create<RemUIOp>(loc, prev, rhs.getValue(loc, builder));
    }
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_AndAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result =
        builder.create<AndIOp>(loc, prev, rhs.getValue(loc, builder));
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_OrAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result =
        builder.create<OrIOp>(loc, prev, rhs.getValue(loc, builder));
    lhs.store(loc, builder, result);
    return lhs;
  }
  case clang::BinaryOperator::Opcode::BO_XorAssign: {
    assert(lhs.isReference);
    auto prev = lhs.getValue(loc, builder);

    mlir::Value result =
        builder.create<XOrIOp>(loc, prev, rhs.getValue(loc, builder));
    lhs.store(loc, builder, result);
    return lhs;
  }

  default: {
    BO->dump();
    llvm_unreachable("unhandled opcode");
  }
  }
}

ValueCategory MLIRScanner::VisitExprWithCleanups(ExprWithCleanups *E) {
  auto ret = Visit(E->getSubExpr());
  for (auto &child : E->children()) {
    child->dump();
    llvm::errs() << "cleanup not handled\n";
  }
  return ret;
}

ValueCategory MLIRScanner::CommonFieldLookup(mlir::Location loc,
                                             clang::QualType CT,
                                             const FieldDecl *FD,
                                             mlir::Value val, bool isLValue) {
  assert(FD && "Attempting to lookup field of nullptr");
  auto rd = FD->getParent();

  auto ST = cast<llvm::StructType>(getLLVMType(CT));

  size_t fnum = 0;

  auto CXRD = dyn_cast<CXXRecordDecl>(rd);

  if (isLLVMStructABI(rd, ST)) {
    auto &layout = Glob.CGM.getTypes().getCGRecordLayout(rd);
    fnum = layout.getLLVMFieldNo(FD);
  } else {
    fnum = 0;
    if (CXRD)
      fnum += CXRD->getDefinition()->getNumBases();
    for (auto field : rd->fields()) {
      if (field == FD) {
        break;
      }
      fnum++;
    }
  }

  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    auto shape = std::vector<int64_t>(mt.getShape());
    if (shape.size() > 1) {
      shape.erase(shape.begin());
      auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      shape[0] = ShapedType::kDynamic;
      auto mt1 = mlir::MemRefType::get(shape, mt.getElementType(),
                                       MemRefLayoutAttrInterface(),
                                       mt.getMemorySpace());
      mlir::Value sub0 = builder.create<polygeist::SubIndexOp>(
          loc, mt0, val, getConstantIndex(0));
      mlir::Value sub1 = builder.create<polygeist::SubIndexOp>(
          loc, mt1, sub0, getConstantIndex(fnum));
      if (isLValue)
        sub1 = ValueCategory(sub1, /*isReference*/ true).getValue(loc, builder);
      return ValueCategory(sub1, /*isReference*/ true);
    }
    auto ET = mt.getElementType();
    assert(isa<mlir::LLVM::LLVMStructType>(ET) ||
           isa<mlir::LLVM::LLVMArrayType>(ET));
    val = builder.create<polygeist::Memref2PointerOp>(
        loc, LLVM::LLVMPointerType::get(ET.getContext(), mt.getMemorySpaceAsInt()), val);
  }

  auto PT = cast<mlir::LLVM::LLVMPointerType>(val.getType());
  mlir::Value vec[] = {builder.create<ConstantIntOp>(loc, 0, 32),
                       builder.create<ConstantIntOp>(loc, fnum, 32)};
  // With opaque pointers, get the pointee type from the clang QualType.
  auto srcElemType = Glob.getMLIRType(CT);
  if (!isa<mlir::LLVM::LLVMStructType, mlir::LLVM::LLVMArrayType>(srcElemType)) {
    llvm::errs() << "function: " << function << "\n";
    // rd->dump();
    FD->dump();
    FD->getType()->dump();
    llvm::errs() << " val: " << val << " - pt: " << PT << " fn: " << fnum
                 << " ST: " << *ST << "\n";
  }
  mlir::Type ET;
  if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(srcElemType)) {
    ET = ST.getBody()[fnum];
  } else {
    ET = cast<mlir::LLVM::LLVMArrayType>(srcElemType).getElementType();
  }
  mlir::Value commonGEP = builder.create<mlir::LLVM::GEPOp>(
      loc, mlir::LLVM::LLVMPointerType::get(val.getContext(), PT.getAddressSpace()),
      srcElemType, val, vec);

  bool isArray = false;
  auto subType = Glob.getMLIRType(
      Glob.CGM.getContext().getPointerType(FD->getType()), &isArray);
  assert(!isArray);
  if (rd->isUnion()) {
    if (auto PT2 = dyn_cast<LLVM::LLVMPointerType>(subType)) {
      // With opaque pointers, ptr-to-ptr bitcast is identity.
      (void)PT2;
    } else {
      auto mt = cast<mlir::MemRefType>(subType);
      commonGEP = builder.create<polygeist::Pointer2MemrefOp>(
          loc,
          mlir::MemRefType::get(
              mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
              wrapIntegerMemorySpace(PT.getAddressSpace(), mt.getContext())),
          commonGEP);
    }
  } else {
    if (auto mt = dyn_cast<mlir::MemRefType>(subType)) {
      commonGEP = builder.create<polygeist::Pointer2MemrefOp>(
          loc,
          mlir::MemRefType::get(
              mt.getShape(), mt.getElementType(), MemRefLayoutAttrInterface(),
              wrapIntegerMemorySpace(PT.getAddressSpace(), mt.getContext())),
          commonGEP);
    }
  }
  if (isLValue)
    commonGEP =
        ValueCategory(commonGEP, /*isReference*/ true).getValue(loc, builder);
  return ValueCategory(commonGEP, /*isReference*/ true);
}

ValueCategory MLIRScanner::VisitDeclRefExpr(DeclRefExpr *E) {
  auto loc = getMLIRLocation(E->getLocation());
  auto name = E->getDecl()->getName().str();

  if (auto tocall = dyn_cast<FunctionDecl>(E->getDecl())) {
    auto f = Glob.GetOrCreateMLIRFunction(tocall);
    auto FT = f.getFunctionType();
    mlir::Type RT = LLVM::LLVMVoidType::get(f.getContext());
    if (FT.getNumResults() != 0)
      RT = FT.getResult(0);
    LLVM::LLVMFunctionType LFT = LLVM::LLVMFunctionType::get(
        RT, FT.getInputs(), /*unsupported presentlyFT.isVariadic()*/ false);

    return ValueCategory(builder.create<polygeist::GetFuncOp>(
                             loc, LLVM::LLVMPointerType::get(LFT.getContext()), f.getName()),
                         /*isReference*/ true);
  }

  if (auto VD = dyn_cast<VarDecl>(E->getDecl())) {
    if (Captures.find(VD) != Captures.end()) {
      FieldDecl *field = Captures[VD];
      auto res = CommonFieldLookup(
          loc, cast<CXXMethodDecl>(EmittingFunctionDecl)->getFunctionObjectParameterType(),
          field, ThisVal.val,
          isa<clang::ReferenceType>(
              field->getType()->getUnqualifiedDesugaredType()));
      assert(CaptureKinds.find(VD) != CaptureKinds.end());
      return res;
    }
  }

  if (auto PD = dyn_cast<VarDecl>(E->getDecl())) {
    auto found = params.find(PD);
    if (found != params.end()) {
      auto res = found->second;
      assert(res.val);
      return res;
    }
  }
  if (auto ED = dyn_cast<EnumConstantDecl>(E->getDecl())) {
    auto ty = cast<mlir::IntegerType>(getMLIRType(E->getType()));
    return ValueCategory(
        builder.create<ConstantIntOp>(loc, ty, ED->getInitVal().getExtValue()),
        /*isReference*/ false);

    if (!ED->getInitExpr())
      ED->dump();
    return Visit(ED->getInitExpr());
  }
  if (auto VD = dyn_cast<ValueDecl>(E->getDecl())) {
    if (isa<mlir::LLVM::LLVMPointerType>(Glob.getMLIRType(Glob.CGM.getContext().getPointerType(E->getType()))) ||
        name == "stderr" || name == "stdout" || name == "stdin" ||
        name == "__stderrp" || name == "__stdoutp" || name == "__stdinp" ||
        (E->hasQualifier())) {
      return ValueCategory(builder.create<mlir::LLVM::AddressOfOp>(
                               loc, Glob.GetOrCreateLLVMGlobal(VD)),
                           /*isReference*/ true);
    }

    auto gv = Glob.GetOrCreateGlobal(VD, /*prefix=*/"");

    auto mt = gv.first.getType();
    auto gv2 = builder.create<memref::GetGlobalOp>(loc, mt, gv.first.getName());
    auto shape = std::vector<int64_t>(mt.getShape());
    shape[0] = ShapedType::kDynamic;
    auto val = builder.create<memref::CastOp>(
        loc,
        MemRefType::get(shape, mt.getElementType(), MemRefLayoutAttrInterface(),
                        mt.getMemorySpace()),
        gv2);
    bool isArray = gv.second;
    // TODO check reference
    if (isArray)
      return ValueCategory(val, /*isReference*/ true);
    else
      return ValueCategory(val, /*isReference*/ true);
    // return gv2;
  }
  E->dump();
  E->getDecl()->dump();
  llvm::errs() << "couldn't find " << name << "\n";
  llvm_unreachable("couldnt find value");
  return nullptr;
}

ValueCategory MLIRScanner::VisitOpaqueValueExpr(OpaqueValueExpr *E) {
  if (!E->getSourceExpr()) {
    E->dump();
    assert(E->getSourceExpr());
  }
  auto res = Visit(E->getSourceExpr());
  if (!res.val) {
    E->dump();
    E->getSourceExpr()->dump();
    assert(res.val);
  }
  return res;
}

ValueCategory MLIRScanner::VisitCXXTypeidExpr(clang::CXXTypeidExpr *E) {
  QualType T;
  if (E->isTypeOperand())
    T = E->getTypeOperand(Glob.CGM.getContext());
  else
    T = E->getExprOperand()->getType();
  llvm::Constant *C = Glob.CGM.GetAddrOfRTTIDescriptor(T);
  llvm::errs() << *C << "\n";
  auto ty = getMLIRType(E->getType());
  llvm::errs() << ty << "\n";
  llvm_unreachable("unhandled typeid");
}

ValueCategory
MLIRScanner::VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr *expr) {
  auto loc = getMLIRLocation(expr->getExprLoc());
  assert(ThisVal.val);
  auto toset = Visit(expr->getExpr());
  assert(!ThisVal.isReference);
  assert(toset.val);

  bool isArray = false;
  Glob.getMLIRType(expr->getExpr()->getType(), &isArray);

  auto cfl = CommonFieldLookup(
      loc, cast<CXXMethodDecl>(EmittingFunctionDecl)->getFunctionObjectParameterType(),
      expr->getField(), ThisVal.val, /*isLValue*/ false);
  assert(cfl.val);
  cfl.store(loc, builder, toset, isArray);
  return cfl;
}

ValueCategory MLIRScanner::VisitCXXNoexceptExpr(CXXNoexceptExpr *expr) {
  auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(
      builder.create<ConstantIntOp>(getMLIRLocation(expr->getExprLoc()),
                                    ty, expr->getValue()),
      /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitMemberExpr(MemberExpr *ME) {
  auto loc = getMLIRLocation(ME->getExprLoc());
  auto memberName = ME->getMemberDecl()->getName();
  if (auto sr2 = dyn_cast<OpaqueValueExpr>(ME->getBase())) {
    if (auto sr = dyn_cast<DeclRefExpr>(sr2->getSourceExpr())) {
      if (sr->getDecl()->getName() == "blockIdx") {
        if (memberName == "__fetch_builtin_x") {
        }
        llvm::errs() << "known block index";
      }
      if (sr->getDecl()->getName() == "blockDim") {
        llvm::errs() << "known block dim";
      }
      if (sr->getDecl()->getName() == "threadIdx") {
        llvm::errs() << "known thread index";
      }
      if (sr->getDecl()->getName() == "gridDim") {
        llvm::errs() << "known grid index";
      }
    }
  }
  auto base = Visit(ME->getBase());
  clang::QualType OT = ME->getBase()->getType();
  if (ME->isArrow()) {
    if (!base.val) {
      ME->dump();
    }
    base = base.dereference(loc, builder);
    OT = cast<clang::PointerType>(OT->getUnqualifiedDesugaredType())
             ->getPointeeType();
  }
  if (!base.isReference) {
    EmittingFunctionDecl->dump();
    function.dump();
    ME->dump();
    llvm::errs() << "base value: " << base.val << "\n";
  }
  assert(base.isReference);
  const FieldDecl *field = cast<FieldDecl>(ME->getMemberDecl());
  return CommonFieldLookup(
      loc, OT, field, base.val,
      isa<clang::ReferenceType>(
          field->getType()->getUnqualifiedDesugaredType()));
}

mlir::Value MLIRScanner::GetAddressOfDerivedClass(
    mlir::Location loc, mlir::Value value, const CXXRecordDecl *DerivedClass,
    CastExpr::path_const_iterator Start, CastExpr::path_const_iterator End) {
  const ASTContext &Context = Glob.CGM.getContext();

  SmallVector<const CXXRecordDecl *> ToBase = {DerivedClass};
  SmallVector<const CXXBaseSpecifier *> Bases;
  for (auto I = Start; I != End; I++) {
    const CXXBaseSpecifier *Base = *I;
    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    ToBase.push_back(BaseDecl);
    Bases.push_back(Base);
  }

  for (int i = ToBase.size() - 1; i > 0; i--) {
    const CXXBaseSpecifier *Base = Bases[i - 1];

    const auto *BaseDecl =
        cast<CXXRecordDecl>(Base->getType()->castAs<RecordType>()->getDecl());
    const auto *RD = ToBase[i - 1];
    // Get the layout.
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    assert(!Base->isVirtual() && "Should not see virtual bases here!");

    // Add the offset.

    mlir::Type nt = getMLIRType(Glob.CGM.getContext().getLValueReferenceType(
        Glob.CGM.getContext().getCanonicalTagType(RD)));

    mlir::Value Offset = nullptr;
    if (isLLVMStructABI(RD, /*ST*/ nullptr)) {
      Offset = builder.create<arith::ConstantIntOp>(
          loc, -(ssize_t)Layout.getBaseClassOffset(BaseDecl).getQuantity(), 32);
    } else {
      Offset = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            Base->getType()->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        bool subType = false;
        mlir::Type nt = Glob.getMLIRType(f.getType(), &subType, false);
        Offset = builder.create<arith::SubIOp>(
            loc, Offset,
            builder.create<IndexCastOp>(
                loc, Offset.getType(),
                builder.create<polygeist::TypeSizeOp>(
                    loc, builder.getIndexType(), mlir::TypeAttr::get(nt))));
      }
      assert(found);
    }

    mlir::Value ptr = value;
    if (isa<LLVM::LLVMPointerType>(ptr.getType())) {
      // With opaque pointers, ptr is already the right type
    } else {
      unsigned memorySpace =
          cast<mlir::MemRefType>(ptr.getType()).getMemorySpaceAsInt();
      ptr = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext(), memorySpace),
          ptr);
    }

    mlir::Value idx[] = {Offset};
    ptr = builder.create<LLVM::GEPOp>(loc, ptr.getType(), builder.getI8Type(),
                                       ptr, idx);

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(nt))
      value = ptr; // opaque pointers: ptr-to-ptr bitcast is identity
    else
      value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, ptr);
  }

  return value;
}

mlir::Value MLIRScanner::GetAddressOfBaseClass(
    mlir::Location loc, mlir::Value value, const CXXRecordDecl *DerivedClass,
    ArrayRef<const clang::Type *> BaseTypes, ArrayRef<bool> BaseVirtuals) {
  const CXXRecordDecl *RD = DerivedClass;

  for (auto tup : llvm::zip(BaseTypes, BaseVirtuals)) {

    auto BaseType = std::get<0>(tup);

    const auto *BaseDecl =
        cast<CXXRecordDecl>(BaseType->castAs<RecordType>()->getDecl());
    // Add the offset.

    mlir::Type nt = getMLIRType(
        Glob.CGM.getContext().getPointerType(QualType(BaseType, 0)));

    size_t fnum;
    bool subIndex = true;

    if (isLLVMStructABI(RD, /*ST*/ nullptr)) {
      auto &layout = Glob.CGM.getTypes().getCGRecordLayout(RD);
      if (std::get<1>(tup))
        fnum = layout.getVirtualBaseIndex(BaseDecl);
      else {
        if (!layout.hasNonVirtualBaseLLVMField(BaseDecl)) {
          subIndex = false;
        } else {
          fnum = layout.getNonVirtualBaseLLVMFieldNo(BaseDecl);
        }
      }
    } else {
      assert(!std::get<1>(tup) && "Should not see virtual bases here!");
      fnum = 0;
      bool found = false;
      for (auto f : RD->bases()) {
        if (f.getType().getTypePtr()->getUnqualifiedDesugaredType() ==
            BaseType->getUnqualifiedDesugaredType()) {
          found = true;
          break;
        }
        fnum++;
      }
      assert(found);
    }

    if (subIndex) {
      bool done = false;
      if (auto mt = dyn_cast<MemRefType>(value.getType())) {
        auto shape = std::vector<int64_t>(mt.getShape());
        if (shape.size() > 1) {
          shape.erase(shape.begin());
          auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                           MemRefLayoutAttrInterface(),
                                           mt.getMemorySpace());
          value = builder.create<polygeist::SubIndexOp>(loc, mt0, value,
                                                        getConstantIndex(fnum));
          done = true;
        } else {
          auto ET = mt.getElementType();
          assert(isa<mlir::LLVM::LLVMStructType>(ET) ||
                 isa<mlir::LLVM::LLVMArrayType>(ET));
          value = builder.create<polygeist::Memref2PointerOp>(
              loc, LLVM::LLVMPointerType::get(ET.getContext(), mt.getMemorySpaceAsInt()),
              value);
        }
      }
      if (!done) {
        mlir::Value idx[] = {
            builder.create<arith::ConstantIntOp>(loc, 0, 32),
            builder.create<arith::ConstantIntOp>(loc, fnum, 32)};
        auto PT = cast<LLVM::LLVMPointerType>(value.getType());
        // With opaque pointers, get pointee type from the RD clang type.
        auto srcElemType = Glob.getMLIRType(
            Glob.CGM.getContext().getCanonicalTagType(RD));
        mlir::Type ET;
        if (auto ST =
                dyn_cast<mlir::LLVM::LLVMStructType>(srcElemType)) {
          ET = ST.getBody()[fnum];
        } else {
          ET = cast<mlir::LLVM::LLVMArrayType>(srcElemType)
                   .getElementType();
        }

        value = builder.create<LLVM::GEPOp>(
            loc, LLVM::LLVMPointerType::get(value.getContext(), PT.getAddressSpace()),
            srcElemType, value, idx);
      }
    }

    auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(nt);
    if (auto opt = dyn_cast<mlir::LLVM::LLVMPointerType>(value.getType())) {
      if (!pt) {
        value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
      } else {
        if (opt.getAddressSpace() != pt.getAddressSpace())
          value = builder.create<mlir::LLVM::AddrSpaceCastOp>(loc, pt, value);
      }
    } else {
      assert(isa<MemRefType>(value.getType()));
      if (pt) {
        value = builder.create<polygeist::Memref2PointerOp>(loc, pt, value);
      } else {
        if (value.getType() != nt) {
          auto anyPT = LLVM::LLVMPointerType::get(builder.getContext());
          value =
              builder.create<polygeist::Memref2PointerOp>(loc, anyPT, value);
          value = builder.create<polygeist::Pointer2MemrefOp>(loc, nt, value);
        }
      }
    }

    RD = BaseDecl;
  }

  return value;
}

ValueCategory MLIRScanner::VisitCastExpr(CastExpr *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  switch (E->getCastKind()) {

  case clang::CastKind::CK_NullToPointer: {
    auto llvmType = getMLIRType(E->getType());
    if (isa<LLVM::LLVMPointerType>(llvmType))
      return ValueCategory(builder.create<mlir::LLVM::ZeroOp>(loc, llvmType),
                           /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<polygeist::Pointer2MemrefOp>(
              loc, llvmType,
              builder.create<mlir::LLVM::ZeroOp>(
                  loc, LLVM::LLVMPointerType::get(builder.getContext()))),
          false);
  }
  case clang::CastKind::CK_UserDefinedConversion: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_Dynamic: {
    E->dump();
    llvm_unreachable("dynamic cast not handled yet\n");
  }
  case clang::CastKind::CK_UncheckedDerivedToBase:
  case clang::CastKind::CK_DerivedToBase: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(
                  E->getSubExpr()->getType()->castAs<RecordType>()->getDecl())
            : E->getSubExpr()->getType()->getPointeeCXXRecordDecl();
    SmallVector<const clang::Type *> BaseTypes;
    SmallVector<bool> BaseVirtual;
    for (auto B : E->path()) {
      BaseTypes.push_back(B->getType().getTypePtr());
      BaseVirtual.push_back(B->isVirtual());
    }

    mlir::Value val =
        GetAddressOfBaseClass(loc, se.val, Derived, BaseTypes, BaseVirtual);
    if (E->getCastKind() != clang::CastKind::CK_UncheckedDerivedToBase &&
        !isa<CXXThisExpr>(E->IgnoreParens())) {
      mlir::Value ptr = val;
      if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
        ptr = builder.create<polygeist::Memref2PointerOp>(
            loc, LLVM::LLVMPointerType::get(MT.getContext()), ptr);
      mlir::Value nullptr_llvm =
          builder.create<mlir::LLVM::ZeroOp>(loc, ptr.getType());
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, ptr, nullptr_llvm);
      if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
        nullptr_llvm =
            builder.create<polygeist::Pointer2MemrefOp>(loc, MT, nullptr_llvm);
      val = builder.create<arith::SelectOp>(loc, ne, ptr, nullptr_llvm);
    }
    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_BaseToDerived: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    assert(se.val);
    auto Derived =
        (E->isLValue() || E->isXValue())
            ? cast<CXXRecordDecl>(E->getType()->castAs<RecordType>()->getDecl())
            : E->getType()->getPointeeCXXRecordDecl();
    mlir::Value val = GetAddressOfDerivedClass(loc, se.val, Derived,
                                               E->path_begin(), E->path_end());
    /*
    if (ShouldNullCheckClassCastValue(E)) {
        mlir::Value ptr = val;
        if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
            ptr = builder.create<polygeist::Memref2PointerOp>(loc,
    LLVM::LLVMPointerType::get(MT.getContext()), ptr); auto nullptr_llvm =
    builder.create<mlir::LLVM::ZeroOp>(loc, ptr.getType()); auto ne =
    builder.create<mlir::LLVM::ICmpOp>( loc, mlir::LLVM::ICmpPredicate::ne, ptr,
    nullptr_llvm); if (auto MT = dyn_cast<MemRefType>(ptr.getType()))
           nullptr_llvm = builder.create<polygeist::Pointer2MemrefOp>(loc, MT,
    nullptr_llvm); val = builder.create<arith::SelectOp>(loc, ne, val,
    nullptr_llvm);
    }
    */
    return ValueCategory(val, se.isReference);
  }
  case clang::CastKind::CK_LValueBitCast: {
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.val;
    assert(se.isReference);
    if (auto spt = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nt = getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = dyn_cast<LLVM::LLVMPointerType>(nt);
      if (!pt) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, scalar),
            false);
      }
      if (spt.getAddressSpace() != pt.getAddressSpace()) {
        pt = LLVM::LLVMPointerType::get(pt.getContext(), spt.getAddressSpace());
        scalar = builder.create<mlir::LLVM::AddrSpaceCastOp>(loc, pt, scalar);
      }
      return ValueCategory(scalar, /*isReference*/ true);
    }
    if (!isa<mlir::MemRefType>(scalar.getType())) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto ut = cast<mlir::MemRefType>(scalar.getType());
    auto mlirty =
        getMLIRType(Glob.CGM.getContext().getLValueReferenceType(E->getType()));

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(mlirty)) {
      return ValueCategory(
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar),
          /*isReference*/ true);
    } else if (auto mt = dyn_cast<mlir::MemRefType>(mlirty)) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ut.getShape().size() == mt.getShape().size() + 1 &&
          ut.getElementType() == mt.getElementType()) {
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ true);
      }
      bool badShape = ut.getShape().size() != mt.getShape().size();
      if (!badShape)
        for (size_t i = 1; i < ut.getShape().size(); i++) {
          if (ut.getShape()[i] != mt.getShape()[i]) {
            badShape = true;
            break;
          }
        }
      if (badShape || ut.getElementType() != mt.getElementType()) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(
                loc, ty,
                builder.create<polygeist::Memref2PointerOp>(
                    loc, LLVM::LLVMPointerType::get(builder.getContext()),
                    scalar)),
            /*isReference*/ true);
      }
      return ValueCategory(builder.create<memref::CastOp>(loc, ty, scalar),
                           /*isReference*/ true);
    } else {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
      llvm_unreachable("illegal type for cast");
      llvm_unreachable("illegal type for cast");
    }
  }
  case clang::CastKind::CK_BitCast: {

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              sr->getDecl()->getName() == "polybench_alloc_data") {
            if (auto mt =
                    dyn_cast<mlir::MemRefType>(getMLIRType(E->getType()))) {
              auto shape = std::vector<int64_t>(mt.getShape());
              // shape.erase(shape.begin());
              auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                               MemRefLayoutAttrInterface(),
                                               mt.getMemorySpace());

              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt0);
              return ValueCategory(alloc, /*isReference*/ false);
            }
          }
        }

    if (auto CI = dyn_cast<clang::CallExpr>(E->getSubExpr()))
      if (auto ic = dyn_cast<ImplicitCastExpr>(CI->getCallee()))
        if (auto sr = dyn_cast<DeclRefExpr>(ic->getSubExpr())) {
          if (sr->getDecl()->getIdentifier() &&
              (sr->getDecl()->getName() == "malloc" ||
               sr->getDecl()->getName() == "calloc"))
            if (auto mt =
                    dyn_cast<mlir::MemRefType>(getMLIRType(E->getType()))) {
              auto shape = std::vector<int64_t>(mt.getShape());

              auto elemSize = getTypeSize(
                  loc, cast<clang::PointerType>(
                           E->getType()->getUnqualifiedDesugaredType())
                           ->getPointeeType());
              mlir::Value allocSize = builder.create<IndexCastOp>(
                  loc, mlir::IndexType::get(builder.getContext()),
                  Visit(CI->getArg(0)).getValue(loc, builder));
              if (sr->getDecl()->getName() == "calloc") {
                allocSize = builder.create<MulIOp>(
                    loc, allocSize,
                    builder.create<arith::IndexCastOp>(
                        loc, mlir::IndexType::get(builder.getContext()),
                        Visit(CI->getArg(1)).getValue(loc, builder)));
              }
              mlir::Value args[1] = {
                  builder.create<DivUIOp>(loc, allocSize, elemSize)};
              auto alloc = builder.create<mlir::memref::AllocOp>(loc, mt, args);
              if (sr->getDecl()->getName() == "calloc") {
                mlir::Value val = alloc;
                if (isa<MemRefType>(val.getType())) {
                  val = builder.create<polygeist::Memref2PointerOp>(
                      loc, LLVM::LLVMPointerType::get(builder.getContext()),
                      val);
                } else {
                  // With opaque pointers, ptr-to-ptr is identity (same addr space)
                }
                auto i8_0 = builder.create<arith::ConstantIntOp>(loc, 0, 8);
                auto sizev = builder.create<arith::IndexCastOp>(
                    loc, builder.getI64Type(), allocSize);
                builder.create<LLVM::MemsetOp>(loc, val, i8_0, sizev,
                                                /*isVolatile=*/false);
              }
              return ValueCategory(alloc, /*isReference*/ false);
            }
        }
    auto se = Visit(E->getSubExpr());
    if (!se.val) {
      E->dump();
    }
    auto scalar = se.getValue(loc, builder);
    if (auto spt = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nt = getMLIRType(E->getType());
      LLVM::LLVMPointerType pt = dyn_cast<LLVM::LLVMPointerType>(nt);
      if (!pt) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(loc, nt, scalar),
            false);
      }
      if (spt.getAddressSpace() != pt.getAddressSpace()) {
        pt = LLVM::LLVMPointerType::get(pt.getContext(), spt.getAddressSpace());
        scalar = builder.create<mlir::LLVM::AddrSpaceCastOp>(loc, pt, scalar);
      }
      return ValueCategory(scalar, /*isReference*/ false);
    }
    if (!isa<mlir::MemRefType>(scalar.getType())) {
      E->dump();
      E->getType()->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }
    auto ut = cast<mlir::MemRefType>(scalar.getType());
    auto mlirty = getMLIRType(E->getType());

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(mlirty)) {
      return ValueCategory(
          builder.create<mlir::polygeist::Memref2PointerOp>(loc, PT, scalar),
          /*isReference*/ false);
    } else if (auto mt = dyn_cast<mlir::MemRefType>(mlirty)) {
      auto ty = mlir::MemRefType::get(mt.getShape(), mt.getElementType(),
                                      MemRefLayoutAttrInterface(),
                                      ut.getMemorySpace());
      if (ut.getShape().size() == mt.getShape().size() + 1 &&
          ut.getElementType() == mt.getElementType()) {
        return ValueCategory(builder.create<mlir::polygeist::SubIndexOp>(
                                 loc, ty, scalar, getConstantIndex(0)),
                             /*isReference*/ false);
      }
      bool badShape = ut.getShape().size() != mt.getShape().size();
      if (!badShape)
        for (size_t i = 1; i < ut.getShape().size(); i++) {
          if (ut.getShape()[i] != mt.getShape()[i]) {
            badShape = true;
            break;
          }
        }
      if (badShape || ut.getElementType() != mt.getElementType()) {
        return ValueCategory(
            builder.create<polygeist::Pointer2MemrefOp>(
                loc, ty,
                builder.create<polygeist::Memref2PointerOp>(
                    loc, LLVM::LLVMPointerType::get(builder.getContext()),
                    scalar)),
            /*isReference*/ false);
      }
      return ValueCategory(builder.create<memref::CastOp>(loc, ty, scalar),
                           /*isReference*/ false);
    } else {
      E->dump();
      E->getType()->dump();
      llvm::errs() << " scalar: " << scalar << " mlirty: " << mlirty << "\n";
      llvm_unreachable("illegal type for cast");
      llvm_unreachable("illegal type for cast");
    }
  }
  case clang::CastKind::CK_LValueToRValue: {
    if (auto dr = dyn_cast<DeclRefExpr>(E->getSubExpr())) {
      if (auto VD = dyn_cast<VarDecl>(dr->getDecl()->getCanonicalDecl())) {
        if (NOUR_Constant == dr->isNonOdrUse()) {
          auto VarD = cast<VarDecl>(VD);
          if (!VarD->getInit()) {
            E->dump();
            VarD->dump();
          }
          assert(VarD->getInit());
          return Visit(VarD->getInit());
        }
      }
      if (dr->getDecl()->getIdentifier() &&
          dr->getDecl()->getName() == "warpSize") {
        auto mlirType = getMLIRType(E->getType());
        return ValueCategory(
            builder.create<mlir::NVVM::WarpSizeOp>(loc, mlirType),
            /*isReference*/ false);
      }
      /*
      if (dr->isNonOdrUseReason() == clang::NonOdrUseReason::NOUR_Constant) {
        dr->dump();
        auto VD = cast<VarDecl>(dr->getDecl());
        assert(VD->getInit());
      }
      */
    }
    auto prev = Visit(E->getSubExpr());

    bool isArray = false;
    Glob.getMLIRType(E->getType(), &isArray);
    if (isArray)
      return prev;

    auto lres = prev.getValue(loc, builder);
    if (!prev.isReference) {
      E->dump();
      lres.dump();
    }
    assert(prev.isReference);
    return ValueCategory(lres, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToFloating: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    auto ty = cast<mlir::FloatType>(getMLIRType(E->getType()));
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::SIToFPOp>(loc, ty, scalar),
          /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<mlir::arith::UIToFPOp>(loc, ty, scalar),
          /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    auto ty = cast<mlir::IntegerType>(getMLIRType(E->getType()));
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (signedType)
      return ValueCategory(
          builder.create<mlir::arith::FPToSIOp>(loc, ty, scalar),
          /*isReference*/ false);
    else
      return ValueCategory(
          builder.create<mlir::arith::FPToUIOp>(loc, ty, scalar),
          /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    assert(scalar);
    auto postTy = cast<mlir::IntegerType>(getMLIRType(E->getType()));
    if (isa<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      return ValueCategory(
          builder.create<mlir::LLVM::PtrToIntOp>(loc, postTy, scalar),
          /*isReference*/ false);
    }
    if (isa<mlir::IndexType>(scalar.getType()) ||
        isa<mlir::IndexType>(postTy)) {
      return ValueCategory(builder.create<IndexCastOp>(loc, postTy, scalar),
                           false);
    }
    if (!isa<mlir::IntegerType>(scalar.getType())) {
      E->dump();
      llvm::errs() << " scalar: " << scalar << "\n";
    }
    auto prevTy = cast<mlir::IntegerType>(scalar.getType());
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getSubExpr()->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }

    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (prevTy.getWidth() < postTy.getWidth()) {
      if (signedType) {
        if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, cast<IntegerAttr>(CI.getValue()).getValue().sext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtSIOp>(loc, postTy, scalar),
            /*isReference*/ false);
      } else {
        if (auto CI = scalar.getDefiningOp<arith::ConstantIntOp>()) {
          return ValueCategory(
              builder.create<arith::ConstantOp>(
                  loc, postTy,
                  mlir::IntegerAttr::get(
                      postTy, cast<IntegerAttr>(CI.getValue()).getValue().zext(
                                  postTy.getWidth()))),
              /*isReference*/ false);
        }
        return ValueCategory(
            builder.create<arith::ExtUIOp>(loc, postTy, scalar),
            /*isReference*/ false);
      }
    } else {
      if (auto CI = scalar.getDefiningOp<ConstantIntOp>()) {
        return ValueCategory(
            builder.create<arith::ConstantOp>(
                loc, postTy,
                mlir::IntegerAttr::get(
                    postTy, cast<IntegerAttr>(CI.getValue()).getValue().trunc(
                                postTy.getWidth()))),
            /*isReference*/ false);
      }
      return ValueCategory(builder.create<arith::TruncIOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_FloatingCast: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    if (!isa<mlir::FloatType>(scalar.getType())) {
      E->dump();
      llvm::errs() << "scalar: " << scalar << "\n";
    }

    auto prevTy = cast<mlir::FloatType>(scalar.getType());
    auto postTy = cast<mlir::FloatType>(getMLIRType(E->getType()));
    if (prevTy == postTy)
      return ValueCategory(scalar, /*isReference*/ false);
    if (auto c = scalar.getDefiningOp<ConstantFloatOp>()) {
      APFloat Val = cast<FloatAttr>(c.getValue()).getValue();
      bool ignored;
      Val.convert(postTy.getFloatSemantics(), APFloat::rmNearestTiesToEven,
                  &ignored);
      return ValueCategory(builder.create<arith::ConstantOp>(
                               loc, postTy, mlir::FloatAttr::get(postTy, Val)),
                           false);
    }
    if (prevTy.getWidth() < postTy.getWidth()) {
      return ValueCategory(builder.create<arith::ExtFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    } else {
      return ValueCategory(builder.create<arith::TruncFOp>(loc, postTy, scalar),
                           /*isReference*/ false);
    }
  }
  case clang::CastKind::CK_FloatingComplexCast: {
    auto sub = Visit(E->getSubExpr());
    auto complex = sub.val;
    mlir::FloatType prevScalarTy;
    mlir::FloatType postScalarTy;
    auto prevTy = complex.getType();
    auto postTy = getMLIRType(E->getType());
    if (auto mt = dyn_cast<MemRefType>(postTy)) {
      postScalarTy = cast<mlir::FloatType>(mt.getElementType());
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(postTy)) {
      postScalarTy = cast<mlir::FloatType>(ST.getBody()[0]);
    } else {
      llvm_unreachable("unexpected complex type\n");
    }
    if (auto mt = dyn_cast<MemRefType>(prevTy)) {
      prevScalarTy = cast<mlir::FloatType>(mt.getElementType());
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(prevTy)) {
      prevScalarTy = cast<mlir::FloatType>(ST.getBody()[0]);
    } else {
      llvm_unreachable("unexpected complex type\n");
    }
    auto castScalar = [&](mlir::Value complex, int fnum) -> mlir::Value {
      mlir::Value scalar = getComplexPart(loc, complex, fnum);
      assert(scalar);
      if (prevScalarTy.getWidth() < postScalarTy.getWidth()) {
        return builder.create<arith::ExtFOp>(loc, postScalarTy, scalar);
      } else {
        return builder.create<arith::TruncFOp>(loc, postScalarTy, scalar);
      }
    };
    auto real = castScalar(complex, 0);
    auto imag = castScalar(complex, 1);
    return createComplexFloat(loc, real, imag, E->getType());
  }
  case clang::CastKind::CK_ArrayToPointerDecay: {
    return CommonArrayToPointer(loc, Visit(E->getSubExpr()));

#if 0
    auto mt = cast<mlir::MemRefType>(scalar.val.getType());
    auto shape2 = std::vector<int64_t>(mt.getShape());
    if (shape2.size() == 0) {
      E->dump();
      //nex.dump();
      assert(0);
    }
    shape2[0] = ShapedType::kDynamic;
    auto nex = mlir::MemRefType::get(shape2, mt.getElementType(),
                                     mt.getLayout(), mt.getMemorySpace());
    auto cst = builder.create<mlir::MemRefCastOp>(loc, scalar.val, nex);
    //llvm::errs() << "<ArrayToPtrDecay>\n";
    //E->dump();
    //llvm::errs() << cst << " - " << scalar.val << "\n";
    //auto offs = scalar.offsets;
    //offs.push_back(getConstantIndex(0));
    return ValueCategory(cst, scalar.isReference);
#endif
  }
  case clang::CastKind::CK_FunctionToPointerDecay: {
    auto scalar = Visit(E->getSubExpr());
    assert(scalar.isReference);
    return ValueCategory(scalar.val, /*isReference*/ false);
  }
  case clang::CastKind::CK_ConstructorConversion:
  case clang::CastKind::CK_NoOp: {
    return Visit(E->getSubExpr());
  }
  case clang::CastKind::CK_ToVoid: {
    Visit(E->getSubExpr());
    return nullptr;
  }
  case clang::CastKind::CK_PointerToBoolean: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(scalar.getType())) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getContext()), scalar);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
      auto ne = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, scalar, nullptr_llvm);
      return ValueCategory(ne, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    llvm_unreachable("unhandled ptrtobool cast");
  }
  case clang::CastKind::CK_PointerToIntegral: {
    auto scalar = Visit(E->getSubExpr()).getValue(loc, builder);
    if (auto mt = dyn_cast<mlir::MemRefType>(scalar.getType())) {
      scalar = builder.create<polygeist::Memref2PointerOp>(
          loc, LLVM::LLVMPointerType::get(mt.getContext()), scalar);
    }
    if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(scalar.getType())) {
      auto mlirType = getMLIRType(E->getType());
      auto val = builder.create<mlir::LLVM::PtrToIntOp>(loc, mlirType, scalar);
      return ValueCategory(val, /*isReference*/ false);
    }
    function.dump();
    llvm::errs() << "scalar: " << scalar << "\n";
    E->dump();
    llvm_unreachable("unhandled ptrtoint cast");
  }
  case clang::CastKind::CK_IntegralToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(loc, builder);
    auto prevTy = cast<mlir::IntegerType>(res.getType());
    res = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, res,
        builder.create<ConstantIntOp>(loc, prevTy, 0));
    auto postTy = cast<mlir::IntegerType>(getMLIRType(E->getType()));
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    if (postTy.getWidth() > 1) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingToBoolean: {
    auto res = Visit(E->getSubExpr()).getValue(loc, builder);
    auto prevTy = cast<mlir::FloatType>(res.getType());
    auto postTy = cast<mlir::IntegerType>(getMLIRType(E->getType()));
    bool signedType = true;
    if (auto bit = dyn_cast<clang::BuiltinType>(&*E->getType())) {
      if (bit->isUnsignedInteger())
        signedType = false;
      if (bit->isSignedInteger())
        signedType = true;
    }
    auto Zero = builder.create<ConstantFloatOp>(
        loc, prevTy, APFloat::getZero(prevTy.getFloatSemantics()));
    res = builder.create<arith::CmpFOp>(loc, CmpFPredicate::UNE, res, Zero);
    if (1 < postTy.getWidth()) {
      if (signedType) {
        res = builder.create<arith::ExtSIOp>(loc, postTy, res);
      } else {
        res = builder.create<arith::ExtUIOp>(loc, postTy, res);
      }
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_IntegralToPointer: {
    auto vc = Visit(E->getSubExpr());
    if (!vc.val) {
      E->dump();
    }
    assert(vc.val);
    auto res = vc.getValue(loc, builder);
    auto postTy = getMLIRType(E->getType());
    if (isa<LLVM::LLVMPointerType>(postTy))
      res = builder.create<LLVM::IntToPtrOp>(loc, postTy, res);
    else {
      assert(isa<MemRefType>(postTy));
      res = builder.create<LLVM::IntToPtrOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()), res);
      res = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, res);
    }
    return ValueCategory(res, /*isReference*/ false);
  }
  case clang::CastKind::CK_FloatingRealToComplex: {
    auto sub = Visit(E->getSubExpr());
    auto complex = sub.val;
    auto convertedType = getMLIRType(E->getType());
    auto real = sub.getValue(loc, builder);
    mlir::FloatType fty;
    if (auto mt = dyn_cast<MemRefType>(convertedType)) {
      auto elty = mt.getElementType();
      if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(elty)) {
        fty = cast<FloatType>(ST.getBody()[0]);
      } else {
        fty = cast<FloatType>(elty);
      }
    } else if (auto ST = dyn_cast<mlir::LLVM::LLVMStructType>(convertedType)) {
      fty = cast<FloatType>(ST.getBody()[0]);
    } else {
      llvm_unreachable("unexpected complex type");
    }
    auto zero = builder.create<ConstantFloatOp>(
        loc, fty, APFloat(fty.getFloatSemantics(), "0"));
    return createComplexFloat(loc, real, zero, E->getType());
  }

  default:
    if (EmittingFunctionDecl)
      EmittingFunctionDecl->dump();
    E->dump();
    llvm_unreachable("unhandled cast");
  }
}

ValueCategory
MLIRScanner::VisitConditionalOperator(clang::ConditionalOperator *E) {
  auto loc = getMLIRLocation(E->getExprLoc());
  auto cond = Visit(E->getCond()).getValue(loc, builder);
  assert(cond != nullptr);
  if (auto mt = dyn_cast<mlir::MemRefType>(cond.getType())) {
    cond = builder.create<polygeist::Memref2PointerOp>(
        loc,
        LLVM::LLVMPointerType::get(mt.getContext(), mt.getMemorySpaceAsInt()),
        cond);
  }
  if (auto LT = dyn_cast<mlir::LLVM::LLVMPointerType>(cond.getType())) {
    auto nullptr_llvm = builder.create<mlir::LLVM::ZeroOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cast<mlir::IntegerType>(cond.getType());
  if (!prevTy.isInteger(1)) {
    cond = builder.create<arith::CmpIOp>(
        loc, CmpIPredicate::ne, cond,
        builder.create<ConstantIntOp>(loc, prevTy, 0));
  }
  std::vector<mlir::Type> types;
  if (!E->getType()->isVoidType())
    types.push_back(getMLIRType(E->getType()));
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                              /*hasElseRegion*/ true);

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  builder.setInsertionPointToStart(&ifOp.getThenRegion().back());

  auto trueExpr = Visit(E->getTrueExpr());

  builder.setInsertionPointToStart(&ifOp.getElseRegion().back());

  auto falseExpr = Visit(E->getFalseExpr());

  bool isReference = E->isLValue() || E->isXValue() ||
                     (trueExpr.isReference && falseExpr.isReference);

  builder.setInsertionPointToEnd(&ifOp.getThenRegion().back());

  std::vector<mlir::Value> truearray;
  if (!E->getType()->isVoidType()) {
    if (!trueExpr.val) {
      E->dump();
    }
    assert(trueExpr.val);
    mlir::Value truev;
    if (isReference) {
      assert(trueExpr.isReference);
      truev = trueExpr.val;
    } else {
      if (trueExpr.isReference)
        if (auto mt = dyn_cast<MemRefType>(trueExpr.val.getType()))
          if (mt.getShape().size() != 1) {
            E->dump();
            E->getTrueExpr()->dump();
            llvm::errs() << " trueExpr: " << trueExpr.val << "\n";
            assert(0);
          }
      truev = trueExpr.getValue(loc, builder);
    }
    assert(truev != nullptr);
    truearray.push_back(truev);
    builder.create<mlir::scf::YieldOp>(loc, truearray);
  }

  builder.setInsertionPointToEnd(&ifOp.getElseRegion().back());

  std::vector<mlir::Value> falsearray;
  if (!E->getType()->isVoidType()) {
    mlir::Value falsev;
    if (isReference) {
      assert(falseExpr.isReference);
      falsev = falseExpr.val;
    } else
      falsev = falseExpr.getValue(loc, builder);
    assert(falsev != nullptr);
    falsearray.push_back(falsev);
    builder.create<mlir::scf::YieldOp>(loc, falsearray);
  }

  builder.setInsertionPoint(oldblock, oldpoint);

  for (size_t i = 0; i < truearray.size(); i++)
    types[i] = truearray[i].getType();
  auto newIfOp = builder.create<mlir::scf::IfOp>(loc, types, cond,
                                                 /*hasElseRegion*/ true);
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
  ifOp.erase();
  if (types.size() == 0)
    return ValueCategory();
  return ValueCategory(newIfOp.getResult(0), /*isReference*/ isReference);
}

ValueCategory MLIRScanner::VisitSizeOfPackExpr(SizeOfPackExpr *expr) {
  const auto loc = getMLIRLocation(expr->getExprLoc());
  const auto val = expr->getPackLength();
  const auto ty = cast<mlir::IntegerType>(getMLIRType(expr->getType()));
  return ValueCategory(builder.create<arith::ConstantIntOp>(loc, ty, val),
                       /*isReference*/ false);
}

ValueCategory MLIRScanner::VisitStmtExpr(clang::StmtExpr *stmt) {
  ValueCategory off = nullptr;
  for (auto a : stmt->getSubStmt()->children()) {
    off = Visit(a);
  }
  return off;
}

mlir::Value MLIRASTConsumer::CallMalloc(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value arg) {
  mlir::OpBuilder fbuilder(module->getContext());
  fbuilder.setInsertionPointToStart(module->getBody());
  std::string name = "malloc";
  auto ctx = module->getContext();
  mlir::Type types[] = {mlir::IntegerType::get(ctx, 64)};
  if (CStyleMemRef) {
    if (functions.find(name) == functions.end()) {
      auto funcType = fbuilder.getFunctionType(
          types,
          mlir::MemRefType::get({ShapedType::kDynamic}, builder.getI8Type()));
      functions[name] =
          fbuilder.create<mlir::func::FuncOp>(module->getLoc(), name, funcType);
    }
    std::vector args = {arg};
    auto callOp =
        builder.create<mlir::func::CallOp>(loc, functions[name], args);

    return callOp->getResult(0);
  } else {

    if (llvmFunctions.find(name) == llvmFunctions.end()) {
      auto llvmFnType = LLVM::LLVMFunctionType::get(
          LLVM::LLVMPointerType::get(builder.getContext()), types, false);
      LLVM::Linkage lnk = LLVM::Linkage::External;
      llvmFunctions[name] = fbuilder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), name, llvmFnType, lnk);
    }
    auto i64 = mlir::IntegerType::get(ctx, 64);
    arg = builder.create<IndexCastOp>(loc, i64, arg);
    std::vector args = {arg};
    auto callOp =
        builder.create<mlir::LLVM::CallOp>(loc, llvmFunctions[name], args);
    return callOp->getResult(0);
  }
}
mlir::LLVM::LLVMFuncOp MLIRASTConsumer::GetOrCreateFreeFunction() {
  std::string name = "free";
  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }
  auto ctx = module->getContext();
  mlir::Type types[] = {LLVM::LLVMPointerType::get(ctx)};
  auto llvmFnType =
      LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), types, false);

  LLVM::Linkage lnk = LLVM::Linkage::External;
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::LLVMFuncOp
MLIRASTConsumer::GetOrCreateLLVMFunction(const FunctionDecl *FD) {
  std::string name;
  if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
  else
    name = CGM.getMangledName(FD).str();

  if (name != "malloc" && name != "free")
    name = (PrefixABI + name);

  if (llvmFunctions.find(name) != llvmFunctions.end()) {
    return llvmFunctions[name];
  }

  std::vector<mlir::Type> types;
  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(CC->getThisType()))));
  }
  for (auto parm : FD->parameters()) {
    types.push_back(typeTranslator.translateType(
        anonymize(getLLVMType(parm->getOriginalType()))));
  }

  auto rt =
      typeTranslator.translateType(anonymize(getLLVMType(FD->getReturnType())));

  auto llvmFnType = LLVM::LLVMFunctionType::get(rt, types,
                                                /*isVarArg=*/FD->isVariadic());

  LLVM::Linkage lnk;
  switch (CGM.getFunctionLinkage(FD)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    // Available Externally not supported in MLIR LLVM Dialect
    // lnk = LLVM::Linkage::AvailableExternally;
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }
  // Insert the function into the body of the parent module.
  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());
  return llvmFunctions[name] = builder.create<LLVM::LLVMFuncOp>(
             module->getLoc(), name, llvmFnType, lnk);
}

mlir::LLVM::GlobalOp
MLIRASTConsumer::GetOrCreateLLVMGlobal(const ValueDecl *FD,
                                       std::string prefix) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (llvmGlobals.find(name) != llvmGlobals.end()) {
    return llvmGlobals[name];
  }

  LLVM::Linkage lnk;
  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  auto linkage = CGM.getLLVMLinkageVarDefinition(VD);
  switch (linkage) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = LLVM::Linkage::AvailableExternally;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }

  auto rt = getMLIRType(FD->getType());

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto glob = builder.create<LLVM::GlobalOp>(
      module->getLoc(), rt, /*constant*/ false, lnk, name, mlir::Attribute());

  if (VD->getInit() ||
      VD->isThisDeclarationADefinition() == VarDecl::Definition ||
      VD->isThisDeclarationADefinition() == VarDecl::TentativeDefinition) {
    Block *blk = new Block();
    builder.setInsertionPointToStart(blk);
    mlir::Value res;
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      ms.setEntryAndAllocBlock(blk);
      res = ms.Visit(const_cast<Expr *>(init))
                .getValue(getMLIRLocation(init->getBeginLoc()), builder);
    } else {
      res = builder.create<polygeist::UndefOp>(module->getLoc(), rt);
    }
    bool legal = true;
    for (Operation &op : *blk) {
      auto iface = dyn_cast<MemoryEffectOpInterface>(op);
      if (!iface || !iface.hasNoEffect()) {
        legal = false;
        break;
      }
    }
    if (legal) {
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({res}));
      glob.getInitializerRegion().push_back(blk);
    } else {
      Block *blk2 = new Block();
      builder.setInsertionPointToEnd(blk2);
      mlir::Value nres =
          builder.create<polygeist::UndefOp>(module->getLoc(), rt);
      builder.create<LLVM::ReturnOp>(module->getLoc(),
                                     std::vector<mlir::Value>({nres}));
      glob.getInitializerRegion().push_back(blk2);

      builder.setInsertionPointToStart(module->getBody());
      auto funcName = name + "@init";
      LLVM::GlobalCtorsOp ctors = nullptr;
      for (auto &op : *module->getBody()) {
        if (auto c = dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
          ctors = c;
        }
      }
      SmallVector<mlir::Attribute> funcs;
      funcs.push_back(FlatSymbolRefAttr::get(module->getContext(), funcName));
      SmallVector<mlir::Attribute> idxs;
      idxs.push_back(builder.getI32IntegerAttr(0));
      if (ctors) {
        for (auto f : ctors.getCtors())
          funcs.push_back(f);
        for (auto v : ctors.getPriorities())
          idxs.push_back(v);
        ctors->erase();
      }

      builder.create<LLVM::GlobalCtorsOp>(module->getLoc(),
                                          builder.getArrayAttr(funcs),
                                          builder.getArrayAttr(idxs),
                                          builder.getArrayAttr({}));

      auto llvmFnType = LLVM::LLVMFunctionType::get(
          mlir::LLVM::LLVMVoidType::get(module->getContext()),
          ArrayRef<mlir::Type>(), false);

      auto func = builder.create<LLVM::LLVMFuncOp>(
          module->getLoc(), funcName, llvmFnType, LLVM::Linkage::Private);
      func.getRegion().push_back(blk);
      builder.setInsertionPointToEnd(blk);
      builder.create<LLVM::StoreOp>(
          module->getLoc(), res,
          builder.create<LLVM::AddressOfOp>(module->getLoc(), glob));
      builder.create<LLVM::ReturnOp>(module->getLoc(), ArrayRef<mlir::Value>());
    }
  }
  if (lnk == LLVM::Linkage::Private || lnk == LLVM::Linkage::Internal) {
    SymbolTable::setSymbolVisibility(glob,
                                     mlir::SymbolTable::Visibility::Private);
  }
  return llvmGlobals[name] = glob;
}

std::pair<mlir::memref::GlobalOp, bool>
MLIRASTConsumer::GetOrCreateGlobal(const ValueDecl *FD, std::string prefix,
                                   bool tryInit) {
  std::string name = prefix + CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  if (globals.find(name) != globals.end()) {
    return globals[name];
  }

  bool isArray = false;
  auto rt = getMLIRType(FD->getType(), &isArray);
  unsigned memspace = 0;

  mlir::MemRefType mr =
      cast<MemRefType>(getMLIRType(CGM.getContext().getLValueReferenceType(FD->getType())));
  std::vector<int64_t> shape(mr.getShape());
  if (shape[0] == ShapedType::kDynamic)
    shape[0] = 1;
  mr = mlir::MemRefType::get(shape, mr.getElementType(),
                             MemRefLayoutAttrInterface(),
                             wrapIntegerMemorySpace(memspace, mr.getContext()));

  mlir::SymbolTable::Visibility lnk;
  mlir::Attribute initial_value;

  mlir::OpBuilder builder(module->getContext());
  builder.setInsertionPointToStart(module->getBody());

  auto VD = dyn_cast<VarDecl>(FD);
  if (!VD)
    FD->dump();
  VD = VD->getCanonicalDecl();

  if (VD->isThisDeclarationADefinition() == VarDecl::Definition) {
    initial_value = builder.getUnitAttr();
  } else if (VD->isThisDeclarationADefinition() ==
             VarDecl::TentativeDefinition) {
    initial_value = builder.getUnitAttr();
  }

  switch (CGM.getLLVMLinkageVarDefinition(VD)) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = mlir::SymbolTable::Visibility::Private;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = mlir::SymbolTable::Visibility::Public;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = mlir::SymbolTable::Visibility::Private;
    break;
  }

  auto globalOp = builder.create<mlir::memref::GlobalOp>(
      module->getLoc(), builder.getStringAttr(name),
      /*sym_visibility*/ mlir::StringAttr(), mlir::TypeAttr::get(mr),
      initial_value, mlir::UnitAttr(), /*alignment*/ nullptr);
  SymbolTable::setSymbolVisibility(globalOp, lnk);

  // TODO we should probably set the mem spaces here? But I am not sure that
  // works well with our one-module representation because the host-side "stubs"
  // have the default 0 memespace and it is only the device side globals that
  // should have the gpu specific memspace
  if (VD->hasAttr<CUDAConstantAttr>()) {
    globalOp->setAttr("polygeist.cuda_constant", builder.getUnitAttr());
  } else if (VD->hasAttr<CUDADeviceAttr>()) {
    globalOp->setAttr("polygeist.cuda_device", builder.getUnitAttr());
  }

  globals[name] = std::make_pair(globalOp, isArray);

  if (tryInit)
    if (auto init = VD->getInit()) {
      MLIRScanner ms(*this, module, LTInfo);
      mlir::Block *B = new Block();
      ms.setEntryAndAllocBlock(B);
      OpBuilder builder(module->getContext());
      builder.setInsertionPointToEnd(B);
      auto op = builder.create<memref::AllocaOp>(module->getLoc(), mr);

      bool initialized = false;
      if (isa<InitListExpr>(init)) {
        if (auto A = ms.InitializeValueByInitListExpr(
                op, const_cast<clang::Expr *>(init))) {
          initialized = true;
          initial_value = A;
        }
      } else {
        auto VC = ms.Visit(const_cast<clang::Expr *>(init));
        if (!VC.isReference) {
          if (auto cop = VC.val.getDefiningOp<arith::ConstantOp>()) {
            initial_value = cop.getValue();
            initial_value = SplatElementsAttr::get(
                RankedTensorType::get(mr.getShape(), mr.getElementType()),
                initial_value);
            initialized = true;
          }
        }
      }

      if (!initialized) {
        FD->dump();
        init->dump();
        llvm::errs() << " warning not initializing global: " << name << "\n";
      } else {
        globalOp.setInitialValueAttr(initial_value);
      }
      delete B;
    }

  return globals[name];
}

mlir::Value MLIRASTConsumer::GetOrCreateGlobalLLVMString(
    mlir::Location loc, mlir::OpBuilder &builder, StringRef value) {
  using namespace mlir;
  // Create the global at the entry of the module.
  if (llvmStringGlobals.find(value.str()) == llvmStringGlobals.end()) {
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module->getBody());
    auto type = LLVM::LLVMArrayType::get(
        mlir::IntegerType::get(builder.getContext(), 8), value.size() + 1);
    llvmStringGlobals[value.str()] = builder.create<LLVM::GlobalOp>(
        loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
        "str" + std::to_string(llvmStringGlobals.size()),
        builder.getStringAttr(value.str() + '\0'));
  }

  LLVM::GlobalOp global = llvmStringGlobals[value.str()];
  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  return globalPtr;
}

mlir::func::FuncOp
MLIRASTConsumer::GetOrCreateMLIRFunction(const FunctionDecl *FD,
                                         bool getDeviceStub) {
  assert(FD->getTemplatedKind() !=
         FunctionDecl::TemplatedKind::TK_FunctionTemplate);
  assert(
      FD->getTemplatedKind() !=
      FunctionDecl::TemplatedKind::TK_DependentFunctionTemplateSpecialization);
  std::string name;
  if (getDeviceStub)
    name =
        CGM.getMangledName(GlobalDecl(FD, KernelReferenceKind::Kernel)).str();
  else if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    name = CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
  else
    name = CGM.getMangledName(FD).str();

  name = (PrefixABI + name);

  assert(name != "free");

  llvm::GlobalValue::LinkageTypes LV;
  if (!FD->hasBody())
    LV = llvm::GlobalValue::LinkageTypes::ExternalLinkage;
  else if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
    LV = CGM.getFunctionLinkage(GlobalDecl(CC, CXXCtorType::Ctor_Complete));
  else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
    LV = CGM.getFunctionLinkage(GlobalDecl(CC, CXXDtorType::Dtor_Complete));
  else
    LV = CGM.getFunctionLinkage(FD);

  LLVM::Linkage lnk;
  switch (LV) {
  case llvm::GlobalValue::LinkageTypes::InternalLinkage:
    lnk = LLVM::Linkage::Internal;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalLinkage:
    lnk = LLVM::Linkage::External;
    break;
  case llvm::GlobalValue::LinkageTypes::AvailableExternallyLinkage:
    lnk = LLVM::Linkage::AvailableExternally;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceAnyLinkage:
    lnk = LLVM::Linkage::Linkonce;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakAnyLinkage:
    lnk = LLVM::Linkage::Weak;
    break;
  case llvm::GlobalValue::LinkageTypes::WeakODRLinkage:
    lnk = LLVM::Linkage::WeakODR;
    break;
  case llvm::GlobalValue::LinkageTypes::CommonLinkage:
    lnk = LLVM::Linkage::Common;
    break;
  case llvm::GlobalValue::LinkageTypes::AppendingLinkage:
    lnk = LLVM::Linkage::Appending;
    break;
  case llvm::GlobalValue::LinkageTypes::ExternalWeakLinkage:
    lnk = LLVM::Linkage::ExternWeak;
    break;
  case llvm::GlobalValue::LinkageTypes::LinkOnceODRLinkage:
    lnk = LLVM::Linkage::LinkonceODR;
    break;
  case llvm::GlobalValue::LinkageTypes::PrivateLinkage:
    lnk = LLVM::Linkage::Private;
    break;
  }

  const FunctionDecl *Def = nullptr;
  if (!FD->isDefined(Def, /*checkforfriend*/ true))
    Def = FD;

  if (functions.find(name) != functions.end()) {
    auto function = functions[name];

    if (Def->isThisDeclarationADefinition()) {
      if (LV == llvm::GlobalValue::InternalLinkage ||
          LV == llvm::GlobalValue::PrivateLinkage || !Def->isDefined() ||
          Def->hasAttr<CUDAGlobalAttr>() || Def->hasAttr<CUDADeviceAttr>()) {
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Private);
      } else {
        SymbolTable::setSymbolVisibility(function,
                                         SymbolTable::Visibility::Public);
      }
      mlir::OpBuilder builder(module->getContext());
      NamedAttrList attrs(function->getAttrDictionary());
      attrs.set("llvm.linkage",
                mlir::LLVM::LinkageAttr::get(builder.getContext(), lnk));
      function->setAttrs(attrs.getDictionary(builder.getContext()));
      functionsToEmit.push_back(Def);
    }
    assert(function->getParentOp() == module.get());
    return function;
  }

  std::vector<mlir::Type> types;
  std::vector<std::string> names;

  if (auto CC = dyn_cast<CXXMethodDecl>(FD)) {
    if (CC->isInstance()) {
      auto t = getMLIRType(CC->getThisType());

      bool isArray = false; // isa<clang::ArrayType>(CC->getThisType());
      getMLIRType(CC->getFunctionObjectParameterType(), &isArray);
      if (auto mt = dyn_cast<MemRefType>(t)) {
        auto shape = std::vector<int64_t>(mt.getShape());
        // shape[0] = 1;
        t = mlir::MemRefType::get(shape, mt.getElementType(),
                                  MemRefLayoutAttrInterface(),
                                  mt.getMemorySpace());
      }
      if (!isa<LLVM::LLVMPointerType, MemRefType>(t)) {
        FD->dump();
        CC->getThisType()->dump();
        llvm::errs() << " t: " << t << " isArray: " << (int)isArray
                     << " LLTy: " << *getLLVMType(CC->getThisType())
                     << " mlirty: " << getMLIRType(CC->getThisType()) << "\n";
      }
      assert(((bool)isa<LLVM::LLVMPointerType, MemRefType>(t)));
      types.push_back(t);
      names.push_back("this");
    }
  }
  for (auto parm : FD->parameters()) {
    bool llvmType = name == "main" && types.size() == 1;
    if (auto ava = parm->getAttr<AlignValueAttr>()) {
      if (auto algn = dyn_cast<clang::ConstantExpr>(ava->getAlignment())) {
        for (auto a : algn->children()) {
          if (auto IL = dyn_cast<IntegerLiteral>(a)) {
            if (IL->getValue() == 8192) {
              llvmType = true;
              break;
            }
          }
        }
      }
    }
    if (llvmType && !CStyleMemRef) {
      types.push_back(typeTranslator.translateType(
          anonymize(getLLVMType(parm->getType()))));
    } else {
      bool ArrayStruct = false;
      auto t = getMLIRType(parm->getType(), &ArrayStruct);
      if (ArrayStruct) {
        t = getMLIRType(
            CGM.getContext().getLValueReferenceType(parm->getType()));
      }

      types.push_back(t);
    }
    names.push_back(parm->getName().str());
  }

  bool isArrayReturn = false;
  getMLIRType(FD->getReturnType(), &isArrayReturn);

  std::vector<mlir::Type> rettypes;

  if (isArrayReturn) {
    auto mt = cast<MemRefType>(getMLIRType(
                  CGM.getContext().getLValueReferenceType(FD->getReturnType())));

    auto shape = std::vector<int64_t>(mt.getShape());
    assert(shape.size() == 2);

    types.push_back(mt);
  } else {
    auto rt = getMLIRType(FD->getReturnType());
    if (!isa<mlir::NoneType>(rt)) {
      rettypes.push_back(rt);
    }
  }
  mlir::OpBuilder builder(module->getContext());
  auto funcType = builder.getFunctionType(types, rettypes);

  mlir::func::FuncOp function = mlir::func::FuncOp(mlir::func::FuncOp::create(
      getMLIRLocation(FD->getLocation()), name, funcType));

  if ((FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) &&
      !FD->hasAttr<CUDAHostAttr>()) {
    function->setAttr("polygeist.device_only_func",
                      StringAttr::get(builder.getContext(), "1"));
  }

  if (LV == llvm::GlobalValue::InternalLinkage ||
      LV == llvm::GlobalValue::PrivateLinkage || !FD->isDefined() ||
      FD->hasAttr<CUDAGlobalAttr>() || FD->hasAttr<CUDADeviceAttr>()) {
    SymbolTable::setSymbolVisibility(function,
                                     SymbolTable::Visibility::Private);
  } else {
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Public);
  }
  NamedAttrList attrs(function->getAttrDictionary());
  attrs.set("llvm.linkage",
            mlir::LLVM::LinkageAttr::get(builder.getContext(), lnk));
  function->setAttrs(attrs.getDictionary(builder.getContext()));

  functions[name] = function;
  module->push_back(function);

  if (Def->isThisDeclarationADefinition()) {
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::TK_FunctionTemplate);
    assert(Def->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);
    functionsToEmit.push_back(Def);
  } else {
    emitIfFound.insert(name);
  }
  assert(function->getParentOp() == module.get());
  return function;
}

void MLIRASTConsumer::run() {
  while (functionsToEmit.size()) {
    const FunctionDecl *FD = functionsToEmit.front();
    functionsToEmit.pop_front();
    assert(FD->getTemplatedKind() != FunctionDecl::TK_FunctionTemplate);
    assert(FD->getTemplatedKind() !=
           FunctionDecl::TemplatedKind::
               TK_DependentFunctionTemplateSpecialization);
    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(FD))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(FD))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(FD).str();

    if (done.count(name))
      continue;
    done.insert(name);
    MLIRScanner ms(*this, module, LTInfo);
    ms.init(GetOrCreateMLIRFunction(FD), FD);
  }
}

void MLIRASTConsumer::HandleDeclContext(DeclContext *DC) {

  for (auto D : DC->decls()) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(D)) {
      HandleDeclContext(NS);
      continue;
    }
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(D);
    if (!fd) {
      continue;
    }
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition()) {
        continue;
      }
    }
    if (!fd->hasBody())
      continue;

    if (fd->isTemplated()) {
      continue;
    }

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();

    // Don't create std functions unless necessary
    if (StringRef(name).starts_with("_ZNKSt"))
      continue;
    if (StringRef(name).starts_with("_ZSt"))
      continue;
    if (StringRef(name).starts_with("_ZNSt"))
      continue;
    if (StringRef(name).starts_with("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      functionsToEmit.push_back(fd);
    } else {
    }
  }
}

bool MLIRASTConsumer::HandleTopLevelDecl(DeclGroupRef dg) {
  DeclGroupRef::iterator it;

  if (error)
    return true;

  for (it = dg.begin(); it != dg.end(); ++it) {
    if (auto NS = dyn_cast<clang::NamespaceDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::ExternCContextDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    if (auto NS = dyn_cast<clang::LinkageSpecDecl>(*it)) {
      HandleDeclContext(NS);
      continue;
    }
    FunctionDecl *fd = dyn_cast<clang::FunctionDecl>(*it);
    if (!fd) {
      continue;
    }
    if (!fd->doesThisDeclarationHaveABody()) {
      if (!fd->doesDeclarationForceExternallyVisibleDefinition()) {
        continue;
      }
    }
    if (!fd->hasBody())
      continue;
    if (fd->isTemplated()) {
      continue;
    }

    bool externLinkage = true;
    /*
    auto LV = CGM.getFunctionLinkage(fd);
    if (LV == llvm::GlobalValue::InternalLinkage || LV ==
    llvm::GlobalValue::PrivateLinkage) externLinkage = false; if
    (fd->isInlineSpecified()) externLinkage = false;
    */
    if (!CGM.getContext().DeclMustBeEmitted(fd))
      externLinkage = false;

    std::string name;
    if (auto CC = dyn_cast<CXXConstructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXCtorType::Ctor_Complete)).str();
    else if (auto CC = dyn_cast<CXXDestructorDecl>(fd))
      name =
          CGM.getMangledName(GlobalDecl(CC, CXXDtorType::Dtor_Complete)).str();
    else
      name = CGM.getMangledName(fd).str();

    // Don't create std functions unless necessary
    if (StringRef(name).starts_with("_ZNKSt"))
      continue;
    if (StringRef(name).starts_with("_ZSt"))
      continue;
    if (StringRef(name).starts_with("_ZNSt"))
      continue;
    if (StringRef(name).starts_with("_ZN9__gnu"))
      continue;
    if (name == "cudaGetDevice" || name == "cudaMalloc")
      continue;

    if ((emitIfFound.count("*") && name != "fpclassify" && !fd->isStatic() &&
         externLinkage) ||
        emitIfFound.count(name)) {
      functionsToEmit.push_back(fd);
    } else {
    }
  }

  return true;
}

// Wait until Sema has instantiated all the relevant code
// before running codegen on the selected functions.
void MLIRASTConsumer::HandleTranslationUnit(ASTContext &C) { run(); }

mlir::Location MLIRASTConsumer::getMLIRLocation(clang::SourceLocation loc) {
  auto spellingLoc = SM.getSpellingLoc(loc);
  auto lineNumber = SM.getSpellingLineNumber(spellingLoc);
  auto colNumber = SM.getSpellingColumnNumber(spellingLoc);
  auto fileId = SM.getFilename(spellingLoc);

  // Convert relative paths to absolute paths so MLIR preserves them during serialization
  llvm::SmallString<256> absolutePath;
  if (!fileId.empty() && !llvm::sys::path::is_absolute(fileId)) {
    if (auto ec = llvm::sys::fs::real_path(fileId, absolutePath)) {
      // If real_path fails, use current_path + filename
      llvm::sys::fs::current_path(absolutePath);
      llvm::sys::path::append(absolutePath, fileId);
    }
    fileId = StringRef(absolutePath);
  }

  auto ctx = module->getContext();
  return FileLineColLoc::get(ctx, fileId, lineNumber, colNumber);
}

/// Iteratively get the size of each dim of the given ConstantArrayType inst.
static void getConstantArrayShapeAndElemType(const clang::QualType &ty,
                                             SmallVectorImpl<int64_t> &shape,
                                             clang::QualType &elemTy) {
  shape.clear();

  clang::QualType curTy = ty;
  while (curTy->isConstantArrayType()) {
    auto cstArrTy = cast<clang::ConstantArrayType>(curTy);
    shape.push_back(cstArrTy->getSize().getSExtValue());
    curTy = cstArrTy->getElementType();
  }

  elemTy = curTy;
}

// TODO memoize the results?
static bool
isRecursiveStructImpl(const clang::Type *t,
                      SmallPtrSetImpl<const clang::RecordType *> &seen) {
  if (auto PT = dyn_cast<clang::PointerType>(t)) {
    return isRecursiveStructImpl(
        PT->getPointeeType()->getUnqualifiedDesugaredType(), seen);
  } else if (auto RT = dyn_cast<clang::ReferenceType>(t)) {
    return isRecursiveStructImpl(
        RT->getPointeeType()->getUnqualifiedDesugaredType(), seen);
  } else if (auto RT = dyn_cast<clang::RecordType>(t)) {
    if (seen.count(RT))
      return true;
    seen.insert(RT);

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (CXRD) {
      for (auto f : CXRD->bases()) {
        auto baseTy = f.getType()->getUnqualifiedDesugaredType();
        if (isRecursiveStructImpl(baseTy, seen))
          return true;
      }
    }

    for (auto f : RT->getDecl()->fields()) {
      auto fieldTy = f->getType()->getUnqualifiedDesugaredType();
      if (isRecursiveStructImpl(fieldTy, seen))
        return true;
    }

    return false;
  } else {
    return false;
  }
}

static bool isRecursiveStruct(const clang::RecordType *RT) {
  SmallPtrSet<const clang::RecordType *, 4> seen;
  return isRecursiveStructImpl(RT, seen);
}

mlir::Type MLIRASTConsumer::getMLIRType(clang::QualType qt, bool *implicitRef,
                                        bool allowMerge) {
  if (auto ET = dyn_cast<clang::UsingType>(qt)) {
    return getMLIRType(ET->desugar(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::ParenType>(qt)) {
    return getMLIRType(ET->getInnerType(), implicitRef, allowMerge);
  }
  if (auto ET = dyn_cast<clang::DeducedType>(qt)) {
    return getMLIRType(ET->getDeducedType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::SubstTemplateTypeParmType>(qt)) {
    return getMLIRType(ST->getReplacementType(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TemplateSpecializationType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto ST = dyn_cast<clang::TypedefType>(qt)) {
    return getMLIRType(ST->desugar(), implicitRef, allowMerge);
  }
  if (auto DT = dyn_cast<clang::DecltypeType>(qt)) {
    return getMLIRType(DT->desugar(), implicitRef, allowMerge);
  }

  if (auto DT = dyn_cast<clang::DecayedType>(qt)) {
    bool assumeRef = false;
    auto mlirty = getMLIRType(DT->getOriginalType(), &assumeRef, allowMerge);
    if (memRefABI && assumeRef) {
      // Constant array types like `int A[30][20]` will be converted to LLVM
      // type `[20 x i32]* %0`, which has the outermost dimension size erased,
      // and we can only recover to `memref<?x20xi32>` from there. This prevents
      // us from doing more comprehensive analysis. Here we specifically handle
      // this case by unwrapping the clang-adjusted type, to get the
      // corresponding ConstantArrayType with the full dimensions.
      if (memRefFullRank) {
        clang::QualType origTy = DT->getOriginalType();
        if (origTy->isConstantArrayType()) {
          SmallVector<int64_t, 4> shape;
          clang::QualType elemTy;
          getConstantArrayShapeAndElemType(origTy, shape, elemTy);

          return mlir::MemRefType::get(shape, getMLIRType(elemTy));
        }
      }

      // If -memref-fullrank is unset or it cannot be fulfilled.
      auto mt = dyn_cast<MemRefType>(mlirty);
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2[0] = ShapedType::kDynamic;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    } else {
      return getMLIRType(DT->getAdjustedType(), implicitRef, allowMerge);
    }
    return mlirty;
  }
  if (auto CT = dyn_cast<clang::ComplexType>(qt)) {
    bool assumeRef = false;
    auto subType =
        getMLIRType(CT->getElementType(), &assumeRef, /*allowMerge*/ false);
    if (CombinedStructABI && memRefABI && allowMerge) {
      assert(!assumeRef);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(2, subType);
    }
    mlir::Type types[2] = {subType, subType};
    return mlir::LLVM::LLVMStructType::getLiteral(module->getContext(), types);
  }
  if (auto RT = dyn_cast<clang::RecordType>(qt)) {
    if (RT->getDecl()->isInvalidDecl()) {
      RT->getDecl()->dump();
      RT->dump();
    }
    assert(!RT->getDecl()->isInvalidDecl());
    if (typeCache.find(RT) != typeCache.end())
      return typeCache[RT];
    llvm::Type *LT = CGM.getTypes().ConvertType(qt);
    if (!isa<llvm::StructType>(LT)) {
      qt->dump();
      llvm::errs() << "LT: " << *LT << "\n";
    }
    llvm::StructType *ST = cast<llvm::StructType>(LT);

    // Note: Darwin stdio types like __sFILE/__sFILEX may be opaque.
    // We handle opaque/empty-layout cases uniformly below via the generic
    // fallback when 'types' is empty.

    SmallPtrSet<llvm::Type *, 4> Seen;
    bool notAllSame = false;
    for (size_t i = 0; i < ST->getNumElements(); i++) {
      if (ST->getTypeAtIndex(i) != ST->getTypeAtIndex(0U)) {
        notAllSame = true;
      }
    }

    auto CXRD = dyn_cast<CXXRecordDecl>(RT->getDecl());
    if (isLLVMStructABI(RT->getDecl(), ST)) {
      if (ST->isOpaque() || ST->getNumElements() == 0)
        return typeTranslator.translateType(ST);
      return typeTranslator.translateType(anonymize(ST));
    }

    /* TODO
    if (ST->getNumElements() == 1 && !recursive &&
        !RT->getDecl()->fields().empty() && ++RT->getDecl()->field_begin() ==
    RT->getDecl()->field_end()) { auto subT =
    getMLIRType((*RT->getDecl()->field_begin())->getType(), implicitRef,
    allowMerge); return subT;
    }
    */
    bool recursive = isRecursiveStruct(RT);
    if (recursive)
      typeCache[RT] = LLVM::LLVMStructType::getIdentified(
          module->getContext(), ("polygeist@mlir@" + ST->getName()).str());

    SmallVector<mlir::Type, 4> types;

    bool innerLLVM = false;
    if (CXRD) {
      for (auto f : CXRD->bases()) {
        bool subRef = false;
        auto ty = getMLIRType(f.getType(), &subRef, /*allowMerge*/ false);
        assert(!subRef);
        innerLLVM |= isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                            LLVM::LLVMArrayType>(ty);
        types.push_back(ty);
      }
    }

    for (auto f : RT->getDecl()->fields()) {
      bool subRef = false;
      auto ty = getMLIRType(f->getType(), &subRef, /*allowMerge*/ false);
      assert(!subRef);
      innerLLVM |= isa<LLVM::LLVMPointerType, LLVM::LLVMStructType,
                          LLVM::LLVMArrayType>(ty);
      types.push_back(ty);
    }

    if (types.empty()) {
      if (ST->isOpaque() || ST->getNumElements() == 0) {
        return typeTranslator.translateType(ST);
      }
      if (ST->getNumElements() == 1 && ST->getElementType(0U)->isIntegerTy(8))
        return typeTranslator.translateType(anonymize(ST));
    }

    if (recursive) {
      auto LR = typeCache[RT].setBody(types, /*isPacked*/ false);
      assert(LR.succeeded());
      return typeCache[RT];
    }

    if (!memRefABI || notAllSame || !allowMerge || innerLLVM) {
      auto retTy =
          mlir::LLVM::LLVMStructType::getLiteral(module->getContext(), types);
      return retTy;
    }

    if (!types.size()) {
      // Incomplete or opaque records (e.g., Darwin's __sFILE/__sFILEX) may
      // surface here with no discoverable fields. Treat them as opaque
      // identified LLVM structs to keep pointer/global uses consistent and
      // avoid forcing a memref ABI on an unknown layout.
      RT->dump();
      llvm::errs() << "ST: " << *ST << "\n";
      llvm::errs() << "fields\n";
      for (auto f : RT->getDecl()->fields()) {
        llvm::errs() << " +++ ";
        f->getType()->dump();
        llvm::errs() << " @@@ " << *CGM.getTypes().ConvertType(f->getType())
                     << "\n";
      }
      llvm::errs() << "types\n";
      for (auto t : types)
        llvm::errs() << " --- " << t << "\n";

      // Prefer the original named LLVM struct if available; otherwise fall
      // back to translating the opaque llvm::StructType.
      if (ST->hasName()) {
        auto name = ST->getName().str();
        return LLVM::LLVMStructType::getIdentified(module->getContext(), name);
      }
      return typeTranslator.translateType(ST);
    }
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get(types.size(), types[0]);
  }

  auto t = qt->getUnqualifiedDesugaredType();
  if (t->isVoidType()) {
    mlir::OpBuilder builder(module->getContext());
    return builder.getNoneType();
  }

  // if (auto AT = dyn_cast<clang::VariableArrayType>(t)) {
  //   return getMLIRType(AT->getElementType(), implicitRef, allowMerge);
  // }

  if (auto AT = dyn_cast<clang::ArrayType>(t)) {
    auto PTT = AT->getElementType()->getUnqualifiedDesugaredType();
    if (!CStyleMemRef && PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = ShapedType::kDynamic;
    if (auto CAT = dyn_cast<clang::ConstantArrayType>(AT))
      size = CAT->getSize().getZExtValue();
    if (memRefABI && subRef) {
      auto mt = cast<MemRefType>(ET);
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge ||
        (!CStyleMemRef &&
         isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                LLVM::LLVMFunctionType, LLVM::LLVMStructType>(ET)))
      return LLVM::LLVMArrayType::get(
          ET, (size == ShapedType::kDynamic) ? 0 : size);
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto AT = dyn_cast<clang::VectorType>(t)) {
    bool subRef = false;
    auto ET = getMLIRType(AT->getElementType(), &subRef, allowMerge);
    int64_t size = AT->getNumElements();
    if (CombinedStructABI && subRef) {
      auto mt = cast<MemRefType>(ET);
      auto shape2 = std::vector<int64_t>(mt.getShape());
      shape2.insert(shape2.begin(), size);
      if (implicitRef)
        *implicitRef = true;
      return mlir::MemRefType::get(shape2, mt.getElementType(),
                                   MemRefLayoutAttrInterface(),
                                   mt.getMemorySpace());
    }
    if (!memRefABI || !allowMerge || !CombinedStructABI ||
        (!CStyleMemRef &&
         isa<LLVM::LLVMPointerType, LLVM::LLVMArrayType,
                LLVM::LLVMFunctionType, LLVM::LLVMStructType>(ET))) {
      if (isa<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMPointerType>(ET)) {
        return mlir::LLVM::getVectorType(ET, size);
      }
      return mlir::LLVM::LLVMArrayType::get(ET, size);
    }
    if (implicitRef)
      *implicitRef = true;
    return mlir::MemRefType::get({size}, ET);
  }

  if (auto FT = dyn_cast<clang::FunctionProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (isa<mlir::NoneType>(RT))
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    for (auto T : FT->getParamTypes()) {
      Args.push_back(getMLIRType(T));
    }
    return LLVM::LLVMFunctionType::get(RT, Args, FT->isVariadic());
  }
  if (auto FT = dyn_cast<clang::FunctionNoProtoType>(t)) {
    auto RT = getMLIRType(FT->getReturnType());
    if (isa<mlir::NoneType>(RT))
      RT = LLVM::LLVMVoidType::get(RT.getContext());
    SmallVector<mlir::Type> Args;
    return LLVM::LLVMFunctionType::get(RT, Args, /*isVariadic*/ true);
  }

  if (isa<clang::PointerType, clang::ReferenceType>(t)) {
    int64_t outer = ShapedType::kDynamic;
    auto PTT = isa<clang::PointerType>(t) ? cast<clang::PointerType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType()
                                          : cast<clang::ReferenceType>(t)
                                                ->getPointeeType()
                                                ->getUnqualifiedDesugaredType();

    if (!CStyleMemRef && PTT->isCharType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      return typeTranslator.translateType(T);
    }
    if (PTT->isVoidType()) {
      llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
      auto MT = typeTranslator.translateType(T);
      if (!CStyleMemRef)
        return MT;
      else
        return getVoidMemRefTy();
    }
    bool subRef = false;
    auto subType =
        getMLIRType(isa<clang::PointerType>(t)
                        ? cast<clang::PointerType>(t)->getPointeeType()
                        : cast<clang::ReferenceType>(t)->getPointeeType(),
                    &subRef, /*allowMerge*/ true);

    if (!memRefABI)
      return LLVM::LLVMPointerType::get(subType.getContext());

    if (!CStyleMemRef &&
        isa<LLVM::LLVMArrayType, LLVM::LLVMStructType,
                    LLVM::LLVMPointerType, LLVM::LLVMFunctionType>(subType))
      return LLVM::LLVMPointerType::get(subType.getContext());

    if (isa<clang::ArrayType>(PTT)) {
      if (isa<MemRefType>(subType)) {
        assert(subRef);
        if (allowMerge && isa<clang::PointerType>(t)) {
          // Pointer-to-array (NOT reference-to-array): prepend outer dim.
          // float (*)[M] → memref<?xMxf32> (can index multiple rows).
          // float (&)[M] → memref<Mxf32> (single array reference, no outer dim).
          auto mt = cast<MemRefType>(subType);
          // Only for arrays of scalars (contiguous ND data).
          // Skip arrays of pointers/memrefs (e.g., double *[4]).
          if (!isa<MemRefType,
                                       LLVM::LLVMPointerType>(mt.getElementType())) {
            auto shape2 = std::vector<int64_t>(mt.getShape());
            shape2.insert(shape2.begin(), outer);
            return mlir::MemRefType::get(shape2, mt.getElementType(),
                                         MemRefLayoutAttrInterface(),
                                         mt.getMemorySpace());
          }
        }
        return subType;
      } else {
        if (!CStyleMemRef)
          return LLVM::LLVMPointerType::get(subType.getContext());
      }
    }

    if (isa<clang::VectorType>(PTT) || isa<clang::ComplexType>(PTT)) {
      if (isa<MemRefType>(subType)) {
        assert(subRef);
        auto mt = cast<MemRefType>(subType);
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      } else {
        if (!CStyleMemRef)
          return LLVM::LLVMPointerType::get(subType.getContext());
      }
    }

    if (isa<clang::RecordType>(PTT))
      if (subRef) {
        auto mt = cast<MemRefType>(subType);
        auto shape2 = std::vector<int64_t>(mt.getShape());
        shape2.insert(shape2.begin(), outer);
        return mlir::MemRefType::get(shape2, mt.getElementType(),
                                     MemRefLayoutAttrInterface(),
                                     mt.getMemorySpace());
      }

    assert(!subRef);
    return mlir::MemRefType::get({outer}, subType);
  }

  if (t->isBuiltinType() || isa<clang::EnumType>(t)) {
    if (t->isBooleanType()) {
      OpBuilder builder(module->getContext());
      return builder.getIntegerType(8);
    }
    llvm::Type *T = CGM.getTypes().ConvertType(QualType(t, 0));
    mlir::OpBuilder builder(module->getContext());
    if (T->isVoidTy()) {
      return builder.getNoneType();
    }
    if (T->isFloatTy()) {
      return builder.getF32Type();
    }
    if (T->isDoubleTy()) {
      return builder.getF64Type();
    }
    if (T->isX86_FP80Ty())
      return builder.getF80Type();
    if (T->isFP128Ty())
      return builder.getF128Type();

    if (auto IT = dyn_cast<llvm::IntegerType>(T)) {
      return builder.getIntegerType(IT->getBitWidth());
    }
  }
  qt->dump();
  llvm_unreachable("unhandled type");
}

llvm::Type *MLIRASTConsumer::getLLVMType(clang::QualType t) {
  if (t->isVoidType()) {
    return llvm::Type::getVoidTy(llvmMod.getContext());
  }
  llvm::Type *T = CGM.getTypes().ConvertType(t);
  return T;
}

#include "llvm/TargetParser/Host.h"

#include "clang/Frontend/FrontendAction.h"
class MLIRAction : public clang::ASTFrontendAction {
public:
  std::set<std::string> emitIfFound;
  std::set<std::string> done;
  mlir::OwningOpRef<mlir::ModuleOp> &module;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmStringGlobals;
  std::map<std::string, std::pair<mlir::memref::GlobalOp, bool>> globals;
  std::map<std::string, mlir::func::FuncOp> functions;
  std::map<std::string, mlir::LLVM::GlobalOp> llvmGlobals;
  std::map<std::string, mlir::LLVM::LLVMFuncOp> llvmFunctions;
  MLIRAction(std::string fn, mlir::OwningOpRef<mlir::ModuleOp> &module)
      : module(module) {
    emitIfFound.insert(fn);
  }
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
    return std::unique_ptr<clang::ASTConsumer>(new MLIRASTConsumer(
        emitIfFound, done, llvmStringGlobals, globals, functions, llvmGlobals,
        llvmFunctions, CI.getPreprocessor(), CI.getASTContext(), module,
        CI.getSourceManager(), CI.getCodeGenOpts()));
  }
};

mlir::func::FuncOp MLIRScanner::EmitDirectCallee(const FunctionDecl *FD) {
  return Glob.GetOrCreateMLIRFunction(FD);
}

mlir::Location MLIRScanner::getMLIRLocation(clang::SourceLocation loc) {
  return Glob.getMLIRLocation(loc);
}

mlir::Type MLIRScanner::getMLIRType(clang::QualType t) {
  return Glob.getMLIRType(t);
}

llvm::Type *MLIRScanner::getLLVMType(clang::QualType t) {
  return Glob.getLLVMType(t);
}

mlir::Value MLIRScanner::getTypeSize(mlir::Location loc, clang::QualType t) {
  // llvm::Type *T = Glob.CGM.getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray);
  if (isArray) {
    auto MT = cast<MemRefType>(innerTy);
    size_t num = 1;
    for (auto n : MT.getShape()) {
      assert(n > 0);
      num *= n;
    }
    return builder.create<arith::MulIOp>(
        loc,
        builder.create<polygeist::TypeSizeOp>(
            loc, builder.getIndexType(),
            mlir::TypeAttr::get(MT.getElementType())),
        builder.create<arith::ConstantIndexOp>(loc, num));
  }
  assert(!isArray);
  return builder.create<polygeist::TypeSizeOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

mlir::Value MLIRScanner::getTypeAlign(mlir::Location loc, clang::QualType t) {
  // llvm::Type *T = Glob.CGM.getTypes().ConvertType(t);
  // return (Glob.llvmMod.getDataLayout().getTypeSizeInBits(T) + 7) / 8;
  bool isArray = false;
  auto innerTy = Glob.getMLIRType(t, &isArray, /*allowMerge=*/false);
  assert(!isArray);
  return builder.create<polygeist::TypeAlignOp>(
      loc, builder.getIndexType(),
      mlir::TypeAttr::get(innerTy)); // DLI.getTypeSize(innerTy);
}

#include "clang/Frontend/TextDiagnosticBuffer.h"

static bool parseMLIR(const char *Argv0, std::vector<std::string> filenames,
                      std::string fn, std::vector<std::string> includeDirs,
                      std::vector<std::string> defines,
                      mlir::OwningOpRef<mlir::ModuleOp> &module,
                      llvm::Triple &triple, llvm::DataLayout &DL,
                      llvm::Triple &gpuTriple, llvm::DataLayout &gpuDL) {

  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
  // Buffer diagnostics from argument parsing so that we can output them using a
  // well formed diagnostic object.
  DiagnosticOptions DiagOpts;
  TextDiagnosticBuffer *DiagsBuffer = new TextDiagnosticBuffer;
  DiagnosticsEngine Diags(DiagID, DiagOpts, DiagsBuffer);

  bool Success;
  //{
  const char *binary = Argv0; // CudaLower ? "clang++" : "clang";
  const unique_ptr<Driver> driver(
      new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags));
  mlirclang::ArgumentList Argv;
  Argv.push_back(binary);
  // TODO we should probably preserve the order of these args - they matter in
  // some cases as is with this one - it has to be before the input file
  if (Lang != "") {
    Argv.push_back("-x");
    Argv.emplace_back(Lang);
  }
  for (const auto &filename : filenames) {
    Argv.emplace_back(filename);
  }
  if (FOpenMP)
    Argv.push_back("-fopenmp");
  if (TargetTripleOpt != "") {
    Argv.push_back("-target");
    Argv.emplace_back(TargetTripleOpt);
  }
  if (McpuOpt != "") {
    Argv.emplace_back("-mcpu=", McpuOpt);
  }
  if (Standard != "") {
    Argv.emplace_back("-std=", Standard);
  }
  if (ResourceDir != "") {
    Argv.push_back("-resource-dir");
    Argv.emplace_back(ResourceDir);
  }
  if (SysRoot != "") {
    Argv.push_back("--sysroot");
    Argv.emplace_back(SysRoot);
  }
  if (Verbose) {
    Argv.push_back("-v");
  }
  if (NoCUDAInc) {
    Argv.push_back("-nocudainc");
  }
  if (NoCUDALib) {
    Argv.push_back("-nocudalib");
  }
  if (CUDAGPUArch != "") {
    Argv.emplace_back("--cuda-gpu-arch=", CUDAGPUArch);
  }
  if (CUDAPath != "") {
    Argv.emplace_back("--cuda-path=", CUDAPath);
  }
  if (MArch != "") {
    Argv.emplace_back("-march=", MArch);
  }
  for (const auto &dir : includeDirs) {
    Argv.push_back("-I");
    Argv.emplace_back(dir);
  }
  for (const auto &define : defines) {
    Argv.emplace_back("-D", define);
  }
  for (const auto &Include : Includes) {
    Argv.push_back("-include");
    Argv.emplace_back(Include);
  }

  // Add debug info flag if requested
  extern llvm::cl::opt<bool> EmitDebugInfo;
  if (EmitDebugInfo) {
    Argv.push_back("-g");
  }

  Argv.push_back("-emit-ast");

  const unique_ptr<Compilation> compilation(
      driver->BuildCompilation(Argv.getArguments()));
  JobList &Jobs = compilation->getJobs();
  if (Jobs.size() < 1)
    return false;

  MLIRAction Act(fn, module);

  for (auto &job : Jobs) {
    std::unique_ptr<CompilerInstance> Clang(new CompilerInstance());

    Command *cmd = cast<Command>(&job);
    if (strcmp(cmd->getCreator().getName(), "clang"))
      return false;

    const ArgStringList *args = &cmd->getArguments();

    Success = CompilerInvocation::CreateFromArgs(Clang->getInvocation(), *args,
                                                 Diags);
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    void *GetExecutablePathVP = (void *)(intptr_t)GetExecutablePath;
    // Infer the builtin include path if unspecified.
    if (Clang->getHeaderSearchOpts().UseBuiltinIncludes &&
        Clang->getHeaderSearchOpts().ResourceDir.size() == 0)
      Clang->getHeaderSearchOpts().ResourceDir =
          GetResourcesPath(Argv0, GetExecutablePathVP);

    //}
    Clang->getInvocation().getFrontendOpts().DisableFree = false;

    // Create the actual diagnostics engine.
    Clang->createDiagnostics();
    if (!Clang->hasDiagnostics())
      return false;

    DiagsBuffer->FlushDiagnostics(Clang->getDiagnostics());
    if (!Success)
      return false;

    // Create and execute the frontend action.

    // Create the target instance.
    Clang->setTarget(TargetInfo::CreateTargetInfo(
        Clang->getDiagnostics(), Clang->getTargetOpts()));
    if (!Clang->hasTarget())
      return false;

    // Create TargetInfo for the other side of CUDA and OpenMP compilation.
    if ((Clang->getLangOpts().CUDA ||
         Clang->getLangOpts().OpenMPIsTargetDevice) &&
        !Clang->getFrontendOpts().AuxTriple.empty()) {
      auto TO = std::make_shared<clang::TargetOptions>();
      TO->Triple = llvm::Triple::normalize(Clang->getFrontendOpts().AuxTriple);
      TO->HostTriple = Clang->getTarget().getTriple().str();
      Clang->setAuxTarget(
          TargetInfo::CreateTargetInfo(Clang->getDiagnostics(), *TO));
    }

    // Inform the target of the language options.
    //
    // FIXME: We shouldn't need to do this, the target should be immutable once
    // created. This complexity should be lifted elsewhere.
    Clang->getTarget().adjust(Clang->getDiagnostics(), Clang->getLangOpts(),
                              Clang->getAuxTarget());

    llvm::Triple jobTriple = Clang->getTarget().getTriple();
    if (triple.str() == "" || !jobTriple.isNVPTX()) {
      triple = jobTriple;
      module.get()->setAttr(
          LLVM::LLVMDialect::getTargetTripleAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      DL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          LLVM::LLVMDialect::getDataLayoutAttrName(),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));

      module.get()->setAttr(DataLayoutSpecAttr::name,
                            translateDataLayout(DL, module->getContext()));

      // Add target-cpu and target-features attributes to functions. If
      // we have a decl for the function and it has a target attribute then
      // parse that and add it to the feature set.
      StringRef TargetCPU = Clang->getTarget().getTargetOpts().CPU;
      StringRef TuneCPU = Clang->getTarget().getTargetOpts().TuneCPU;
      std::vector<std::string> Features =
          Clang->getTarget().getTargetOpts().Features;

      if (!TargetCPU.empty()) {
        module.get()->setAttr("polygeist.target-cpu",
                              StringAttr::get(module->getContext(), TargetCPU));
      }
      if (!TuneCPU.empty()) {
        module.get()->setAttr("polygeist.tune-cpu",
                              StringAttr::get(module->getContext(), TuneCPU));
      }
      if (!Features.empty()) {
        llvm::sort(Features);
        module.get()->setAttr(
            "polygeist.target-features",
            StringAttr::get(module->getContext(), llvm::join(Features, ",")));
      }
    }

    // TODO investigate what AMDGCN and AMDGPU are, do we need them both?
    if (jobTriple.isNVPTX() || jobTriple.isAMDGCN() || jobTriple.isAMDGPU()) {
      gpuTriple = jobTriple;
      module.get()->setAttr(
          StringRef("polygeist.gpu_module." +
                    LLVM::LLVMDialect::getTargetTripleAttrName().str()),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getTriple().getTriple()));
      gpuDL = llvm::DataLayout(Clang->getTarget().getDataLayoutString());
      module.get()->setAttr(
          StringRef("polygeist.gpu_module." +
                    LLVM::LLVMDialect::getDataLayoutAttrName().str()),
          StringAttr::get(module->getContext(),
                          Clang->getTarget().getDataLayoutString()));
    }

    for (const auto &FIF : Clang->getFrontendOpts().Inputs) {
      // Reset the ID tables if we are reusing the SourceManager and parsing
      // regular files.
      if (Clang->hasSourceManager() && !Act.isModelParsingAction())
        Clang->getSourceManager().clearIDTables();
      if (Act.BeginSourceFile(*Clang, FIF)) {

        llvm::Error err = Act.Execute();
        if (err) {
          llvm::errs() << "saw error: " << err << "\n";
          return false;
        }
        assert(Clang->hasSourceManager());

        Act.EndSourceFile();
      }
    }

    if (Clang->getDiagnostics().hasErrorOccurred()) {
      return false;
    }
  }
  return true;
}
