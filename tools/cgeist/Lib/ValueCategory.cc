//===- ValueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueCategory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::arith;

/// Helper to infer the element type stored behind an LLVM opaque pointer.
/// Returns nullptr if the type cannot be inferred from the defining op.
static mlir::Type inferPointerElementType(mlir::Value ptr) {
  if (auto alloca = ptr.getDefiningOp<LLVM::AllocaOp>())
    return alloca.getElemType();
  if (auto gep = ptr.getDefiningOp<LLVM::GEPOp>())
    return gep.getElemType();
  if (auto m2p = ptr.getDefiningOp<polygeist::Memref2PointerOp>())
    return cast<MemRefType>(m2p.getSource().getType()).getElementType();
  if (auto p2m = ptr.getDefiningOp<polygeist::Pointer2MemrefOp>())
    return cast<MemRefType>(p2m.getResult().getType()).getElementType();
  if (auto addrOf = ptr.getDefiningOp<LLVM::AddressOfOp>()) {
    auto mod = addrOf->getParentOfType<mlir::ModuleOp>();
    if (mod)
      if (auto global =
              mod.lookupSymbol<LLVM::GlobalOp>(addrOf.getGlobalName()))
        return global.getType();
  }
  return nullptr;
}

ValueCategory::ValueCategory(mlir::Value val, bool isReference)
    : val(val), isReference(isReference) {
  assert(val && "null value");
  if (isReference) {
    if (!(isa<MemRefType>(val.getType()) ||
          isa<LLVM::LLVMPointerType>(val.getType()))) {
      llvm::errs() << "val: " << val << "\n";
    }
    assert((isa<MemRefType>(val.getType()) ||
            isa<LLVM::LLVMPointerType>(val.getType())) &&
           "Reference value must have pointer/memref type");
  }
}

mlir::Value ValueCategory::getValue(mlir::Location loc,
                                    mlir::OpBuilder &builder) const {
  assert(val && "must be not-null");
  if (!isReference)
    return val;
  if (isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
    mlir::Type elemTy = inferPointerElementType(val);
    assert(elemTy && "cannot infer element type for LLVM pointer load");
    return builder.create<mlir::LLVM::LoadOp>(loc, elemTy, val);
  }
  if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have shape 1");
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    return builder.create<memref::LoadOp>(loc, val,
                                          std::vector<mlir::Value>({c0}));
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

void ValueCategory::store(mlir::Location loc, mlir::OpBuilder &builder,
                          mlir::Value toStore) const {
  assert(isReference && "must be a reference");
  assert(val && "expect not-null");
  if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
    // With opaque pointers, unwrap Pointer2MemrefOp back to the source pointer.
    if (auto p2m = toStore.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
      if (isa<LLVM::LLVMPointerType>(p2m.getSource().getType()))
        toStore = p2m.getSource();
    }
    // If toStore is a memref, convert it to an LLVM pointer for the store.
    if (auto mt = dyn_cast<MemRefType>(toStore.getType())) {
      auto destPtrTy =
          LLVM::LLVMPointerType::get(builder.getContext(), pt.getAddressSpace());
      toStore =
          builder.create<polygeist::Memref2PointerOp>(loc, destPtrTy, toStore);
    }
    builder.create<mlir::LLVM::StoreOp>(loc, toStore, val);
    return;
  }
  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have size 1");
    if (isa<mlir::LLVM::LLVMPointerType>(toStore.getType())) {
      if (auto MT = dyn_cast<mlir::MemRefType>(
              cast<MemRefType>(val.getType()).getElementType())) {
        assert(MT.getShape().size() == 1);
        assert(MT.getShape()[0] == ShapedType::kDynamic);
        toStore = builder.create<polygeist::Pointer2MemrefOp>(loc, MT, toStore);
      }
    }
    assert(toStore.getType() ==
               cast<MemRefType>(val.getType()).getElementType() &&
           "expect same type");
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    builder.create<mlir::memref::StoreOp>(loc, toStore, val,
                                          std::vector<mlir::Value>({c0}));
    return;
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

ValueCategory ValueCategory::dereference(mlir::Location loc,
                                         mlir::OpBuilder &builder) const {
  assert(val && "val must be not-null");

  if (isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
    if (!isReference)
      return ValueCategory(val, /*isReference*/ true);
    else {
      // Dereferencing a pointer-to-pointer: load produces another pointer.
      mlir::Type elemTy = inferPointerElementType(val);
      if (!elemTy)
        elemTy = LLVM::LLVMPointerType::get(builder.getContext());
      return ValueCategory(
          builder.create<mlir::LLVM::LoadOp>(loc, elemTy, val),
          /*isReference*/ true);
    }
  }

  if (auto mt = cast<mlir::MemRefType>(val.getType())) {
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    auto shape = std::vector<int64_t>(mt.getShape());

    if (isReference) {
      if (shape.size() > 1) {
        shape.erase(shape.begin());
        auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                         mt.getLayout(), mt.getMemorySpace());
        return ValueCategory(
            builder.create<polygeist::SubIndexOp>(loc, mt0, val, c0),
            /*isReference*/ true);
      } else {
        // shape[0] = -1;
        return ValueCategory(builder.create<mlir::memref::LoadOp>(
                                 loc, val, std::vector<mlir::Value>({c0})),
                             /*isReference*/ true);
      }
    }
    return ValueCategory(val, /*isReference*/ true);
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

// TODO: too long and difficult to understand.
void ValueCategory::store(mlir::Location loc, mlir::OpBuilder &builder,
                          ValueCategory toStore, bool isArray) const {
  assert(toStore.val);
  if (isArray) {
    if (!toStore.isReference) {
      llvm::errs() << " toStore.val: " << toStore.val << " isref "
                   << toStore.isReference << " isar" << isArray << "\n";
    }
    assert(toStore.isReference);
    auto zeroIndex = builder.create<ConstantIndexOp>(loc, 0);

    if (auto smt = dyn_cast<mlir::MemRefType>(toStore.val.getType())) {
      assert(smt.getShape().size() <= 2);

      if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
        assert(smt.getElementType() == mt.getElementType());
        if (mt.getShape().size() != smt.getShape().size()) {
          llvm::errs() << " val: " << val << " tsv: " << toStore.val << "\n";
          llvm::errs() << " mt: " << mt << " smt: " << smt << "\n";
        }
        assert(mt.getShape().size() == smt.getShape().size());
        assert(smt.getShape().back() == mt.getShape().back());

        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<ConstantIndexOp>(loc, i));
          builder.create<mlir::memref::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              val, idx);
        }
      } else {
        auto pt = cast<mlir::LLVM::LLVMPointerType>(val.getType());
        // Infer the aggregate type the opaque pointer points to.
        mlir::Type pointeeType = inferPointerElementType(val);
        assert(pointeeType && "cannot infer pointee type for LLVM pointer");
        mlir::Type elty;
        if (auto at = dyn_cast<LLVM::LLVMArrayType>(pointeeType)) {
          elty = at.getElementType();
          if (smt.getShape().back() != at.getNumElements()) {
            llvm::errs() << " pt: " << pt << " smt: " << smt << "\n";
            llvm::errs() << " val: " << val << " val.isRef: " << isReference
                         << " ts: " << toStore.val
                         << " ts.isRef: " << toStore.isReference
                         << " isArray: " << isArray << "\n";
          }
          assert(smt.getShape().back() == at.getNumElements());
        } else {
          auto st = dyn_cast<LLVM::LLVMStructType>(pointeeType);
          elty = st.getBody()[0];
          assert(smt.getShape().back() == (ssize_t)st.getBody().size());
        }
        if (elty != smt.getElementType()) {
          llvm::errs() << " pt: " << pt << " smt: " << smt << "\n";
          llvm::errs() << " elty: " << elty
                       << " smt.getElementType(): " << smt.getElementType()
                       << "\n";
          llvm::errs() << " val: " << val << " val.isRef: " << isReference
                       << " ts: " << toStore.val
                       << " ts.isRef: " << toStore.isReference
                       << " isArray: " << isArray << "\n";
        }
        assert(elty == smt.getElementType());
        auto resPtrTy =
            LLVM::LLVMPointerType::get(builder.getContext(), pt.getAddressSpace());

        auto zero32 = builder.create<ConstantIntOp>(loc, 0, 32);
        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<ConstantIndexOp>(loc, i));
          mlir::Value lidx[] = {zero32,
                                builder.create<ConstantIntOp>(loc, i, 32)};
          builder.create<mlir::LLVM::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              builder.create<mlir::LLVM::GEPOp>(loc, resPtrTy, pointeeType,
                                                val, lidx));
        }
      }
    } else if (auto smt = dyn_cast<mlir::MemRefType>(val.getType())) {
      assert(smt.getShape().size() <= 2);

      auto pt = cast<LLVM::LLVMPointerType>(toStore.val.getType());
      // Infer the aggregate type the opaque pointer points to.
      mlir::Type pointeeType = inferPointerElementType(toStore.val);
      assert(pointeeType && "cannot infer pointee type for LLVM pointer");
      mlir::Type elty;
      if (auto at = dyn_cast<LLVM::LLVMArrayType>(pointeeType)) {
        elty = at.getElementType();
        assert(smt.getShape().back() == at.getNumElements());
      } else {
        auto st = dyn_cast<LLVM::LLVMStructType>(pointeeType);
        elty = st.getBody()[0];
        assert(smt.getShape().back() == (ssize_t)st.getBody().size());
      }
      assert(elty == smt.getElementType());
      auto resPtrTy =
          LLVM::LLVMPointerType::get(builder.getContext(), pt.getAddressSpace());

      auto zero32 = builder.create<ConstantIntOp>(loc, 0, 32);
      for (ssize_t i = 0; i < smt.getShape().back(); i++) {
        SmallVector<mlir::Value, 2> idx;
        if (smt.getShape().size() == 2)
          idx.push_back(zeroIndex);
        idx.push_back(builder.create<ConstantIndexOp>(loc, i));
        mlir::Value lidx[] = {zero32,
                              builder.create<ConstantIntOp>(loc, i, 32)};
        auto gepResult = builder.create<mlir::LLVM::GEPOp>(
            loc, resPtrTy, pointeeType, toStore.val, lidx);
        builder.create<mlir::memref::StoreOp>(
            loc,
            builder.create<mlir::LLVM::LoadOp>(loc, elty, gepResult),
            val, idx);
      }
    } else
      store(loc, builder, toStore.getValue(loc, builder));
  } else {
    store(loc, builder, toStore.getValue(loc, builder));
  }
}
