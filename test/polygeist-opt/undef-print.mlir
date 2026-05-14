// RUN: polygeist-opt %s | FileCheck %s

module {
  func.func @undef_print() -> i32 {
    %0 = "polygeist.undef"() : () -> i32
    return %0 : i32
  }
}

// CHECK-LABEL: func.func @undef_print
// CHECK: %[[U:.+]] = polygeist.undef : i32
// CHECK-NOT: "polygeist.undef"() : () -> i32
// CHECK: return %[[U]] : i32
