# Polygeist Overview

Polygeist is an MLIR-based front-end that “raises” C/C++ (and CUDA/OpenMP dialects) into higher-level MLIR representations. It hooks into Clang’s front-end, iterates over the AST nodes produced by Clang, and emits standard MLIR dialects (Affine/SCF/LLVM/GPU) instead of lowering directly to LLVM IR. Once in MLIR, you can run polyhedral optimizations, affine transformations, GPU transpilation passes, or lower back to LLVM and binaries. The repository contains:

- **`cgeist`** – A Clang-based driver that parses C/C++ (including OpenMP), walks the Clang AST, and emits structured MLIR (Affine, SCF, GPU dialects).
- **`polygeist-opt`** – Optimization passes that perform loop fusion, tiling, parallelization, and rewrite kernels into higher-level dialects.
- **Polymer extensions** – Optional Pluto/ISL-backed passes for advanced polyhedral scheduling.
- **GPU transpilation work** – Raising CUDA kernels to MLIR for retargeting (CPU↔GPU) and performance portability experiments.

Typical use cases:

1. **Raising existing C/C++** – Convert code bases into MLIR to leverage MLIR’s transformation ecosystem.
2. **Automatic parallelization** – Run Pluto/Polymer passes to expose more parallelism, then lower to OpenMP or GPU dialects.
3. **Transpilation** – Convert GPU kernels to CPU variants (or vice versa) using the MLIR GPU/Async lowerings.
4. **Research playground** – Prototype new compiler passes or dialects without re-implementing a full front-end.

Polygeist sits alongside upstream LLVM/MLIR; you either build it against a pre-existing LLVM tree (Option 1) or as an external LLVM project (Option 2, below). Everything else in this README focuses on building/testing.

## Key Components & Directories

- `tools/cgeist/` – Driver and Clang integration code.
- `tools/polygeist-opt/` – Command-line optimizer hosting the Polygeist passes.
- `tools/polymer/` – Optional Polygeist Polymer extensions (Pluto/ISL helpers).
- `lib/` – MLIR dialects, analyses, and transform implementations.
- `test/` – MLIR and Cgeist regression tests (`ninja check-polygeist-opt`, `ninja check-cgeist`).
- `docs/` (within Polymer) – Additional Polygeist papers and scripts referenced by the citations at the end of this file.

Polygeist relies on upstream LLVM/Clang/MLIR, so you must keep the LLVM submodule (or external checkout) in sync with the Polygeist revision you build.

# Build instructions

## Requirements 
- Working C and C++ toolchains(compiler, linker)
- cmake
- make or ninja

## 1. Clone Polygeist
```sh
git clone --recursive https://github.com/llvm/Polygeist
cd Polygeist
```

## 2. Install LLVM, MLIR, Clang, and Polygeist

### Option 1: Using pre-built LLVM, MLIR, and Clang

Polygeist can be built by providing paths to a pre-built MLIR and Clang toolchain.

#### 1. Build LLVM, MLIR, and Clang:
```sh
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

To enable compilation to cuda add `-DMLIR_ENABLE_CUDA_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `CUDACXX`, `CUDA_PATH`, and/or `-DCMAKE_CUDA_COMPILER`)

To enable the ROCM backend add `-DMLIR_ENABLE_ROCM_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `-DHIP_CLANG_INCLUDE_PATH`, and/or `ROCM_PATH`)

For ISL-enabled polymer, `polly` must be added to the `LLVM_ENABLE_PROJECTS` variable.

For faster compilation we recommend using `-DLLVM_USE_LINKER=lld`.

#### 2. Build Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

For faster compilation we recommend using `-DPOLYGEIST_USE_LINKER=lld`.

##### GPU backends

To enable the CUDA backend add `-DPOLYGEIST_ENABLE_CUDA=1`

To enable the ROCM backend add `-DPOLYGEIST_ENABLE_ROCM=1`

##### Polymer

To enable polymer, add `-DPOLYGEIST_ENABLE_POLYMER=1`

There are two configurations of polymer that can be built - one with Pluto and one with ISL. 

###### Pluto
Add `-DPOLYGEIST_POLYMER_ENABLE_PLUTO=1`
This will cause the cmake invokation to pull and build the dependencies for polymer. To specify a custom directory for the dependencies, specify `-DPOLYMER_DEP_DIR=<absolute-dir>`. The dependencies will be build using the `tools/polymer/build_polymer_deps.sh`.

To run the polymer pluto tests, use `ninja check-polymer`.

###### ISL

Add `-DPOLYGEIST_POLYMER_ENABLE_ISL=1`
This requires an `llvm-project` build with `polly` enabled as a subproject.


### Option 2: Using unified LLVM, MLIR, Clang, and Polygeist build

Polygeist can also be built as an external LLVM project using [LLVM_EXTERNAL_PROJECTS](https://llvm.org/docs/CMake.html#llvm-related-variables).

1. Build LLVM, MLIR, Clang, and Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`
`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`

# Citing Polygeist

If you use Polygeist, please consider citing the relevant publications:

``` bibtex
@inproceedings{polygeistPACT,
  title = {Polygeist: Raising C to Polyhedral MLIR},
  author = {Moses, William S. and Chelini, Lorenzo and Zhao, Ruizhe and Zinenko, Oleksandr},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
  numpages = {12},
  location = {Virtual Event},
  series = {PACT '21},
  publisher = {Association for Computing Machinery},
  year = {2021},
  address = {New York, NY, USA},
  keywords = {Polygeist, MLIR, Polyhedral, LLVM, Compiler, C++, Pluto, Polly, OpenScop, Parallel, OpenMP, Affine, Raising, Transformation, Splitting, Automatic-Parallelization, Reduction, Polybench},
}
@inproceedings{10.1145/3572848.3577475,
  author = {Moses, William S. and Ivanov, Ivan R. and Domke, Jens and Endo, Toshio and Doerfert, Johannes and Zinenko, Oleksandr},
  title = {High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs},
  year = {2023},
  isbn = {9798400700156},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3572848.3577475},
  doi = {10.1145/3572848.3577475},
  booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages = {119–134},
  numpages = {16},
  keywords = {MLIR, polygeist, CUDA, barrier synchronization},
  location = {Montreal, QC, Canada},
  series = {PPoPP '23}
}
@inproceedings{10444828,
  author = {Ivanov, Ivan R. and Zinenko, Oleksandr and Domke, Jens and Endo, Toshio and Moses, William S.},
  booktitle = {2024 IEEE/ACM International Symposium on Code Generation and Optimization (CGO)},
  title = {Retargeting and Respecializing GPU Workloads for Performance Portability},
  year = {2024},
  volume = {},
  issn = {},
  pages = {119-132},
  doi = {10.1109/CGO57630.2024.10444828},
  url = {https://doi.ieeecomputersociety.org/10.1109/CGO57630.2024.10444828},
  publisher = {IEEE Computer Society},
  address = {Los Alamitos, CA, USA},
  month = {mar}
}
```
