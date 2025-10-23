# ARC — A Modern C++ SIMD and Math Library

[![Build](https://github.com/richardthorell/arc/actions/workflows/tests.yml/badge.svg)](https://github.com/richardthorell/arc/actions)
![License](https://img.shields.io/github/license/richardthorell/arc)
![C++](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Platform](https://img.shields.io/badge/platform-cross--platform-lightgrey.svg)

---

### Overview

**ARC** is a modern, header-only C++ library designed to provide a clean and extensible abstraction over SIMD operations.  
It aims to unify low-level SIMD intrinsics under a single, intuitive API while maintaining full performance through compile-time specialization.

The library provides:
- Type-safe SIMD abstractions (`simd<T, N>` and `simd_mask<N>`)
- Free-function based API (no heavy class hierarchies)
- Consistent arithmetic, logical, and comparison operations
- Masked and conditional operations
- Extensible design for different architectures (SSE, AVX, NEON, etc.)
- Easy integration into modern CMake projects

---

### Building & Testing

This is a **header-only** library — no installation required.

To build and run the tests (using **Catch2** and **CMake**):

```bash
git clone https://github.com/richardthorell/arc.git
cd arc/tests
cmake -S . -B build
cmake --build build
ctest --test-dir build
