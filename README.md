# miniprot (Rust port)

This project is a Rust port of [`lh3/miniprot`](https://github.com/lh3/miniprot).

However, it is not a trivial rewrite of the original C code, and certainly not an AI-generated line-by-line translation of the upstream repository. In this port, I intentionally removed many legacy C patterns and favored implementations that are more direct, maintainable, and idiomatic in Rust.

## Project Scope

- Preserve the core mathematical model and algorithmic ideas of the original miniprot.
- Reorganize the implementation in Rust instead of reproducing the original C control flow, memory layout, macro style, and historical baggage line by line.
- Prioritize clarity, maintainability, and performance over source-level similarity to the upstream code.

## Non-Goals

- This project does not aim to be a bit-for-bit clone of the C implementation.
- It does not guarantee that internal intermediate states, floating-point details, iteration order, tie-breaking behavior, parallel scheduling, or final outputs will exactly match the upstream implementation on every input.
- Preserving legacy C coding patterns is not itself a compatibility goal.

## Consistency Boundary

This project only aims to stay consistent with the original implementation at the mathematical and algorithmic level:

- the problem definition should remain aligned
- the core algorithmic intent should remain aligned
- the main scoring, chaining, and alignment logic should remain aligned

It does not guarantee computational identity or byte-for-byte identity. In other words, even when two implementations follow the same algorithmic ideas, the exact numerical path, edge-case behavior, output details, and sometimes even parts of the final result may differ. This is an intentional design boundary, not by itself a bug definition.

## Build

```bash
cargo build --release
```

## Performance

Benchmarked against upstream C oracle (`lh3/miniprot` v0.18-r281) on human
genome GRCh38.p14 (`GCF_000001405.40`, 928 MB gzipped FASTA). Both binaries
compiled with arch-native flags (`-C target-cpu=native` / `-march=native
-mtune=native`). 200 protein queries derived from chromosome 1 ORFs.

### aarch64 (Apple M2, 4 threads)

| Phase            | C oracle | Rust     | Rust vs C    |
|------------------|---------:|---------:|-------------:|
| Index build      | 5m00s    | **3m27s** | **31% faster** |
| Map (1t, CPU)    | 6.26s    | **5.23s** | **16.5% faster** |
| Map (4t, CPU)    | 8.92s    | **6.10s** | **31.6% faster** |

### x86_64 (Intel Xeon Gold 5320, Ice Lake, 4 threads)

| Phase            | C oracle | Rust     | Rust vs C    |
|------------------|---------:|---------:|-------------:|
| Index build      | 78.3s    | **40.5s** | **48% faster** |
| Map (1t, user)   | 23.01s   | **22.82s** | **0.8% faster** |
| Map (1t, total)  | 27.14s   | **25.96s** | **4.4% faster** |
| Map (4t, total)  | 8.78s    | **8.52s**  | **3.0% faster** |

### What makes it fast

- **Arch-native SIMD DP** — NEON on aarch64, SSE4.1 on x86_64. Both match
  the C implementation's SIMD kernel (`nasw-sse.c`) with native intrinsics,
  not an SSE→NEON translation shim.
- **Splice fast path** — SIMD inner loop skips 12 splice-state operations
  per column when no splice signals are present, matching the C scalar path
  optimization.
- **Fast approximate log2** — bit-manipulation log2 in chain scoring,
  identical to C's `mp_log2`, avoiding math library calls in the hot loop.
- **`zlib-rs`** — pure-Rust gzip backend for FASTA decompression during
  index build, with a linear unpack buffer replacing per-base closures.
- **Radix sort** — LSB radix sort for k-mer anchors, ~3–5× faster than
  comparison sort for the typical input sizes.
- **Rayon parallelism** — work-stealing thread pool for both index building
  and per-query mapping.
- **Thread-local buffer reuse** — profile matrix and SIMD scratch arrays
  reused across DP calls via thread-local storage, cutting ~1 GB of temporary
  allocations per benchmark run.

Reproduce with:

```bash
# index build
miniprot -t 4 -d human.mpi GCF_000001405.40_GRCh38.p14_genomic.fna.gz
# mapping
/usr/bin/time -p miniprot -t 4 human.mpi queries.fa > /dev/null
```

## Test

The oracle and parity tests depend on the upstream C implementation and its bundled fixtures.
They are not required to build the Rust binary.

Use either of these setups:

```bash
git submodule update --init --recursive
cargo test
```

or point `MINIPROT_C_ORACLE` at an existing upstream `miniprot` binary.

## Note

If you need the original miniprot, use the upstream project directly: [`lh3/miniprot`](https://github.com/lh3/miniprot).

If you want a Rust reimplementation of miniprot that removes much of the legacy C style and is written as Rust rather than as translated C, that is what this repository is for.
