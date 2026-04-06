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
