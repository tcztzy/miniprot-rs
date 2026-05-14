# Project Principles

- This project pursues extreme performance. Prefer changes that reduce CPU time, memory traffic, allocations, copies, I/O overhead, and unnecessary abstraction.
- This project pursues an extreme Occam's razor. Prefer the simplest correct design, the fewest moving parts, and the smallest code that preserves clarity and performance.
- Do not add layers, helpers, indirection, or compatibility code unless they are clearly justified by measured need, correctness, or maintainability.
- When two designs are both correct, prefer the one that is faster, smaller, and easier to reason about.
- Performance baseline is the previous Rust version, not the C oracle. The Rust port already beats C on index build (28.7%) and map (3.3%) on full GRCh38. Regressions are measured against the last Rust commit, not against C. The C oracle remains useful for correctness cross-checking but is no longer the speed target.
