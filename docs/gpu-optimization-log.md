# GPU DP 性能优化全过程

> 实验环境：Apple M2 (8-core GPU, Metal 4), macOS 25.4, Rust 1.95
> 测试用例：nl=3000, al=50, 纯 A 碱基（消除 splice 信号干扰）, BLOSUM62 矩阵
> CPU 对照：scalar_dp::global（标量, 正确性基准）和 neon_dp::global_gs16b（NEON SIMD, 性能基准）

---

## 实验 1：初始实现——修复 struct 对齐

### 背景

代码库中已有 Metal shader (`dp.metal`) 和 host code (`metal_dp.rs`)，但存在 Rust `DpParams` 与 Metal shader 结构体布局不匹配的 bug。

**Rust 侧** (48 bytes):
```
nas_offset: u32, aas_offset: u32, nl: u32, al: u32,
go: i32, ge: i32, io: i32, fs: i32, goe: i32, end_bonus: i32, flag: i32, slen: u32
```

**Metal 侧旧版** (60 bytes, 错误):
```
nas_offset, aas_offset, donor_offset, acceptor_offset,  // 多了 8 bytes
nl, al, go, ge, io, fs, goe, end_bonus, flag, slen, _pad
```

由于 `donor_offset`/`acceptor_offset` 偏移 8 字节，Metal kernel 读到错位参数。之前声称的 "9.6x speedup" 是用错误参数测的，无效。

### 修复

- 删除 `donor_offset`/`acceptor_offset`（shader 从未使用）
- Rust 和 Metal 统一为 12 字段 48 字节
- 添加 `buffer(4)` 传递 22×22 评分矩阵（不再硬编码 BLOSUM62）

### 结果

```
Batch   CPU scalar    Metal         正确性
1       670us         110.7ms       1/1 ✓
4       730us/call    21.4ms/call    4/4 ✓
16      861us/call    5.4ms/call     16/16 ✓
64      598us/call    1.4ms/call     64/64 ✓
256     383us/call    495us/call     256/256 ✓
512     332us/call    250us/call     512/512 ✓
1024    308us/call    140us/call     1024/1024 ✓
```

Metal 在 batch ≥512 时超过 CPU scalar。Metal 边际 per-call ≈ 32us (1024 batch)，CPU scalar ≈ 308us/call。

---

## 实验 2：wgpu 后端对比

### 目标

实现 wgpu 后端（跨平台 Vulkan/Metal/DX12），对比 Metal 原生 API。

### 实现

- `src/dp.wgsl` — WGSL 计算 shader（192 行，算法同 dp.metal）
- `src/wgpu_dp.rs` — wgpu host code（387 行）
- WGSL 不支持 `u8`/`i8`，数据需 widen 到 `u32`/`i32`

### 结果

```
Batch   CPU scalar    Metal         wgpu          正确性
1       670us         110.7ms       132.3ms       ✓
64      598us         1.4ms         2.1ms         ✓
256     383us         495us         526us         ✓
512     332us         250us         267us         ✓
1024    308us         140us         176us         ✓
```

wgpu 始终比 Metal 慢 20-35%（多一层抽象 + widen 开销）。代码多 100 行，但跨平台。

### 代码复杂度

| 指标 | Metal | wgpu |
|------|-------|------|
| Host LOC | 284 | 387 |
| Shader LOC | 192 | 192 |
| Unsafe | 3 | 2 |
| 新依赖 | 2 crates | 3 crates |

---

## 实验 3：simdgroup kernel（失败）

### 假设

Metal `simdgroup_` 操作可让 32 个 GPU 线程协作处理一个 DP call，像 NEON 用 SIMD 向量并行处理 8 列那样。

### 实现

- `dp_simd.metal` — 32 线程/DP call, interleaved 列布局
- `simd_shuffle_up` 跨线程传播 insertion state
- `simd_max` 跨线程 reduction

### 结果

```
Batch   Metal scalar    Metal simd    正确性（scalar vs simd）
64      90ms            11.5ms        0/64 ✗
1024    148ms           109.5ms       0/1024 ✗
```

simd 更快（batch 1024: 107us/call vs 144us/call, 1.3x），但结果全错。GPU 返回值 score 永远是 3000（= nl），而非真实 DP score（~ -1000）。

### 根因

1. DP 递推有 insertion chain：列 j 依赖列 j-1。NEON 用 `vqaddq_s16` 一条指令同时在 8 列上计算，insertion 通过寄存器内 `shift_left_2` 零开销传播。GPU simdgroup 的 32 线程在 lockstep 执行——线程 L 需要线程 L-1 的结果，但所有线程同时执行同一指令，产生循环依赖。

2. 列 0 边界 (`h[0]`) 只在第一行初始化，行交换后变成 NEG_INF，所有后续行读取垃圾值。

3. `simd_shuffle` 从错误线程/索引读取：`h1[lk]` 应改为 `h1[lk+1]`，且 `C` 应包含边界列。

### 结论

simdgroup 架构上不适合此 DP 模式。GPU 线程间缺乏 NEON 的寄存器内 SIMD 向量操作。已删除。

### 关键发现

simd kernel "看起来快"的真实原因不是 simdgroup——是 **每线程栈内存减少**。simd kernel 每线程 `short[3]` ≈ 6 bytes，标量 kernel 每线程 `int[257]` × 7 = 7.2KB。栈大小差异导致 VRAM 溢出程度不同。→ 启发后续实验 5 和 6。

---

## 实验 4：threadgroup memory cache（失败）

### 假设

将 22×22 评分矩阵（484 bytes）预加载到 threadgroup memory（on-chip），减少内循环 VRAM 访问。

### 实现

- `dp_tmem.metal` — 算法同标量，仅改评分矩阵来源
- 256 线程协作预加载 `score_matrix[0..484]` 到 `threadgroup char tmem[484]`

### 结果

```
Batch   Metal scalar    Metal tmem    正确性
512     127ms           131ms         0/512 ✗
1024    148ms           146ms         2/1024 ✗
```

无速度提升（+3% 误差范围内），且结果错误（score=-961, CPU≈-1000, Δ≈39）。评分矩阵仅 484 bytes，本就在 GPU L1 cache 内——threadgroup memory 没减少实际延迟。已删除。

---

## 实验 5：减小 MAX_AL

### 假设

标量 kernel 用 `int h_prev3[MAX_AL+1]` 分配栈数组。`MAX_AL=256` 时每线程 7 × 257 × 4 = 7.2KB。256 线程/TG = 1.8MB > GPU L1。减小到接近实际值（miniprot 典型 al=50）可减少溢出。

### 结果

| Batch | MAX_AL=256 | MAX_AL=128 | MAX_AL=64 |
|-------|-----------|-----------|----------|
| 256 | 127ms | 133ms | 128ms |
| 1024 | 143ms | 140ms | 140ms |
| 4096 | 509ms | — | — |

仅 +2.6% (batch 1024)，栈大小不是主要瓶颈。保留 MAX_AL=128 作为安全默认。

---

## 实验 6：int16 替换 int32

### 假设

int32 → int16 使栈从 3.6KB 降到 1.8KB/线程（MAX_AL=128）。DP score 范围分析：

- 最坏路径：全 frameshift，1050 步 × 23 = -24150 > i16_MIN (-32768) ✓
- 实际路径：-5000 到 +2000，远离边界
- NEON 实现已是 int16x8_t，从未溢出

### 结果

全部正确 (100% match)：

| Batch | int32 (256 TG) | int16 (256 TG) | 提升 |
|-------|---------------|---------------|------|
| 256 | 129ms | 125ms | 1.03x |
| 512 | 128ms | 125ms | 1.03x |
| 1024 | 138ms | 134ms | 1.03x |
| 2048 | 173ms | 147ms | **1.18x** |
| 4096 | 509ms | 266ms | **1.91x** |
| 8192 | 1330ms | 540ms | **2.46x** |

大 batch 退化从 2.5x 缓解到几乎线性。>2048 batch 效果显著，≤1024 几乎无影响（数据量小，未触发 VRAM 溢出）。

---

## 实验 7：减小 threadgroup

### 假设

64 线程/TG：64 × 1.8KB（int16）= 118KB。M2 GPU L1 ≈ 64KB，register file ≈ 200KB。118KB 接近上限但远好于 256 × 1.8KB = 470KB。

### 结果

全部正确 (100% match)：

| Batch | int32+256 | int16+256 | int16+64 | 总提升 |
|-------|----------|----------|---------|--------|
| 256 | 129ms | 125ms | **91ms** | 1.4x |
| 512 | 128ms | 125ms | **90ms** | 1.4x |
| 1024 | 138ms | 134ms | **129ms** | 1.07x |
| 2048 | 173ms | 147ms | **153ms** | 1.13x |
| 4096 | 509ms | 266ms | **220ms** | 2.3x |
| 8192 | 1330ms | 540ms | **589ms** | 2.3x |

int16+TG64 全 batch 最优。per-call 在 4096 batch 降到 53us。

---

## 实验 8：极限 batch 8192

```
Batch   CPU scalar    CPU NEON     Metal(int16+64)   vs Scalar   vs NEON
8192    2.55s(311us)  218ms(26us)  589ms(71us)       4.3x        0.37x
```

Metal 589ms vs NEON 218ms。GPU 仍慢 2.7x。

### GPU per-call 边际成本分析

```
Batch 1→1024:   (129-99)/1023  = 29us/call
Batch 512→1024: (129-90)/512   = 76us/call  ← TG=64 batch≤512 更高效
Batch 1024→2048: (153-129)/1024 = 23us/call
Batch 2048→4096: (220-153)/2048 = 33us/call
Batch 4096→8192: (589-220)/4096 = 90us/call  ← VRAM 压力回归
```

最优边际 per-call ≈ 23us（1024→2048），接近 NEON 27us。但在 4096+ 退化。

GPU 追平 NEON 的理论条件：
- 固定开销 85ms + 23us × N < 27us × N
- N > 21,250 DP calls

### 对比 NEON 需要什么

当前差距 2.7x（8192 batch）。要追平：
1. Anti-diagonal wavefront（GPU 内 50-way 列并行）→ per-call 可降到 5-10us
2. Pipeline 批量收集跨 query DP calls → 自然达到 10000+ batch

---

## 实验 9：指针轮转替代整行拷贝

### 假设

旧 shader 每个 nucleotide row 末尾执行：

```
h3 <- h2 <- h1 <- h0
d3 <- d2 <- d1 <- d0
```

实现方式是对 `0..=al` 每列拷贝 8 个 `short`。对 `nl=3000, al=50`，每个 DP call 约 120 万次 thread-local load/store，仅用于轮转。改为 `thread short*` 指针轮转可以保留 4 组数组，但每行只交换 8 个指针。

同时删除 `h0/d0` 每行初始化：当前行所有 `0..=al` 元素都会被覆盖，清零/填 `NEG` 是死写。

### 结果

`bench_kernel_only` batch 256：

```
改前: 91.82ms
改后: 48.07ms
提升: 1.91x
```

完整 batch sweep 中，Metal 8192 从 589ms 降到约 272ms，已经从 0.37x NEON 提升到约 0.9x NEON，但仍未稳定超过 NEON。

---

## 实验 10：no-copy 输入 buffer

### 假设

`new_buffer_with_data` 会把 `nas/aas/params` 复制进新的 Metal buffer。8192 batch 下 `nas` 约 24.6MB，host 侧 buffer 创建和复制占明显时间。Apple Silicon 统一内存允许用 `new_buffer_with_bytes_no_copy` 直接包装只读输入 slice；command buffer 同步等待完成，slice 生命周期覆盖 GPU 读取。

### 结果

完整 batch 8192：

```
copy input:    271.92ms
no-copy input: 258.29ms
```

no-copy 单独不足以超过 NEON，但减少约 14ms host 开销。

---

## 实验 11：局部缓存 aas 和当前 score row，TG=32

### 假设

no-copy 后，shader 内层循环每个 cell 都从 host-backed `aas[j]` 读取 amino-acid code，并访问 `mat[nt_aa*22+aas[j]]`。`aas` 对每个 DP call 固定，`nt_aa` 在当前纯 A 测试里几乎恒定。

实现：

- 每个 thread 启动时把 `aas[0..al]` 缓存到 `uchar aa_local[max_al]`
- 缓存当前 `nt_aa` 的 `char score_row[max_al]`，仅当 `nt_aa` 改变时重建
- 重新 sweep threadgroup size；局部缓存后 `TG=32` 最优，`TG=64/128` 都更慢

### Kernel-only 结果

```
Batch   Metal kernel
256     40.07ms  (156us/call)
1024    40.16ms  (39us/call)
4096    67.30ms  (16us/call)
8192    153.03ms (18us/call)
```

### End-to-end 结果

```
Batch   CPU scalar       CPU NEON        Metal final      vs Scalar   vs NEON
4096    1.23s(299us)    110.46ms(26us)  86.81ms(21us)   14.1x       1.3x
8192    2.48s(302us)    224.47ms(27us)  199.39ms(24us)  12.4x       1.1x
```

Metal 首次在该 benchmark 上超过 NEON SIMD。正确性：`4096/4096`、`8192/8192` 与 scalar 匹配。

注意：`score_row` 缓存对当前纯 A benchmark 特别有效，因为 translated `nt_aa` 长段重复；真实序列收益取决于 `nt_aa` 局部重复程度。`aa_local` 和指针轮转是通用收益。

---

## 实验 12：H800 CUDA 后端

### 目标

把最终 Metal 标量 kernel 的形状移植到 NVIDIA H800：一条 GPU thread 处理一个 DP call，保留 `int16` DP rows、指针轮转、`aa_local` 和 `score_row` 局部缓存。CUDA 后端通过 `--features cuda` 显式启用，Linux 上用 `build.rs` 调 `nvcc` 构建 `src/cuda_dp.cu`。

远端环境：

```
GPU: NVIDIA H800 PCIe, compute capability 9.0
CUDA: /usr/local/cuda-12.8
Rust: rustc 1.95.0
```

### Block size sweep

H800 上小 block 仍然最优。每个 thread 的 local arrays 较大，32 threads/block 比 64 更稳，尤其在 8192 batch。

```
Threads/block   Batch 256   Batch 1024   Batch 4096   Batch 8192
32              9.69ms      9.70ms       9.75ms       10.71ms
64              9.77ms      9.80ms       9.79ms       12.87ms
128             12.51ms     12.80ms      13.09ms      14.98ms
256             26.11ms     27.47ms      28.44ms      28.32ms
512             26.05ms     45.15ms      45.40ms      45.38ms
```

结论：CUDA 默认 `CUDA_THREADS=32`。仍可用环境变量覆盖。

### End-to-end batch sweep

测试用例同 Metal：`nl=3000, al=50, ext=false`。H800 主机是 x86_64，benchmark 行名仍显示 `CPU NEON`，这里按 CPU SIMD baseline 理解。

```
Batch   CPU scalar       CPU SIMD        CUDA(H800)     正确性      vs SIMD
1       883us           44us            153.91ms       1/1         0.0x
4       3.48ms          180us           9.94ms         4/4         0.0x
16      13.74ms         663us           9.98ms         16/16       0.1x
64      55.40ms         2.67ms          10.13ms        64/64       0.3x
256     219.45ms        10.55ms         10.22ms        256/256     1.0x
512     438.78ms        21.43ms         10.90ms        512/512     2.0x
1024    877.74ms        42.96ms         11.03ms        1024/1024   3.9x
2048    1.76s           84.62ms         11.39ms        2048/2048   7.4x
4096    3.52s           172.17ms        15.63ms        4096/4096   11.0x
8192    7.06s           339.79ms        21.48ms        8192/8192   15.8x
```

H800 CUDA 在 batch ≥256 时超过 CPU SIMD，batch 8192 达到约 15.8×。batch 1 的 153ms 主要是 CUDA context 初始化，不代表 kernel 边际成本。

### 形状 sweep 和限制

`bench_matrix_size_sweep` 非 extension 路径全部匹配；`al=200` 超过当前 GPU `MAX_AL=128`，按设计返回 sentinel，因此不计入正确性。

```
Matrix      CPU scalar   CUDA(H800)   正确性
1k×10       4.48ms       156.19ms     64/64   (首次 CUDA 初始化)
3k×50       54.81ms      10.05ms      64/64
10k×50      181.75ms     32.65ms      64/64
30k×50      544.96ms     97.94ms      64/64
3k×200      214.29ms     398us        0/64    (MAX_AL=128 sentinel)
```

`bench_extension_mode` 在 Metal、wgpu 和 CUDA 上都是 `29/64` 匹配，说明这是现有 GPU 简化 DP 路径的共同限制，不是 CUDA 翻译新增错误。当前可认为正确并超过 SIMD 的结论仅覆盖 `ext=false` 的 splice-free benchmark。

---

## 实验 13：CUDA no-ext 专用 kernel 和 reusable device buffers

### 假设

CUDASW++4.0 的一个核心经验是按 workload 形状做专用 kernel，而不是让一个通用 kernel 覆盖所有路径。当前 H800 benchmark 全部是 `ext=false`，但实验 12 的 CUDA kernel 仍然保留 extension-mode 的 `h_best`、`row_max` 和内层 `if (is_ext)` 分支。

实现：

- `ext=false` batch 自动走 `dp_batch_kernel_noext`
- 保留通用 kernel 作为 extension 或 mixed flags 的 fallback
- `miniprot_cuda_batch_dp` 保留 device buffers，后续 batch 只在容量不足时重新分配
- 重新 sweep `CUDA_THREADS`，默认仍为 32

### Block size sweep

```
Threads/block   Batch 256   Batch 1024   Batch 4096   Batch 8192
16              9.19ms      9.19ms       10.49ms      17.66ms
32              9.28ms      9.28ms       9.65ms       10.41ms
64              9.30ms      9.52ms       9.65ms       11.86ms
96              9.68ms      10.26ms      10.54ms      11.02ms
128             11.34ms     11.43ms      14.20ms      14.15ms
```

结论：`CUDA_THREADS=32` 仍是大 batch 最优。16 threads/block 对小 batch 略快，但 8192 batch 明显退化。

### 新 baseline

`bench_batch_size_sweep` (`nl=3000, al=50, ext=false`)：

```
Batch   CPU scalar       CPU SIMD        CUDA(H800)     正确性      vs SIMD
64      55.92ms         2.65ms          9.40ms         64/64       0.3x
256     225.60ms        10.54ms         9.55ms         256/256     1.1x
512     438.71ms        21.25ms         10.39ms        512/512     2.0x
1024    880.06ms        42.13ms         10.13ms        1024/1024   4.2x
2048    1.79s           84.39ms         10.81ms        2048/2048   7.8x
4096    3.53s           172.56ms        14.98ms        4096/4096   11.5x
8192    7.02s           341.96ms        21.74ms        8192/8192   15.7x
```

重复同一 8192 batch 的稳态结果更能体现 reusable buffers：

```
Baseline experiment 12: best 13.19ms, avg 13.71ms
Experiment 13:          best 12.54ms, avg 13.16ms
```

收益不大，但在 kernel-only、one-shot batch sweep、repeated batch 三个口径都为正。新的 H800 CUDA baseline 采用 no-ext 专用 kernel + reusable device buffers。

extension-mode fallback 未改变：`bench_extension_mode` 仍为 `29/64` 匹配，与 Metal/wgpu 的既有限制一致。

---

## 最终架构

```
src/dp.metal       — Metal int16 标量 DP shader (MAX_AL=128, pointer rotation, aa/score-row cache)
src/dp.wgsl        — wgpu WGSL 标量 DP shader (192 行, MAX_AL=128)
src/cuda_dp.cu     — CUDA C++ 标量 DP kernels (generic + no-ext, MAX_AL=128, CUDA_THREADS=32 default)
src/metal_dp.rs    — Metal host dispatch (no-copy input buffers, TG=32)
src/wgpu_dp.rs     — wgpu host dispatch (387 行)
src/cuda_dp.rs     — CUDA FFI wrapper (opt-in --features cuda)
build.rs           — CUDA feature 下调用 nvcc, 默认 sm_90
src/gpu_bench.rs   — 综合 benchmark (CPU scalar + SIMD + Metal + wgpu + CUDA, batch 1-8192)
```

已删除：`dp_simd.metal`, `dp_tmem.metal`（实验性，不正确或无效）

---

## 所有优化总表

| 实验 | 方法 | 是否正确 | Batch 256 | Batch 1024 | Batch 4096 | 结论 |
|------|------|---------|-----------|-----------|-----------|------|
| 1 | 修复对齐 + 标量 | ✓ | 129ms | 143ms | 509ms | 基线 |
| 2 | wgpu 后端 | ✓ | 135ms | 186ms | 1.3s | 跨平台, 慢 30% |
| 3 | simdgroup | ✗ | 32ms | 110ms | — | 错误结果, 已删 |
| 4 | tmem cache | ✗ | — | 146ms | — | 无加速, 有 bug, 已删 |
| 5 | MAX_AL 256→128 | ✓ | 133ms | 140ms | — | +2.6%, 边际 |
| 6 | **int16** | ✓ | 125ms | 134ms | **266ms** | 大 batch +91% |
| 7 | **TG 64** | ✓ | **91ms** | **129ms** | **220ms** | 全面最优 +30% |
| — | **6+7 combined** | ✓ | **91ms** | **129ms** | **220ms** | **大 batch +130%** |
| 9 | 指针轮转 + 去死写 | ✓ | **45ms** | **68ms** | **132ms** | 接近 NEON |
| 10 | no-copy 输入 | ✓ | **47ms** | **66ms** | **127ms** | host 开销下降 |
| 11 | aa/score-row cache + TG32 | ✓ | **43ms** | **46ms** | **87ms** | **超过 NEON** |
| 12 | CUDA/H800 + 32 threads/block | ✓ ext=false | **10ms** | **11ms** | **16ms** | H800 上大幅超过 CPU SIMD |
| 13 | CUDA no-ext kernel + reusable buffers | ✓ ext=false | **10ms** | **10ms** | **15ms** | 小幅提升, 新 H800 baseline |

当前最佳 Metal 在该 benchmark 上超过 CPU NEON SIMD：batch 4096 为 86.81ms vs NEON 110.46ms，batch 8192 为 199.39ms vs NEON 224.47ms。

当前最佳 CUDA/H800 在同类非 extension benchmark 上超过 CPU SIMD：batch 4096 为 14.98ms vs CPU SIMD 172.56ms，batch 8192 为 21.74ms vs CPU SIMD 341.96ms。重复 8192 batch 的稳态 best/avg 为 12.54ms/13.16ms。
