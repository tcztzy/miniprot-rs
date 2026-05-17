#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

struct DpParams {
    uint32_t nas_offset, aas_offset, nl, al;
    int32_t go, ge, io, fs, goe, end_bonus, flag;
    uint32_t slen;
};

struct DpResult {
    int32_t score, nt_len, aa_len;
};

static constexpr int NEG = -16384;
static constexpr uint8_t AA_STOP = 20;
static constexpr int MAX_AL = 128;
#ifndef CUDA_THREADS
#define CUDA_THREADS 32
#endif

__device__ __forceinline__ int imax2(int a, int b) {
    return a > b ? a : b;
}

__global__ void dp_batch_kernel(
    const uint8_t* __restrict__ nas_buf,
    const uint8_t* __restrict__ aas_buf,
    const DpParams* __restrict__ params,
    DpResult* __restrict__ results,
    const int8_t* __restrict__ mat,
    size_t n
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const DpParams p = params[tid];
    if (p.nl < 2 || p.al == 0) {
        results[tid] = DpResult{0, 0, 0};
        return;
    }
    if (p.al > MAX_AL) {
        results[tid] = DpResult{-1, 0, 0};
        return;
    }

    const uint8_t* __restrict__ nas = nas_buf + p.nas_offset;
    const uint8_t* __restrict__ aas = aas_buf + p.aas_offset;
    const int al = (int)p.al;
    const int nl = (int)p.nl;
    const bool is_ext = (p.flag & 6) != 0;

    short ha[MAX_AL + 1], hb[MAX_AL + 1], hc[MAX_AL + 1], hd[MAX_AL + 1];
    short da[MAX_AL + 1], db[MAX_AL + 1], dc[MAX_AL + 1], dd[MAX_AL + 1];
    short h_best[MAX_AL + 1];
    short *h3 = ha, *h2 = hb, *h1 = hc, *h0 = hd;
    short *d3 = da, *d2 = db, *d1 = dc, *d0 = dd;
    uint8_t aa_local[MAX_AL];
    int8_t score_row[MAX_AL];
    uint8_t score_nt = 255;

    for (int j = 0; j <= al; ++j) {
        h3[j] = NEG; h2[j] = NEG; h1[j] = NEG;
        d3[j] = NEG; d2[j] = NEG; d1[j] = NEG;
    }
    for (int j = 0; j < al; ++j) aa_local[j] = aas[j];
    h3[0] = 0;
    h2[0] = (short)-p.fs;
    h1[0] = (short)-p.fs;

    int best_sc = NEG;
    int best_sc_log = NEG;
    int best_i = -1;
    const int pen_len = al * 3;

    for (int i = 2; i < nl; ++i) {
        const uint8_t nt_aa = nas[i];
        const int gei = nt_aa == AA_STOP ? p.fs : p.ge;
        if (nt_aa != score_nt) {
            for (int j = 0; j < al; ++j) {
                score_row[j] = mat[(int)nt_aa * 22 + aa_local[j]];
            }
            score_nt = nt_aa;
        }

        const int od0 = (int)h3[0] - p.go;
        const int ed0 = (int)d3[0];
        const int dv0 = imax2(od0, ed0) - gei;
        d0[0] = (short)dv0;
        h0[0] = (short)imax2(dv0, imax2((int)h1[0] - p.fs, (int)h2[0] - p.fs));

        int ist = NEG;
        int row_max = NEG;

        for (int j = 0; j < al; ++j) {
            const int col = j + 1;
            int best = (int)h3[j] + (int)score_row[j];

            const int oi = (int)h0[j] - p.go;
            const int ti = imax2(oi, ist) - p.ge;
            ist = ti;
            if (ti > best) best = ti;

            const int od = (int)h3[col] - p.go;
            const int td = imax2(od, (int)d3[col]) - gei;
            d0[col] = (short)td;
            if (td > best) best = td;

            int t = (int)h1[j] - p.fs; if (t > best) best = t;
            t = (int)h2[j] - p.fs; if (t > best) best = t;
            t = (int)h1[col] - p.fs; if (t > best) best = t;
            t = (int)h2[col] - p.fs; if (t > best) best = t;

            h0[col] = (short)best;
            if (is_ext) row_max = imax2(row_max, best);
        }

        if (is_ext) {
            const int end_sc = (int)h0[al] + p.end_bonus;
            int tmp_sc = imax2(row_max, end_sc);
            int len_pen = 0;
            const int row_off = i - pen_len;
            if (row_off >= 2) {
                const float xf = (float)row_off;
                int bits = __float_as_int(xf);
                float log_2 = (float)(((bits >> 23) & 255) - 128);
                bits &= ~(255 << 23);
                bits += 127 << 23;
                const float z = __int_as_float(bits);
                log_2 += (-0.34484843f * z + 2.02466578f) * z - 0.67487759f;
                len_pen = (int)(0.5f * log_2 + 0.5f);
            }
            const int tmp_sc_log = tmp_sc - len_pen;
            if (tmp_sc_log > best_sc_log) {
                best_sc = tmp_sc;
                best_sc_log = tmp_sc_log;
                best_i = i;
                for (int j = 0; j <= al; ++j) h_best[j] = h0[j];
            }
            if (best_sc_log - tmp_sc_log > 100) break;
        }

        short* ht = h3; h3 = h2; h2 = h1; h1 = h0; h0 = ht;
        short* dt = d3; d3 = d2; d2 = d1; d1 = d0; d0 = dt;
    }

    if (is_ext) {
        int best_aa = 0;
        for (int j = 0; j < al; ++j) {
            int sc = (int)h_best[j + 1];
            if (j == al - 1) sc += p.end_bonus;
            if (sc == best_sc) {
                best_aa = j + 1;
                break;
            }
        }
        results[tid] = DpResult{best_sc, best_i + 1, best_aa};
    } else {
        results[tid] = DpResult{(int)h1[al], nl, al};
    }
}

__global__ void dp_batch_kernel_noext(
    const uint8_t* __restrict__ nas_buf,
    const uint8_t* __restrict__ aas_buf,
    const DpParams* __restrict__ params,
    DpResult* __restrict__ results,
    const int8_t* __restrict__ mat,
    size_t n
) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const DpParams p = params[tid];
    if (p.nl < 2 || p.al == 0) {
        results[tid] = DpResult{0, 0, 0};
        return;
    }
    if (p.al > MAX_AL) {
        results[tid] = DpResult{-1, 0, 0};
        return;
    }

    const uint8_t* __restrict__ nas = nas_buf + p.nas_offset;
    const uint8_t* __restrict__ aas = aas_buf + p.aas_offset;
    const int al = (int)p.al;
    const int nl = (int)p.nl;

    short ha[MAX_AL + 1], hb[MAX_AL + 1], hc[MAX_AL + 1], hd[MAX_AL + 1];
    short da[MAX_AL + 1], db[MAX_AL + 1], dc[MAX_AL + 1], dd[MAX_AL + 1];
    short *h3 = ha, *h2 = hb, *h1 = hc, *h0 = hd;
    short *d3 = da, *d2 = db, *d1 = dc, *d0 = dd;
    uint8_t aa_local[MAX_AL];
    int8_t score_row[MAX_AL];
    uint8_t score_nt = 255;

    for (int j = 0; j <= al; ++j) {
        h3[j] = NEG; h2[j] = NEG; h1[j] = NEG;
        d3[j] = NEG; d2[j] = NEG; d1[j] = NEG;
    }
    for (int j = 0; j < al; ++j) aa_local[j] = aas[j];
    h3[0] = 0;
    h2[0] = (short)-p.fs;
    h1[0] = (short)-p.fs;

    for (int i = 2; i < nl; ++i) {
        const uint8_t nt_aa = nas[i];
        const int gei = nt_aa == AA_STOP ? p.fs : p.ge;
        if (nt_aa != score_nt) {
            for (int j = 0; j < al; ++j) {
                score_row[j] = mat[(int)nt_aa * 22 + aa_local[j]];
            }
            score_nt = nt_aa;
        }

        const int od0 = (int)h3[0] - p.go;
        const int ed0 = (int)d3[0];
        const int dv0 = imax2(od0, ed0) - gei;
        d0[0] = (short)dv0;
        h0[0] = (short)imax2(dv0, imax2((int)h1[0] - p.fs, (int)h2[0] - p.fs));

        int ist = NEG;
        for (int j = 0; j < al; ++j) {
            const int col = j + 1;
            int best = (int)h3[j] + (int)score_row[j];

            const int oi = (int)h0[j] - p.go;
            const int ti = imax2(oi, ist) - p.ge;
            ist = ti;
            if (ti > best) best = ti;

            const int od = (int)h3[col] - p.go;
            const int td = imax2(od, (int)d3[col]) - gei;
            d0[col] = (short)td;
            if (td > best) best = td;

            int t = (int)h1[j] - p.fs; if (t > best) best = t;
            t = (int)h2[j] - p.fs; if (t > best) best = t;
            t = (int)h1[col] - p.fs; if (t > best) best = t;
            t = (int)h2[col] - p.fs; if (t > best) best = t;

            h0[col] = (short)best;
        }

        short* ht = h3; h3 = h2; h2 = h1; h1 = h0; h0 = ht;
        short* dt = d3; d3 = d2; d2 = d1; d1 = d0; d0 = dt;
    }

    results[tid] = DpResult{(int)h1[al], nl, al};
}

static int launch_dp(
    const uint8_t* d_nas,
    const uint8_t* d_aas,
    const DpParams* d_params,
    DpResult* d_results,
    const int8_t* d_matrix,
    size_t n,
    bool noext
) {
    const int threads = CUDA_THREADS;
    const int blocks = (int)((n + threads - 1) / threads);
    if (noext) {
        dp_batch_kernel_noext<<<blocks, threads>>>(d_nas, d_aas, d_params, d_results, d_matrix, n);
    } else {
        dp_batch_kernel<<<blocks, threads>>>(d_nas, d_aas, d_params, d_results, d_matrix, n);
    }
    return (int)cudaGetLastError();
}

static bool all_noext(const DpParams* params, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if ((params[i].flag & 6) != 0) return false;
    }
    return true;
}

struct DeviceCache {
    uint8_t* nas = nullptr;
    uint8_t* aas = nullptr;
    DpParams* params = nullptr;
    DpResult* results = nullptr;
    int8_t* matrix = nullptr;
    size_t nas_cap = 0;
    size_t aas_cap = 0;
    size_t params_cap = 0;
    size_t results_cap = 0;
    size_t matrix_cap = 0;
};

static DeviceCache g_cache;

static cudaError_t reserve_device(void** ptr, size_t* cap, size_t bytes) {
    if (bytes <= *cap) return cudaSuccess;
    void* next = nullptr;
    cudaError_t err = cudaMalloc(&next, bytes);
    if (err != cudaSuccess) return err;
    cudaFree(*ptr);
    *ptr = next;
    *cap = bytes;
    return cudaSuccess;
}

struct PreparedBatch {
    uint8_t* nas = nullptr;
    uint8_t* aas = nullptr;
    DpParams* params = nullptr;
    DpResult* results = nullptr;
    int8_t* matrix = nullptr;
    size_t n = 0;
    bool noext = false;
};

static void destroy_prepared(PreparedBatch* batch) {
    if (!batch) return;
    cudaFree(batch->matrix);
    cudaFree(batch->results);
    cudaFree(batch->params);
    cudaFree(batch->aas);
    cudaFree(batch->nas);
    free(batch);
}

extern "C" int miniprot_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return err == cudaSuccess && count > 0 ? 1 : 0;
}

extern "C" int miniprot_cuda_batch_dp(
    const uint8_t* nas,
    size_t nas_len,
    const uint8_t* aas,
    size_t aas_len,
    const DpParams* params,
    size_t n,
    const int8_t* matrix,
    DpResult* results
) {
    cudaError_t err;
    err = reserve_device((void**)&g_cache.nas, &g_cache.nas_cap, nas_len); if (err != cudaSuccess) return (int)err;
    err = reserve_device((void**)&g_cache.aas, &g_cache.aas_cap, aas_len); if (err != cudaSuccess) return (int)err;
    err = reserve_device((void**)&g_cache.params, &g_cache.params_cap, n * sizeof(DpParams)); if (err != cudaSuccess) return (int)err;
    err = reserve_device((void**)&g_cache.results, &g_cache.results_cap, n * sizeof(DpResult)); if (err != cudaSuccess) return (int)err;
    err = reserve_device((void**)&g_cache.matrix, &g_cache.matrix_cap, 22 * 22); if (err != cudaSuccess) return (int)err;

    err = cudaMemcpy(g_cache.nas, nas, nas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(g_cache.aas, aas, aas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(g_cache.params, params, n * sizeof(DpParams), cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(g_cache.matrix, matrix, 22 * 22, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;

    int code = launch_dp(g_cache.nas, g_cache.aas, g_cache.params, g_cache.results, g_cache.matrix, n, all_noext(params, n));
    if (code == cudaSuccess) code = (int)cudaDeviceSynchronize();
    if (code == cudaSuccess) {
        code = (int)cudaMemcpy(results, g_cache.results, n * sizeof(DpResult), cudaMemcpyDeviceToHost);
    }

    return code;
}

extern "C" int miniprot_cuda_prepared_batch_create(
    const uint8_t* nas,
    size_t nas_len,
    const uint8_t* aas,
    size_t aas_len,
    const DpParams* params,
    size_t n,
    const int8_t* matrix,
    PreparedBatch** out
) {
    *out = nullptr;
    PreparedBatch* batch = (PreparedBatch*)malloc(sizeof(PreparedBatch));
    if (!batch) return (int)cudaErrorMemoryAllocation;
    batch->nas = nullptr;
    batch->aas = nullptr;
    batch->params = nullptr;
    batch->results = nullptr;
    batch->matrix = nullptr;
    batch->n = n;
    batch->noext = all_noext(params, n);

    cudaError_t err;
    err = cudaMalloc((void**)&batch->nas, nas_len); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMalloc((void**)&batch->aas, aas_len); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMalloc((void**)&batch->params, n * sizeof(DpParams)); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMalloc((void**)&batch->results, n * sizeof(DpResult)); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMalloc((void**)&batch->matrix, 22 * 22); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }

    err = cudaMemcpy(batch->nas, nas, nas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMemcpy(batch->aas, aas, aas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMemcpy(batch->params, params, n * sizeof(DpParams), cudaMemcpyHostToDevice); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }
    err = cudaMemcpy(batch->matrix, matrix, 22 * 22, cudaMemcpyHostToDevice); if (err != cudaSuccess) { destroy_prepared(batch); return (int)err; }

    *out = batch;
    return (int)cudaSuccess;
}

extern "C" int miniprot_cuda_prepared_batch_run(
    PreparedBatch* batch,
    DpResult* results
) {
    if (!batch) return (int)cudaErrorInvalidValue;
    int code = launch_dp(batch->nas, batch->aas, batch->params, batch->results, batch->matrix, batch->n, batch->noext);
    if (code == cudaSuccess) code = (int)cudaDeviceSynchronize();
    if (code == cudaSuccess) {
        code = (int)cudaMemcpy(results, batch->results, batch->n * sizeof(DpResult), cudaMemcpyDeviceToHost);
    }
    return code;
}

extern "C" void miniprot_cuda_prepared_batch_destroy(PreparedBatch* batch) {
    destroy_prepared(batch);
}

extern "C" int miniprot_cuda_bench_dispatch_only(
    const uint8_t* nas,
    size_t nas_len,
    const uint8_t* aas,
    size_t aas_len,
    const DpParams* params,
    size_t n,
    const int8_t* matrix,
    float* warmup_ms,
    float* timed_ms
) {
    uint8_t *d_nas = nullptr, *d_aas = nullptr;
    DpParams* d_params = nullptr;
    DpResult* d_results = nullptr;
    int8_t* d_matrix = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_nas, nas_len); if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void**)&d_aas, aas_len); if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void**)&d_params, n * sizeof(DpParams)); if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void**)&d_results, n * sizeof(DpResult)); if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void**)&d_matrix, 22 * 22); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(d_nas, nas, nas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(d_aas, aas, aas_len, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(d_params, params, n * sizeof(DpParams), cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;
    err = cudaMemcpy(d_matrix, matrix, 22 * 22, cudaMemcpyHostToDevice); if (err != cudaSuccess) return (int)err;

    err = cudaEventCreate(&start); if (err != cudaSuccess) return (int)err;
    err = cudaEventCreate(&stop); if (err != cudaSuccess) return (int)err;

    const bool noext = all_noext(params, n);
    launch_dp(d_nas, d_aas, d_params, d_results, d_matrix, n, noext);
    err = cudaDeviceSynchronize(); if (err != cudaSuccess) return (int)err;

    cudaEventRecord(start);
    launch_dp(d_nas, d_aas, d_params, d_results, d_matrix, n, noext);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(warmup_ms, start, stop);

    cudaEventRecord(start);
    launch_dp(d_nas, d_aas, d_params, d_results, d_matrix, n, noext);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(timed_ms, start, stop);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_matrix);
    cudaFree(d_results);
    cudaFree(d_params);
    cudaFree(d_aas);
    cudaFree(d_nas);
    return (int)cudaGetLastError();
}
