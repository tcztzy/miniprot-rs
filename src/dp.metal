#include <metal_stdlib>
using namespace metal;

// Must match Rust DpParams (16-byte aligned, 64 bytes total)
struct DpParams {
    uint nas_offset;
    uint aas_offset;
    uint donor_offset;
    uint acceptor_offset;
    uint nl;
    uint al;
    int  go;
    int  ge;
    int  io;
    int  fs;
    int  goe;
    int  end_bonus;
    int  flag;
    uint slen;
    uint _pad;
};

struct DpResult {
    int  score;
    int  nt_len;
    int  aa_len;
    int  _pad;
};

constant int NEG_INF = -536870912; // i32::MIN / 4
constant uint AA_STOP = 20;
constant uint AA_AMBI = 21;

// BLOSUM62 matrix: 22 x 22
constant char BLOSUM62[22][22] = {
    { 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0,-1,-4},
    {-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-4},
    {-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3,-1,-4},
    {-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3,-1,-4},
    { 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-1,-4},
    {-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2,-1,-4},
    {-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2,-1,-4},
    { 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3,-1,-4},
    {-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3,-1,-4},
    {-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3,-1,-4},
    {-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1,-1,-4},
    {-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2,-1,-4},
    {-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1,-1,-4},
    {-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1,-1,-4},
    {-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2,-1,-4},
    { 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2,-1,-4},
    { 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0,-1,-4},
    {-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3,-1,-4},
    {-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1,-1,-4},
    { 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,-1,-4},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4},
    {-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1},
};

// Thread-local DP matrix: 4 rows of up to MAX_AL + 1 elements.
// Each row needs slen+1 elements where slen <= (MAX_AL + 7) / 8.
// For scalar DP (no SIMD), we just need al+1 columns.
constant uint MAX_AL = 256;

kernel void dp_batch(
    device const uchar* nas_buf      [[buffer(0)]],
    device const uchar* aas_buf      [[buffer(1)]],
    device const DpParams* params    [[buffer(2)]],
    device DpResult* results         [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 0) { /* bounds-checked by caller */ }
    DpParams p = params[tid];
    if (p.nl < 2 || p.al == 0) {
        results[tid] = DpResult{0, 0, 0, 0};
        return;
    }

    device const uchar* nas = nas_buf + p.nas_offset;
    device const uchar* aas = aas_buf + p.aas_offset;
    int al = (int)p.al;
    int nl = (int)p.nl;
    bool is_ext = (p.flag & 6) != 0;

    // Allocate DP rows from thread-local stack (scalar DP, no SIMD).
    // Use arrays sized for MAX_AL. Only use al+1 elements.
    // Metal doesn't allow variable-length stack arrays, so use MAX_AL.
    // For safety, fall back if al exceeds MAX_AL.
    if (al > (int)MAX_AL) {
        results[tid] = DpResult{-1, 0, 0, 0};
        return;
    }

    // Scalar DP arrays
    int h_prev3[MAX_AL + 1];
    int h_prev2[MAX_AL + 1];
    int h_prev1[MAX_AL + 1];
    int h_cur[MAX_AL + 1];
    int d_prev3[MAX_AL + 1];
    int d_prev2[MAX_AL + 1];
    int d_prev1[MAX_AL + 1];
    int d_cur[MAX_AL + 1];

    // Initialize with NEG_INF
    for (int j = 0; j <= al; j++) {
        h_prev3[j] = NEG_INF;
        h_prev2[j] = NEG_INF;
        h_prev1[j] = NEG_INF;
        d_prev3[j] = NEG_INF;
        d_prev2[j] = NEG_INF;
        d_prev1[j] = NEG_INF;
    }

    h_prev3[0] = 0;
    h_prev2[0] = -p.fs;
    h_prev1[0] = -p.fs;

    int max_sc = NEG_INF;
    int max_sc_log = NEG_INF;
    int max_i = -1;
    int h_best[MAX_AL + 1];

    const int pen_len = al * 3;

    for (int i = 2; i < nl; i++) {
        // Reset current rows
        for (int j = 0; j <= al; j++) {
            h_cur[j] = NEG_INF;
            d_cur[j] = NEG_INF;
        }

        int gei = (nas[i] == AA_STOP) ? p.fs : p.ge;

        int open_d0 = h_prev3[0] - p.go;
        int ext_d0 = d_prev3[0];
        int d0 = max(open_d0, ext_d0) - gei;
        d_cur[0] = d0;
        h_cur[0] = max(d0, max(h_prev1[0] - p.fs, h_prev2[0] - p.fs));

        int i_state = NEG_INF;
        int row_max = NEG_INF;

        for (int j = 0; j < al; j++) {
            int col = j + 1;
            int best = h_prev3[j] + (int)BLOSUM62[nas[i]][aas[j]];
            int state = 0;

            int open_i = h_cur[j] - p.go;
            int ext_i = i_state;
            int t = max(open_i, ext_i) - p.ge;
            i_state = t;
            if (t > best) { best = t; state = 1; }

            int open_d = h_prev3[col] - p.go;
            int ext_d = d_prev3[col];
            t = max(open_d, ext_d) - gei;
            d_cur[col] = t;
            if (t > best) { best = t; state = 2; }

            // Frameshift paths
            t = h_prev1[j] - p.fs;
            if (t > best) { best = t; state = 6; }
            t = h_prev2[j] - p.fs;
            if (t > best) { best = t; state = 7; }
            t = h_prev1[col] - p.fs;
            if (t > best) { best = t; state = 8; }
            t = h_prev2[col] - p.fs;
            if (t > best) { best = t; state = 9; }

            h_cur[col] = best;
            row_max = max(row_max, best);
        }

        if (is_ext) {
            int end_sc = h_cur[al] + p.end_bonus;
            int tmp_sc = max(row_max, end_sc);

            // Fast approximate log2 for penalty
            int len_pen = 0;
            int row_offset = i - pen_len;
            if (row_offset >= 2) {
                float xf = (float)row_offset;
                int bits = as_type<int>(xf);
                float log_2 = (float)(((bits >> 23) & 255) - 128);
                bits &= ~(255 << 23);
                bits += 127 << 23;
                float z = as_type<float>(bits);
                log_2 += (-0.34484843f * z + 2.02466578f) * z - 0.67487759f;
                len_pen = (int)(0.5f * log_2 + 0.5f);
            }

            int tmp_sc_log = tmp_sc - len_pen;
            if (tmp_sc_log > max_sc_log) {
                max_sc = tmp_sc;
                max_sc_log = tmp_sc_log;
                max_i = i;
                for (int j = 0; j <= al; j++) {
                    h_best[j] = h_cur[j];
                }
            }
            if (max_sc_log - tmp_sc_log > 100) { // xdrop
                break;
            }
        }

        // Swap rows
        for (int j = 0; j <= al; j++) {
            int tmp;
            tmp = h_prev3[j]; h_prev3[j] = h_prev2[j]; h_prev2[j] = h_prev1[j]; h_prev1[j] = h_cur[j]; h_cur[j] = tmp;
            tmp = d_prev3[j]; d_prev3[j] = d_prev2[j]; d_prev2[j] = d_prev1[j]; d_prev1[j] = d_cur[j]; d_cur[j] = tmp;
        }
    }

    if (is_ext) {
        int best_aa = 0;
        for (int j = 0; j < al; j++) {
            int sc = h_best[j + 1];
            if (j == al - 1) sc += p.end_bonus;
            if (sc == max_sc) {
                best_aa = j + 1;
                break;
            }
        }
        results[tid] = DpResult{max_sc, max_i + 1, best_aa, 0};
    } else {
        int score = h_prev1[al];
        results[tid] = DpResult{score, nl, al, 0};
    }
}
