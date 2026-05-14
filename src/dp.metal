#include <metal_stdlib>
using namespace metal;

// Must match Rust DpParams exactly (48 bytes, 4-byte aligned)
struct DpParams {
    uint nas_offset;
    uint aas_offset;
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
};

struct DpResult {
    int  score;
    int  nt_len;
    int  aa_len;
};

constant int NEG_INF = -536870912; // i32::MIN / 4
constant uint AA_STOP = 20;
constant uint AA_AMBI = 21;

// Expects 22*22 matrix in row-major: score_matrix[nt_aa * 22 + aa_aa]
constant uint MATRIX_SIZE = 484;

kernel void dp_batch(
    device const uchar* nas_buf      [[buffer(0)]],
    device const uchar* aas_buf      [[buffer(1)]],
    device const DpParams* params    [[buffer(2)]],
    device DpResult* results         [[buffer(3)]],
    device const char* score_matrix  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    DpParams p = params[tid];
    if (p.nl < 2 || p.al == 0) {
        results[tid] = DpResult{0, 0, 0};
        return;
    }

    device const uchar* nas = nas_buf + p.nas_offset;
    device const uchar* aas = aas_buf + p.aas_offset;
    int al = (int)p.al;
    int nl = (int)p.nl;
    bool is_ext = (p.flag & 6) != 0;

    // Stack arrays sized for typical protein queries. Larger al → CPU fallback.
    const uint max_al = 128;
    if (al > (int)max_al) {
        results[tid] = DpResult{-1, 0, 0};
        return;
    }

    // Scalar DP arrays on stack
    int h_prev3[max_al + 1];
    int h_prev2[max_al + 1];
    int h_prev1[max_al + 1];
    int h_cur[max_al + 1];
    int d_prev3[max_al + 1];
    int d_prev2[max_al + 1];
    int d_prev1[max_al + 1];
    int d_cur[max_al + 1];

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
    int h_best[max_al + 1];

    const int pen_len = al * 3;

    for (int i = 2; i < nl; i++) {
        for (int j = 0; j <= al; j++) {
            h_cur[j] = NEG_INF;
            d_cur[j] = NEG_INF;
        }

        int gei = (nas[i] == AA_STOP) ? p.fs : p.ge;
        uchar nt_aa = nas[i];

        int open_d0 = h_prev3[0] - p.go;
        int ext_d0 = d_prev3[0];
        int d0 = max(open_d0, ext_d0) - gei;
        d_cur[0] = d0;
        h_cur[0] = max(d0, max(h_prev1[0] - p.fs, h_prev2[0] - p.fs));

        int i_state = NEG_INF;
        int row_max = NEG_INF;

        for (int j = 0; j < al; j++) {
            int col = j + 1;
            int mat_score = (int)score_matrix[nt_aa * 22 + aas[j]];
            int best = h_prev3[j] + mat_score;

            int open_i = h_cur[j] - p.go;
            int ext_i = i_state;
            int t = max(open_i, ext_i) - p.ge;
            i_state = t;
            if (t > best) { best = t; }

            int open_d = h_prev3[col] - p.go;
            int ext_d = d_prev3[col];
            t = max(open_d, ext_d) - gei;
            d_cur[col] = t;
            if (t > best) { best = t; }

            // Frameshift paths
            t = h_prev1[j] - p.fs;
            if (t > best) { best = t; }
            t = h_prev2[j] - p.fs;
            if (t > best) { best = t; }
            t = h_prev1[col] - p.fs;
            if (t > best) { best = t; }
            t = h_prev2[col] - p.fs;
            if (t > best) { best = t; }

            h_cur[col] = best;
            row_max = max(row_max, best);
        }

        if (is_ext) {
            int end_sc = h_cur[al] + p.end_bonus;
            int tmp_sc = max(row_max, end_sc);

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
            if (max_sc_log - tmp_sc_log > 100) {
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
        results[tid] = DpResult{max_sc, max_i + 1, best_aa};
    } else {
        int score = h_prev1[al];
        results[tid] = DpResult{score, nl, al};
    }
}
