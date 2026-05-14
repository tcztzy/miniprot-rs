// GPU-accelerated DP via WGSL compute shader.
// One thread per DP call — scalar DP with stack-allocated arrays.
// Note: WGSL has no u8/i8. Data is stored as u32/i32 (1 byte per element).

struct DpParams {
    nas_offset: u32,
    aas_offset: u32,
    nl: u32,
    al: u32,
    go: i32,
    ge: i32,
    io: i32,
    fs: i32,
    goe: i32,
    end_bonus: i32,
    flag: i32,
    slen: u32,
}

struct DpResult {
    score: i32,
    nt_len: i32,
    aa_len: i32,
}

const NEG_INF: i32 = -536870912;
const AA_STOP: u32 = 20u;
const MAX_AL: u32 = 128u;

@group(0) @binding(0) var<storage, read> nas_buf: array<u32>;
@group(0) @binding(1) var<storage, read> aas_buf: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<DpParams>;
@group(0) @binding(3) var<storage, read_write> results: array<DpResult>;
@group(0) @binding(4) var<storage, read> score_matrix: array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let p = params[tid];

    if (p.nl < 2u || p.al == 0u) {
        results[tid] = DpResult(0, 0, 0);
        return;
    }

    let al = i32(p.al);
    let nl = i32(p.nl);
    let is_ext = (p.flag & 6) != 0;

    if (al > i32(MAX_AL)) {
        results[tid] = DpResult(-1, 0, 0);
        return;
    }

    var h_prev3: array<i32, 257>;
    var h_prev2: array<i32, 257>;
    var h_prev1: array<i32, 257>;
    var h_cur:   array<i32, 257>;
    var d_prev3: array<i32, 257>;
    var d_prev2: array<i32, 257>;
    var d_prev1: array<i32, 257>;
    var d_cur:   array<i32, 257>;

    for (var j: i32 = 0; j <= al; j++) {
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

    var max_sc = NEG_INF;
    var max_sc_log = NEG_INF;
    var max_i: i32 = -1;
    var h_best: array<i32, 257>;

    let pen_len = al * 3;

    for (var i: i32 = 2; i < nl; i++) {
        for (var j: i32 = 0; j <= al; j++) {
            h_cur[j] = NEG_INF;
            d_cur[j] = NEG_INF;
        }

        // nas/aas are stored as u32, each element is a single byte value
        let nt_aa = nas_buf[p.nas_offset + u32(i)];
        var gei = p.ge;
        if (nt_aa == AA_STOP) {
            gei = p.fs;
        }

        var open_d0 = h_prev3[0] - p.go;
        var ext_d0 = d_prev3[0];
        var d0 = max(open_d0, ext_d0) - gei;
        d_cur[0] = d0;
        h_cur[0] = max(d0, max(h_prev1[0] - p.fs, h_prev2[0] - p.fs));

        var i_state = NEG_INF;
        var row_max = NEG_INF;

        for (var j: i32 = 0; j < al; j++) {
            let col = j + 1;
            // score_matrix is 22x22, row-major i32 (each element is a byte widened to i32)
            let mat_score = score_matrix[nt_aa * 22u + aas_buf[p.aas_offset + u32(j)]];
            var best = h_prev3[j] + mat_score;

            let open_i = h_cur[j] - p.go;
            let ext_i = i_state;
            var t = max(open_i, ext_i) - p.ge;
            i_state = t;
            if (t > best) { best = t; }

            let open_d = h_prev3[col] - p.go;
            let ext_d = d_prev3[col];
            t = max(open_d, ext_d) - gei;
            d_cur[col] = t;
            if (t > best) { best = t; }

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
            let end_sc = h_cur[al] + p.end_bonus;
            var tmp_sc = max(row_max, end_sc);

            var len_pen: i32 = 0;
            let row_offset = i - pen_len;
            if (row_offset >= 2) {
                let xf = f32(row_offset);
                let bits = bitcast<u32>(xf);
                var log_2 = f32(i32((bits >> 23u) & 255u) - 128);
                var zb = bits;
                zb &= ~(255u << 23u);
                zb += 127u << 23u;
                let z = bitcast<f32>(zb);
                log_2 += (-0.34484843 * z + 2.02466578) * z - 0.67487759;
                len_pen = i32(0.5 * log_2 + 0.5);
            }

            let tmp_sc_log = tmp_sc - len_pen;
            if (tmp_sc_log > max_sc_log) {
                max_sc = tmp_sc;
                max_sc_log = tmp_sc_log;
                max_i = i;
                for (var j: i32 = 0; j <= al; j++) {
                    h_best[j] = h_cur[j];
                }
            }
            if (max_sc_log - tmp_sc_log > 100) {
                break;
            }
        }

        for (var j: i32 = 0; j <= al; j++) {
            var tmp: i32;
            tmp = h_prev3[j]; h_prev3[j] = h_prev2[j]; h_prev2[j] = h_prev1[j]; h_prev1[j] = h_cur[j]; h_cur[j] = tmp;
            tmp = d_prev3[j]; d_prev3[j] = d_prev2[j]; d_prev2[j] = d_prev1[j]; d_prev1[j] = d_cur[j]; d_cur[j] = tmp;
        }
    }

    if (is_ext) {
        var best_aa: i32 = 0;
        for (var j: i32 = 0; j < al; j++) {
            var sc = h_best[j + 1];
            if (j == al - 1) {
                sc += p.end_bonus;
            }
            if (sc == max_sc) {
                best_aa = j + 1;
                break;
            }
        }
        results[tid] = DpResult(max_sc, max_i + 1, best_aa);
    } else {
        results[tid] = DpResult(h_prev1[al], nl, al);
    }
}
