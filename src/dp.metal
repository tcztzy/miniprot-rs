#include <metal_stdlib>
using namespace metal;

struct DpParams {
    uint nas_offset; uint aas_offset; uint nl; uint al;
    int go, ge, io, fs, goe, end_bonus, flag; uint slen;
};
struct DpResult { int score; int nt_len; int aa_len; };

constant short NEG = -16384;
constant uint AA_STOP = 20;
constant uint max_al = 128;

kernel void dp_batch(
    device const uchar* nas_buf      [[buffer(0)]],
    device const uchar* aas_buf      [[buffer(1)]],
    device const DpParams* params    [[buffer(2)]],
    device DpResult* results         [[buffer(3)]],
    device const char* mat           [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    DpParams p = params[tid];
    if (p.nl < 2 || p.al == 0) { results[tid]=DpResult{0,0,0}; return; }

    device const uchar* nas = nas_buf + p.nas_offset;
    device const uchar* aas = aas_buf + p.aas_offset;
    int al=(int)p.al, nl=(int)p.nl;
    bool is_ext=(p.flag&6)!=0;
    if (al>(int)max_al){results[tid]=DpResult{-1,0,0};return;}

    short h3[max_al+1],h2[max_al+1],h1[max_al+1],h0[max_al+1];
    short d3[max_al+1],d2[max_al+1],d1[max_al+1],d0[max_al+1];
    short h_best[max_al+1];

    for(int j=0;j<=al;j++){h3[j]=NEG;h2[j]=NEG;h1[j]=NEG;d3[j]=NEG;d2[j]=NEG;d1[j]=NEG;}
    h3[0]=0;h2[0]=(short)-p.fs;h1[0]=(short)-p.fs;

    int best_sc=(int)NEG, best_sc_log=(int)NEG, best_i=-1;
    const int pen_len=al*3;

    for(int i=2;i<nl;i++){
        for(int j=0;j<=al;j++){h0[j]=NEG;d0[j]=NEG;}

        int gei=(nas[i]==AA_STOP)?p.fs:p.ge;
        uchar nt_aa=nas[i];

        // Column 0: int arithmetic, store short
        int od0=(int)h3[0]-p.go, ed0=(int)d3[0];
        int dv0=max(od0,ed0)-gei;
        d0[0]=(short)dv0;
        h0[0]=(short)max(dv0,max((int)h1[0]-p.fs,(int)h2[0]-p.fs));

        int ist=(int)NEG, row_max=(int)NEG;

        for(int j=0;j<al;j++){
            int col=j+1;
            int ms=(int)mat[nt_aa*22+aas[j]];

            // All arithmetic in int, store back to short
            int best=(int)h3[j]+ms;

            int oi=(int)h0[j]-p.go;
            int ti=max(oi,ist)-p.ge;
            ist=ti;
            if(ti>best)best=ti;

            int od=(int)h3[col]-p.go;
            int td=max(od,(int)d3[col])-gei;
            d0[col]=(short)td;
            if(td>best)best=td;

            int t;
            t=(int)h1[j]-p.fs;if(t>best)best=t;
            t=(int)h2[j]-p.fs;if(t>best)best=t;
            t=(int)h1[col]-p.fs;if(t>best)best=t;
            t=(int)h2[col]-p.fs;if(t>best)best=t;

            h0[col]=(short)best;
            row_max=max(row_max,best);
        }

        if(is_ext){
            int end_sc=(int)h0[al]+p.end_bonus;
            int tmp_sc=max(row_max,end_sc), len_pen=0, row_off=i-pen_len;
            if(row_off>=2){
                float xf=(float)row_off;
                int bits=as_type<int>(xf);
                float log_2=(float)(((bits>>23)&255)-128);
                bits&=~(255<<23);bits+=127<<23;
                float z=as_type<float>(bits);
                log_2+=(-0.34484843f*z+2.02466578f)*z-0.67487759f;
                len_pen=(int)(0.5f*log_2+0.5f);
            }
            int tmp_sc_log=tmp_sc-len_pen;
            if(tmp_sc_log>best_sc_log){
                best_sc=tmp_sc;best_sc_log=tmp_sc_log;best_i=i;
                for(int j=0;j<=al;j++)h_best[j]=h0[j];
            }
            if(best_sc_log-tmp_sc_log>100)break;
        }

        for(int j=0;j<=al;j++){
            short t;
            t=h3[j];h3[j]=h2[j];h2[j]=h1[j];h1[j]=h0[j];h0[j]=t;
            t=d3[j];d3[j]=d2[j];d2[j]=d1[j];d1[j]=d0[j];d0[j]=t;
        }
    }

    if(is_ext){
        int best_aa=0;
        for(int j=0;j<al;j++){int sc=(int)h_best[j+1];if(j==al-1)sc+=p.end_bonus;
            if(sc==best_sc){best_aa=j+1;break;}}
        results[tid]=DpResult{best_sc,best_i+1,best_aa};
    }else{
        results[tid]=DpResult{(int)h1[al],nl,al};
    }
}
