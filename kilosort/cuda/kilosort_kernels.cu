// Optional fused CUDA kernels for kilosort.spikedetect.template_match and
// kilosort.template_matching.run_matching. Built into a shared library by build.bat
// (Windows) or build.sh (Linux) and loaded with ctypes by kilosort/cuda_kernels.py.
// Each entry point returns 0 on success, -1 if the shapes are outside what the kernels
// support (the caller then uses the PyTorch code), or a CUDA error code.
#include <cuda_runtime.h>
#include <stdint.h>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

// Bump when an exported signature changes; must match KERNELS_VERSION in cuda_kernels.py.
EXPORT int ks_version() { return 3; }

#define TPB 256
#define MAXNC 32
#define MAXS 16
#define MAXNC2 256
#define MAXNT0 64

// ---------------------------------------------------------------------------------------
// template_match
//
// For each universal template f and time t:
//   A[s,k] = sum_j weigh[s,j,f] * B[iC[j,f], k, t]       (s < S sizes, k < K temporal templates)
//   As[f,t] = max_{s,k} |A[s,k]|,  imaxs[f,t] = (1 + argmax) * sign(A[argmax]),  argmax = s*K + k
// Ties resolve to the smallest index, as torch.max does. The sum over j is two partial sums
// (first half, second half) added at the end, which reproduces the cuBLAS result of the
// PyTorch einsum bit for bit for KS4's default shapes (NC=10, S=5, K=6).

template <int NC, int S, int K>
__device__ __forceinline__ void tm_body(const float* __restrict__ B, const int* ch, const float* ws,
                                        int t, int T, float& best, int& bi, float& bval)
{
#pragma unroll
    for (int k = 0; k < K; k++) {
        float b[NC];
#pragma unroll
        for (int j = 0; j < NC; j++) b[j] = __ldg(&B[((int64_t)ch[j] * K + k) * T + t]);
#pragma unroll
        for (int s = 0; s < S; s++) {
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int j = 0; j < NC / 2; j++) a0 = fmaf(ws[s * NC + j], b[j], a0);
#pragma unroll
            for (int j = NC / 2; j < NC; j++) a1 = fmaf(ws[s * NC + j], b[j], a1);
            const float acc = a0 + a1;
            const float a = fabsf(acc);
            const int idx = s * K + k;
            if (a > best || (a == best && idx < bi)) { best = a; bi = idx; bval = acc; }
        }
    }
}

__device__ __forceinline__ void tm_body_generic(const float* __restrict__ B, const int* ch, const float* ws,
                                                int t, int T, int NC, int S, int K,
                                                float& best, int& bi, float& bval)
{
    for (int k = 0; k < K; k++) {
        for (int s = 0; s < S; s++) {
            float a0 = 0.f, a1 = 0.f;
            const int h = NC / 2;
            for (int j = 0; j < h; j++) a0 = fmaf(ws[s * NC + j], B[((int64_t)ch[j] * K + k) * T + t], a0);
            for (int j = h; j < NC; j++) a1 = fmaf(ws[s * NC + j], B[((int64_t)ch[j] * K + k) * T + t], a1);
            const float acc = a0 + a1;
            const float a = fabsf(acc);
            const int idx = s * K + k;
            if (a > best || (a == best && idx < bi)) { best = a; bi = idx; bval = acc; }
        }
    }
}

__global__ void __launch_bounds__(TPB) tm_kernel(
    const float* __restrict__ B, const int64_t* __restrict__ iC, const float* __restrict__ w,
    float* __restrict__ As, int64_t* __restrict__ imaxs, int K, int T, int NC, int F, int S)
{
    const int f = blockIdx.x;  // f fastest: concurrent blocks share a time chunk of B in L2
    const int t = blockIdx.y * TPB + threadIdx.x;
    __shared__ int ch[MAXNC];
    __shared__ float ws[MAXS * MAXNC];
    for (int i = threadIdx.x; i < NC; i += TPB) ch[i] = (int)iC[(int64_t)i * F + f];
    for (int i = threadIdx.x; i < S * NC; i += TPB) ws[i] = w[(int64_t)i * F + f];  // w is (S, NC, F)
    __syncthreads();
    if (t >= T) return;

    float best = -1.f, bval = 0.f;
    int bi = 0;
    if (NC == 10 && S == 5 && K == 6) tm_body<10, 5, 6>(B, ch, ws, t, T, best, bi, bval);
    else tm_body_generic(B, ch, ws, t, T, NC, S, K, best, bi, bval);
    const int64_t sgn = (bval > 0.f) - (bval < 0.f);
    As[(int64_t)f * T + t] = best;
    imaxs[(int64_t)f * T + t] = (int64_t)(1 + bi) * sgn;
}

// Amaxs[f,t] = max_j As[iC2[j,f], t], computed only within nt0 samples of a point with
// As[f,t] > Th. Elsewhere it is set to 0: template_match max-pools Amaxs over +-nt0 and only
// compares it at points with As > Th, so those values are never read.
__global__ void __launch_bounds__(TPB) nbmax_kernel(
    const float* __restrict__ As, const int64_t* __restrict__ iC2, float* __restrict__ Amaxs,
    int T, int NC2, int F, float Th, int nt0)
{
    const int f = blockIdx.x;
    const int t0 = blockIdx.y * TPB;
    const int t = t0 + threadIdx.x;
    __shared__ float row[TPB + 2 * MAXNT0];
    __shared__ int nb[MAXNC2];
    for (int i = threadIdx.x; i < TPB + 2 * nt0; i += TPB) {
        const int tt = t0 - nt0 + i;
        row[i] = (tt >= 0 && tt < T) ? As[(int64_t)f * T + tt] : 0.f;
    }
    for (int i = threadIdx.x; i < NC2; i += TPB) nb[i] = (int)iC2[(int64_t)i * F + f];
    __syncthreads();
    if (t >= T) return;

    bool need = false;
    for (int d = 0; d <= 2 * nt0; d++) need |= row[threadIdx.x + d] > Th;
    float m = 0.f;
    if (need) {
        m = __int_as_float(0xff800000);  // -inf
        for (int j = 0; j < NC2; j++) m = fmaxf(m, As[(int64_t)nb[j] * T + t]);
    }
    Amaxs[(int64_t)f * T + t] = m;
}

EXPORT int ks_template_match(const float* B, const int64_t* iC, const float* w, float* As,
                             int64_t* imaxs, const int64_t* iC2, float* Amaxs, int K, int T,
                             int NC, int F, int S, int NC2, float Th, int nt0)
{
    if (NC > MAXNC || S > MAXS || NC2 > MAXNC2 || nt0 > MAXNT0) return -1;
    if ((T + TPB - 1) / TPB > 65535) return -1;
    dim3 grid(F, (T + TPB - 1) / TPB);
    tm_kernel<<<grid, TPB>>>(B, iC, w, As, imaxs, K, T, NC, F, S);
    nbmax_kernel<<<grid, TPB>>>(As, iC2, Amaxs, T, NC2, F, Th, nt0);
    return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------------------
// run_matching: one peel step as a few kernels, with one host sync (reading the peak count).
// Every float operation matches the PyTorch code it replaces, so results are bit-identical,
// except ks_matching_xres_fast, which computes the residual's template projection itself
// instead of using cuBLAS (only used when settings['fast_matching'] is True).
//
// peaks:   Cmax[c] = max(Cfmax[c-nt .. c+nt]) (padded with -inf, as max_pool1d). Column c is a
//          peak if Cmax[c] > Th^2 and |Cmax[c] - Cfmax[c]| < 1e-9 (float32, as the PyTorch code).
//          out[0] = number of peaks, out[1..] = their columns in ascending order (as torch.nonzero).
// record:  for each peak i at column x: y = imax[x]; st[i] = (x, y); amps[i] = B[y,x] / nm[y];
//          th_amps[i] = sqrt(Cmax[x]); iY[i] = y.
// xres:    Xres[c, iX[i] + t - nt/2] -= amp[i] * E[c, i, t] (rounded multiply, rounded subtract),
//          with E = einsum('ijk, jl -> kil', U[iY], W) from cuBLAS (exact path).
// bupdate: B[u, iX[i] + d - nt] -= amp[i] * ctc[u, iY[i], d] for d in [0, 2nt], over the units u
//          whose ctc column for iY[i] is non-zero (CSC: rows[ptr[y]..ptr[y+1]), vals[e, d]); the
//          other units' entries are exact zeros and would leave B unchanged.
// colmax:  for each column c within nt of a peak (skipping the zeroed edges), over units u:
//          Cf = relu(B[u,c])^2 / nm[u]; Cfmax[c] = max_u Cf, imax[c] = first argmax.
//          Overlapping windows make several threads compute the same column; they write the
//          same values.
// Peak groups passed to xres and bupdate are KS4's interleaved groups (start j, stride n),
// whose windows don't overlap, so their writes never race.

#define MAXNTM 4096

__global__ void __launch_bounds__(TPB) peaks_kernel(
    const float* __restrict__ Cfmax, int NT, int nt, float thr, uint8_t* __restrict__ flags)
{
    extern __shared__ float row[];  // TPB + 2*nt
    const int c0 = blockIdx.x * TPB;
    for (int i = threadIdx.x; i < TPB + 2 * nt; i += TPB) {
        const int c = c0 - nt + i;
        row[i] = (c >= 0 && c < NT) ? Cfmax[c] : __int_as_float(0xff800000);
    }
    __syncthreads();
    const int c = c0 + threadIdx.x;
    if (c >= NT) return;
    float m = row[threadIdx.x];
    for (int d = 1; d <= 2 * nt; d++) m = fmaxf(m, row[threadIdx.x + d]);
    const float v = row[threadIdx.x + nt];
    flags[c] = (m > thr) && (fabsf(__fsub_rn(m, v)) < 1e-9f);
}

// Single block: ordered compaction of the flags.
__global__ void compact_kernel(const uint8_t* __restrict__ flags, int NT, int64_t* __restrict__ out)
{
    __shared__ int counts[1024];
    const int tid = threadIdx.x;
    const int chunk = (NT + 1023) / 1024;
    const int a = tid * chunk, b = min(NT, a + chunk);
    int n = 0;
    for (int c = a; c < b; c++) n += flags[c];
    counts[tid] = n;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {   // inclusive scan
        const int x = tid >= off ? counts[tid - off] : 0;
        __syncthreads();
        counts[tid] += x;
        __syncthreads();
    }
    int pos = counts[tid] - n;
    for (int c = a; c < b; c++) if (flags[c]) out[1 + pos++] = c;
    if (tid == 1023) out[0] = counts[1023];
}

__global__ void record_kernel(const float* __restrict__ B, const float* __restrict__ nm,
                              const int64_t* __restrict__ imax, const float* __restrict__ Cfmax,
                              const int64_t* __restrict__ iX, int nsp, int NT, int nt,
                              int64_t* __restrict__ st, float* __restrict__ amps,
                              float* __restrict__ th_amps, int64_t* __restrict__ iY)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsp) return;
    const int64_t x = iX[i];
    const int64_t y = imax[x];
    float m = __int_as_float(0xff800000);
    const int64_t lo = x - nt > 0 ? x - nt : 0;
    const int64_t hi = x + nt < NT - 1 ? x + nt : NT - 1;
    for (int64_t c = lo; c <= hi; c++) m = fmaxf(m, Cfmax[c]);
    st[2 * (int64_t)i] = x;
    st[2 * (int64_t)i + 1] = y;
    amps[i] = __fdiv_rn(B[y * NT + x], nm[y]);
    th_amps[i] = __fsqrt_rn(m);
    iY[i] = y;
}

// E is (C, ng, ntw) contiguous; group spike q is peak j + q*n (pointers already offset by j).
__global__ void xres_kernel(float* __restrict__ Xres, const float* __restrict__ E,
                            const int64_t* __restrict__ iX, const float* __restrict__ amp,
                            int ng, int stride, int C, int NT, int ntw)
{
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (c, i, t), t fastest
    if (q >= (int64_t)C * ng * ntw) return;
    const int t = (int)(q % ntw);
    const int i = (int)((q / ntw) % ng);
    const int c = (int)(q / ((int64_t)ntw * ng));
    const int64_t col = iX[(int64_t)i * stride] + t - ntw / 2;
    const float v = __fmul_rn(amp[(int64_t)i * stride], E[q]);
    float* p = &Xres[(int64_t)c * NT + col];
    *p = __fsub_rn(*p, v);
}

// Fast path: E[c, i, t] = sum_p U[iY_i, p, c] * W[p, t] computed here (sequential FMA),
// so it can differ from cuBLAS at float rounding.
__global__ void xres_fast_kernel(float* __restrict__ Xres, const float* __restrict__ U,
                                 const float* __restrict__ W, const int64_t* __restrict__ iX,
                                 const int64_t* __restrict__ iY, const float* __restrict__ amp,
                                 int ng, int stride, int C, int NT, int ntw, int npc)
{
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (c, i, t), t fastest
    if (q >= (int64_t)C * ng * ntw) return;
    const int t = (int)(q % ntw);
    const int i = (int)((q / ntw) % ng);
    const int c = (int)(q / ((int64_t)ntw * ng));
    const int64_t y = iY[(int64_t)i * stride];
    float e = 0.f;
    for (int p = 0; p < npc; p++) e = fmaf(U[(y * npc + p) * C + c], W[p * ntw + t], e);
    const int64_t col = iX[(int64_t)i * stride] + t - ntw / 2;
    const float v = __fmul_rn(amp[(int64_t)i * stride], e);
    float* x = &Xres[(int64_t)c * NT + col];
    *x = __fsub_rn(*x, v);
}

__global__ void bupdate_kernel(float* __restrict__ B, const int64_t* __restrict__ ptr,
                               const int64_t* __restrict__ rows, const float* __restrict__ vals,
                               const int64_t* __restrict__ iX, const int64_t* __restrict__ iY,
                               const float* __restrict__ amp, int ng, int stride, int NT, int nt)
{
    const int k = blockIdx.y;                                          // k-th non-zero unit
    const int W = 2 * nt + 1;
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (spike, d), d fastest
    if (q >= (int64_t)ng * W) return;
    const int64_t i = (q / W) * stride;
    const int64_t e = ptr[iY[i]] + k;
    if (e >= ptr[iY[i] + 1]) return;
    const int d = (int)(q % W);
    const int64_t col = iX[i] + d - nt;
    const float v = __fmul_rn(amp[i], vals[e * W + d]);
    float* p = &B[rows[e] * NT + col];
    *p = __fsub_rn(*p, v);
}

__global__ void colmax_kernel(const float* __restrict__ B, const float* __restrict__ nm,
                              const int64_t* __restrict__ iX, int nsp, int NU, int NT, int nt,
                              float* __restrict__ Cfmax, int64_t* __restrict__ imax)
{
    const int W = 2 * nt + 1;
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= (int64_t)nsp * W) return;
    const int64_t c = iX[q / W] + (q % W) - nt;
    if (c < nt || c >= NT - nt) return;  // these columns are zeroed and never change
    float best = -1.f;
    int bi = 0;
    for (int u = 0; u < NU; u++) {
        const float r = fmaxf(B[(int64_t)u * NT + c], 0.f);
        const float v = __fdiv_rn(__fmul_rn(r, r), nm[u]);
        if (v > best) { best = v; bi = u; }
    }
    Cfmax[c] = best;
    imax[c] = bi;
}

EXPORT int ks_matching_peaks(const float* Cfmax, int NT, int nt, float thr, uint8_t* flags, int64_t* out)
{
    if (nt > MAXNTM) return -1;
    const size_t shmem = (TPB + 2 * nt) * sizeof(float);
    peaks_kernel<<<(NT + TPB - 1) / TPB, TPB, shmem>>>(Cfmax, NT, nt, thr, flags);
    compact_kernel<<<1, 1024>>>(flags, NT, out);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_record(const float* B, const float* nm, const int64_t* imax, const float* Cfmax,
                              const int64_t* iX, int nsp, int NT, int nt, int64_t* st, float* amps,
                              float* th_amps, int64_t* iY)
{
    if (nsp == 0) return 0;
    record_kernel<<<(nsp + 255) / 256, 256>>>(B, nm, imax, Cfmax, iX, nsp, NT, nt, st, amps, th_amps, iY);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_xres(float* Xres, const float* E, const int64_t* iX, const float* amp,
                            int ng, int stride, int C, int NT, int ntw)
{
    if (ng == 0) return 0;
    const int64_t n = (int64_t)C * ng * ntw;
    xres_kernel<<<(unsigned)((n + 255) / 256), 256>>>(Xres, E, iX, amp, ng, stride, C, NT, ntw);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_xres_fast(float* Xres, const float* U, const float* W, const int64_t* iX,
                                 const int64_t* iY, const float* amp, int ng, int stride, int C,
                                 int NT, int ntw, int npc)
{
    if (ng == 0) return 0;
    const int64_t n = (int64_t)C * ng * ntw;
    xres_fast_kernel<<<(unsigned)((n + 255) / 256), 256>>>(Xres, U, W, iX, iY, amp, ng, stride, C, NT, ntw, npc);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_bupdate(float* B, const int64_t* ptr, const int64_t* rows, const float* vals,
                               const int64_t* iX, const int64_t* iY, const float* amp, int ng, int stride,
                               int kmax, int NT, int nt)
{
    if (ng == 0 || kmax == 0) return 0;
    if (kmax > 65535) return -1;
    const int W = 2 * nt + 1;
    dim3 grid((unsigned)(((int64_t)ng * W + 255) / 256), kmax);
    bupdate_kernel<<<grid, 256>>>(B, ptr, rows, vals, iX, iY, amp, ng, stride, NT, nt);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_colmax(const float* B, const float* nm, const int64_t* iX, int nsp,
                              int NU, int NT, int nt, float* Cfmax, int64_t* imax)
{
    if (nsp == 0) return 0;
    const int64_t n = (int64_t)nsp * (2 * nt + 1);
    colmax_kernel<<<(unsigned)((n + 127) / 128), 128>>>(B, nm, iX, nsp, NU, NT, nt, Cfmax, imax);
    return (int)cudaGetLastError();
}
