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
// run_matching, one peel step
//
// bupdate: B[u, iX[i] + d - nt] -= amp[i] * ctc[u, iY[i], d]   for d in [0, 2nt], all units u.
//          Same float operations as the PyTorch code (rounded multiply, rounded subtract, no FMA).
//          The caller passes a group of spikes whose windows don't overlap, so there are no races.
// colmax:  for each column c in cols with nt <= c < NT - nt, over units u:
//          Cf = relu(B[u,c])^2 / nm[u];  Cfmax[c] = max_u Cf,  imax[c] = first argmax.

__global__ void bupdate_kernel(float* __restrict__ B, const float* __restrict__ ctc,
                               const int64_t* __restrict__ iX, const int64_t* __restrict__ iY,
                               const float* __restrict__ amp, int nsp, int NU, int NT, int nt)
{
    const int u = blockIdx.y;
    const int W = 2 * nt + 1;
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // (spike, d), d fastest
    if (q >= (int64_t)nsp * W) return;
    const int i = (int)(q / W), d = (int)(q % W);
    const int64_t col = iX[i] + d - nt;
    const float v = __fmul_rn(amp[i], ctc[((int64_t)u * NU + iY[i]) * W + d]);
    float* p = &B[(int64_t)u * NT + col];
    *p = __fsub_rn(*p, v);
}

__global__ void colmax_kernel(const float* __restrict__ B, const float* __restrict__ nm,
                              const int64_t* __restrict__ cols, int ncols, int NU, int NT, int nt,
                              float* __restrict__ Cfmax, int64_t* __restrict__ imax)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= ncols) return;
    const int64_t c = cols[q];
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

EXPORT int ks_matching_bupdate(float* B, const float* ctc, const int64_t* iX, const int64_t* iY,
                               const float* amp, int nsp, int NU, int NT, int nt)
{
    if (nsp == 0) return 0;
    if (NU > 65535) return -1;
    const int W = 2 * nt + 1;
    dim3 grid((unsigned)(((int64_t)nsp * W + 255) / 256), NU);
    bupdate_kernel<<<grid, 256>>>(B, ctc, iX, iY, amp, nsp, NU, NT, nt);
    return (int)cudaGetLastError();
}

EXPORT int ks_matching_colmax(const float* B, const float* nm, const int64_t* cols, int ncols,
                              int NU, int NT, int nt, float* Cfmax, int64_t* imax)
{
    if (ncols == 0) return 0;
    colmax_kernel<<<(ncols + 127) / 128, 128>>>(B, nm, cols, ncols, NU, NT, nt, Cfmax, imax);
    return (int)cudaGetLastError();
}
