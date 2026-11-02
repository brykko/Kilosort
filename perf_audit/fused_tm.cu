// Fused replacement for the 40-chunk loop in kilosort.spikedetect.template_match.
// For each universal template f and time t:
//   A[s,k] = sum_j weigh[s,j,f] * B[iC[j,f], k, t]       (s < S sizes, k < K temporal templates)
//   As[f,t] = max_{s,k} |A[s,k]|,  imaxs[f,t] = (1 + argmax) * sign(A[argmax]),  argmax index = s*K + k
// Ties resolve to the smallest index, as torch.max does. The sum over j is two partial sums
// (first half, second half) added at the end: this reproduces cuBLAS's result bit for bit for
// KS4's default shapes (checked against einsum on the excerpt).
//
// Second kernel: Amaxs[f,t] = max_j As[iC2[j,f], t], computed only where KS4 can read it:
// within +-nt0 samples of a point with As[f,t] > Th (elsewhere Amaxs is never compared,
// since a detection needs As > Th and Amaxs is max-pooled over +-nt0 before the comparison).
#include <cuda_runtime.h>
#include <stdint.h>

#define MAXNC 32
#define MAXS 16
#define TPB 256

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
    const int f = blockIdx.x;                       // f fastest: concurrent blocks share a t-chunk of B in L2
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

#define MAXNT0 64
__global__ void __launch_bounds__(TPB) nbmax_kernel(
    const float* __restrict__ As, const int64_t* __restrict__ iC2, float* __restrict__ Amaxs,
    int T, int NC2, int F, float Th, int nt0)
{
    const int f = blockIdx.x;
    const int t0 = blockIdx.y * TPB;
    const int t = t0 + threadIdx.x;
    __shared__ float row[TPB + 2 * MAXNT0];
    __shared__ int nb[256];
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

extern "C" __declspec(dllexport) int fused_template_match(
    const float* B, const int64_t* iC, const float* w, float* As, int64_t* imaxs,
    const int64_t* iC2, float* Amaxs, int K, int T, int NC, int F, int S, int NC2, float Th, int nt0)
{
    if (NC > MAXNC || S > MAXS || NC2 > 256 || nt0 > MAXNT0) return -1;
    dim3 grid(F, (T + TPB - 1) / TPB);
    tm_kernel<<<grid, TPB>>>(B, iC, w, As, imaxs, K, T, NC, F, S);
    nbmax_kernel<<<grid, TPB>>>(As, iC2, Amaxs, T, NC2, F, Th, nt0);
    return (int)cudaGetLastError();
}
