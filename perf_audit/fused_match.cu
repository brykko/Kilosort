// Fused kernels for kilosort.template_matching.run_matching (one peel step).
//
// bupdate: B[u, iX[i] + d - nt] -= amp[i] * ctc[u, iY[i], d]     for d in [0, 2nt], all units u
//          Same float ops as KS4 (one rounded multiply, one rounded subtract; no FMA).
//          Caller passes one of KS4's j::2 halves, whose windows don't overlap, so there are no races.
// colmax:  for each column c in cols (skipping the zeroed edges), over units u:
//          Cf = relu(B[u,c])^2 / nm[u];  Cfmax[c] = max_u Cf,  imax[c] = first argmax
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void bupdate_kernel(float* __restrict__ B, const float* __restrict__ ctc,
                               const int64_t* __restrict__ iX, const int64_t* __restrict__ iY,
                               const float* __restrict__ amp, int nsp, int NU, int NT, int nt)
{
    const int u = blockIdx.y;
    const int W = 2 * nt + 1;
    const int64_t q = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;   // (spike, d) flattened, d fastest
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
    if (c < nt || c >= NT - nt) return;          // KS4 zeroes these columns; they never change
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

extern "C" __declspec(dllexport) int fused_bupdate(float* B, const float* ctc, const int64_t* iX,
                                                   const int64_t* iY, const float* amp,
                                                   int nsp, int NU, int NT, int nt)
{
    if (nsp == 0) return 0;
    const int W = 2 * nt + 1;
    dim3 grid((unsigned)(((int64_t)nsp * W + 255) / 256), NU);
    bupdate_kernel<<<grid, 256>>>(B, ctc, iX, iY, amp, nsp, NU, NT, nt);
    return (int)cudaGetLastError();
}

extern "C" __declspec(dllexport) int fused_colmax(const float* B, const float* nm, const int64_t* cols,
                                                  int ncols, int NU, int NT, int nt,
                                                  float* Cfmax, int64_t* imax)
{
    if (ncols == 0) return 0;
    colmax_kernel<<<(ncols + 127) / 128, 128>>>(B, nm, cols, ncols, NU, NT, nt, Cfmax, imax);
    return (int)cudaGetLastError();
}
