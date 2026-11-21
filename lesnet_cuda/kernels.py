"""The LESNet CUDA kernels: one file, two partner layouts, same maths.

One layer, receiver i, partner j, slice k (W slices):

    z_ij = a_i[k] + b_j[k] + pf_ij . E[k] + c1[k]          a = x_r @ A,  b = x_s @ B
    v_ij = leaky(z_ij)
    w_ij = softmax over valid partners j of v_ij             (per slice)
    y_i  = sum_j w_ij v_ij                                   (soft-max combine)

An entity with no valid partner gets y = 0.  Nothing in here knows what an entity is.

Partner layouts
---------------
dense  (`les_layer`)       every valid entity is a partner of every other: pf is (B,N,N,PF),
                           mask is (B,N), validity = mask_i & mask_j & i != j.  O(N^2 W) per layer.
                           The pair-feature tile is staged in shared memory, partners-outer.

list   (`les_layer_list`)  each receiver names its own partners: nbr is (R,K) int32 holding a
                           sender ROW (or -1 for "no partner"), pf is (R,K,PF).  O(R K W) per layer.
                           This is how k-nearest and radius neighbourhoods run, and it also covers
                           receivers != senders (senders that never receive, e.g. fixed scenery).
                           With K = N-1 and every other entity listed it is the dense layer exactly
                           (tests.py asserts that).

Both walk the partners once with a running (online) softmax, so the (R, partners, W) pair tensor is
never materialised, and both write the same three saved tensors (y, running max, softmax sum) that the
backward needs.  Everything is fp32 and checked against the float64 oracle in ref.py, values and every
gradient, by tests.py and tests_sizes.py.

The packed projection: when receivers and senders are the same set, `ab = x @ [A|B]` is one (R, 2W)
matrix (a in columns [0,W), b in [W,2W)) and the backward writes the partner gradient into the same
shape, so a layer is three matmuls rather than six.

Why the knobs are what they are (measured on sm_75, GTX 1660 Ti; see TRAVEL_GUIDE.md): these kernels
are latency bound, not bandwidth bound, so everything here is instruction-level parallelism.  A block
walks TI receivers at once so TI independent softmax chains interleave, and b is loaded once per
partner rather than once per (receiver, partner).
"""
import os
import torch

SLOPE = 0.1
# Leave the packed parameter gradient as two strided halves instead of copying them out; set
# LES_CONTIG_GRAD=1 to restore the copies (only useful for A/B measuring them).
CONTIG_GRAD = os.environ.get("LES_CONTIG_GRAD") == "1"
MAXPF = 8        # upper bound on the pair-feature width (a slice's E column lives in registers)
_ext = None
_ext_error = None

_CPP = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> les_fwd(torch::Tensor ab, torch::Tensor pf, torch::Tensor E,
                                   torch::Tensor c1, torch::Tensor mask, int64_t N,
                                   int64_t TI, bool smem);
std::vector<torch::Tensor> les_bwd(torch::Tensor gy, torch::Tensor ab, torch::Tensor pf,
                                   torch::Tensor E, torch::Tensor c1, torch::Tensor mask,
                                   torch::Tensor y, torch::Tensor mx, torch::Tensor ss,
                                   int64_t N, int64_t TI, bool smem);
std::vector<torch::Tensor> les_fwd_list(torch::Tensor a, torch::Tensor b, torch::Tensor pf,
                                        torch::Tensor nbr, torch::Tensor E, torch::Tensor c1,
                                        int64_t TI);
void les_bwd_list(torch::Tensor gy, torch::Tensor a, torch::Tensor b, torch::Tensor pf,
                  torch::Tensor nbr, torch::Tensor E, torch::Tensor c1, torch::Tensor y,
                  torch::Tensor mx, torch::Tensor ss, torch::Tensor ga, torch::Tensor gb,
                  torch::Tensor gE, torch::Tensor gc1, int64_t TI);
torch::Tensor les_knn(torch::Tensor xy, torch::Tensor order, torch::Tensor cell_start,
                      torch::Tensor cell_of, torch::Tensor mask, int64_t K, int64_t N,
                      int64_t Gx, int64_t Gy, double radius);
std::vector<int64_t> les_info();
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define SLOPE 0.1f
// Upper bound on the pair-feature width.  It exists only because a slice's column of E has to live in
// registers; the width itself is a runtime value.  Raising it costs registers: measure after you do.
#define MAXPF 8

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------------------ pair-feature dots
// PF is a runtime property (which relations matter is a per-task choice), so each dot dispatches at
// runtime onto a compile-time-unrolled arm for the widths that load as one wide vector, and onto a
// MAXPF-bounded predicated loop for any other width.  Constant indices are what keep `Ek` in
// registers.  Widths 4 and 8 are multiples of 16 B and 6 of 8 B, and every offset these are used at is
// a whole number of pair vectors, so the wide loads are aligned.
__device__ __forceinline__ float pf_dot_rt(const float* __restrict__ q,
                                           const float* __restrict__ Ek, int PF) {
    if (PF == 4) {
        const float4 p = *reinterpret_cast<const float4*>(q);
        return p.x * Ek[0] + p.y * Ek[1] + p.z * Ek[2] + p.w * Ek[3];
    }
    if (PF == 6) {
        const float2* p = reinterpret_cast<const float2*>(q);
        const float2 p0 = p[0], p1 = p[1], p2 = p[2];
        return p0.x * Ek[0] + p0.y * Ek[1] + p1.x * Ek[2] + p1.y * Ek[3] + p2.x * Ek[4] + p2.y * Ek[5];
    }
    if (PF == 8) {
        const float4 p0 = *reinterpret_cast<const float4*>(q);
        const float4 p1 = *(reinterpret_cast<const float4*>(q) + 1);
        return p0.x * Ek[0] + p0.y * Ek[1] + p0.z * Ek[2] + p0.w * Ek[3]
             + p1.x * Ek[4] + p1.y * Ek[5] + p1.z * Ek[6] + p1.w * Ek[7];
    }
    float r = 0.f;
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) if (t < PF) r += q[t] * Ek[t];
    return r;
}

// All TI tile entities' pair-dot against one partner (dense kernels).  Hoists the width dispatch out
// of the per-entity work: the branch is paid once per partner.  `qstride` is the stride between tile
// entities' pair vectors.
template <int TI>
__device__ __forceinline__ void pf_dot_all(const float* __restrict__ q, int qstride, int PF,
                                           const float* __restrict__ Ek, float* __restrict__ out) {
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) out[ii] = pf_dot_rt(q + (size_t)ii * qstride, Ek, PF);
}

// Accumulate one pair vector into the E gradient.
__device__ __forceinline__ void pf_acc_rt(const float* __restrict__ q, float dz,
                                          float* __restrict__ gEk, int PF) {
    if (PF == 4) {
        const float4 p = *reinterpret_cast<const float4*>(q);
        gEk[0] += p.x * dz; gEk[1] += p.y * dz; gEk[2] += p.z * dz; gEk[3] += p.w * dz;
    } else if (PF == 8) {
        const float4 p0 = *reinterpret_cast<const float4*>(q);
        const float4 p1 = *(reinterpret_cast<const float4*>(q) + 1);
        gEk[0] += p0.x * dz; gEk[1] += p0.y * dz; gEk[2] += p0.z * dz; gEk[3] += p0.w * dz;
        gEk[4] += p1.x * dz; gEk[5] += p1.y * dz; gEk[6] += p1.z * dz; gEk[7] += p1.w * dz;
    } else {
        #pragma unroll
        for (int t = 0; t < MAXPF; ++t) if (t < PF) gEk[t] += q[t] * dz;
    }
}

// One online-softmax step.  Exactly one of the two rescaling factors is exp(0) == 1, so a single
// expf of -|m - v| yields whichever one is not 1 (MUFU is a quarter-rate pipe on sm_75).  `newmax` is
// false when the running max is unchanged; the first accepted partner takes the newmax arm with
// m = -1e30, where sc = exp(-huge) = 0 wipes the (empty) accumulators.  m starts at -1e30 rather than
// -inf so m - v never becomes inf - inf.
__device__ __forceinline__ void softmax_step(float v, bool valid, float& m, float& s, float& acc) {
    const float d = m - v;
    const float ad = __expf(-fabsf(d));
    const bool newmax = valid && (d < 0.f);
    const float sc = newmax ? ad : 1.f;
    const float e = valid ? (newmax ? 1.f : ad) : 0.f;
    s = s * sc + e;
    acc = acc * sc + e * v;
    if (newmax) m = v;
}

// ===================================================================================== DENSE layout
// Forward, partners outer / entities inner: b[.,j,k] is loaded once per (block, j) and reused by all
// TI entities, and with SMEM the tile's whole pair-feature block (TI*N*PF floats) is staged in shared
// memory so the inner loop issues no global loads at all.  Each entity keeps its own online-softmax
// recurrence and the TI recurrences interleave, which is the ILP wanted anyway.
template <int TI, bool SMEM>
__global__ __launch_bounds__(1024) void k_fwd(
        const float* __restrict__ a, int astride, const float* __restrict__ b, int bstride,
        const float* __restrict__ pf, const float* __restrict__ E, const float* __restrict__ c1,
        const unsigned char* __restrict__ mask, float* __restrict__ y, float* __restrict__ mx,
        float* __restrict__ ss, int N, int W, int PF, int nTiles) {
    extern __shared__ float spf[];                       // TI*N*PF floats when SMEM
    const int b0 = blockIdx.x / nTiles, it = blockIdx.x - b0 * nTiles;
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    const int i0 = it * TI;
    const int nt = min(TI, N - i0);
    // Stage BEFORE the k >= W early-exit: the block is rounded up to a multiple of 32 threads, so some
    // threads must return, and any of them skipping the barrier would hang or corrupt the tile.
    if (SMEM) {
        const float* pfb = pf + ((size_t)b0 * N + i0) * N * PF;
        const int total = nt * N * PF;
        for (int t = threadIdx.x; t < total; t += blockDim.x) spf[t] = pfb[t];
        __syncthreads();
    }
    if (k >= W) return;
    const float ck = c1[k];
    float Ek[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) Ek[t] = (t < PF) ? E[(size_t)t * W + k] : 0.f;
    const unsigned char* mrow = mask + (size_t)b0 * N;
    const float* brow = b + (size_t)b0 * N * bstride;
    float ai[TI], m[TI], s[TI], acc[TI];
    bool ok[TI];
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        ai[ii] = 0.f; m[ii] = -1e30f; s[ii] = 0.f; acc[ii] = 0.f;
        ok[ii] = (ii < nt) && mrow[i0 + ii];
        if (ok[ii]) ai[ii] = a[((size_t)b0 * N + i0 + ii) * astride + k];
    }
    #pragma unroll 1
    for (int j = 0; j < N; ++j) {
        const float bj = brow[(size_t)j * bstride + k];  // one load serving all TI entities
        const bool vj = mrow[j];
        const float* q0 = SMEM ? (spf + (size_t)j * PF)
                               : (pf + ((size_t)(b0 * N + i0) * N + j) * PF);
        float zd[TI];
        pf_dot_all<TI>(q0, N * PF, PF, Ek, zd);
        #pragma unroll
        for (int ii = 0; ii < TI; ++ii) {
            if (!ok[ii]) continue;
            const float z = ai[ii] + bj + ck + zd[ii];
            softmax_step(z > 0.f ? z : z * SLOPE, vj && (j != i0 + ii), m[ii], s[ii], acc[ii]);
        }
    }
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        if (ii >= nt) break;
        const size_t r = (size_t)b0 * N + i0 + ii;
        if (!mrow[i0 + ii]) {
            y[r * W + k] = 0.f; mx[r * W + k] = -1e30f; ss[r * W + k] = 0.f;
        } else {
            y[r * W + k] = s[ii] > 0.f ? acc[ii] / s[ii] : 0.f;
            mx[r * W + k] = m[ii];
            ss[r * W + k] = s[ii];
        }
    }
}

// Backward.  __launch_bounds__(1024) caps ptxas at 64 registers, which buys the occupancy that
// matters here; the few spilled bytes of the per-entity arrays cost less than the lost occupancy.
// The per-entity arrays and the ii loop MUST stay fully unrolled: with a rolling loop the compiler
// demotes them to local memory and the backward gets 3-5x slower at TI >= 4.
template <int TI, bool SMEM>
__global__ __launch_bounds__(1024) void k_bwd(
        const float* __restrict__ gy, const float* __restrict__ a, int astride,
        const float* __restrict__ b, int bstride, const float* __restrict__ pf,
        const float* __restrict__ E, const float* __restrict__ c1,
        const unsigned char* __restrict__ mask, const float* __restrict__ y,
        const float* __restrict__ mx, const float* __restrict__ ss,
        float* __restrict__ ga, float* __restrict__ gb, int gstride,
        float* __restrict__ gE, float* __restrict__ gc1, int N, int W, int PF, int nTiles) {
    extern __shared__ float spf[];
    const int b0 = blockIdx.x / nTiles, it = blockIdx.x - b0 * nTiles;
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    const int i0 = it * TI;
    if (SMEM) {                                          // stage + sync before any early return
        const int nt = min(TI, N - i0);
        const float* pfb = pf + ((size_t)b0 * N + i0) * N * PF;
        const int total = nt * N * PF;
        for (int t = threadIdx.x; t < total; t += blockDim.x) spf[t] = pfb[t];
        __syncthreads();
    }
    if (k >= W) return;
    const float ck = c1[k];
    float Ek[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) Ek[t] = (t < PF) ? E[(size_t)t * W + k] : 0.f;
    const unsigned char* mrow = mask + (size_t)b0 * N;
    const float* brow = b + (size_t)b0 * N * bstride;
    float ai[TI], gys[TI], mxr[TI], yr[TI], gac[TI];
    bool ok[TI];
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        ai[ii] = gys[ii] = mxr[ii] = yr[ii] = gac[ii] = 0.f;
        ok[ii] = false;
        if (i0 + ii >= N || !mrow[i0 + ii]) continue;
        const int r = b0 * N + i0 + ii;
        const float s_ = ss[(size_t)r * W + k];
        if (!(s_ > 0.f)) continue;
        ok[ii] = true;
        ai[ii] = a[(size_t)r * astride + k];
        gys[ii] = gy[(size_t)r * W + k] / s_;
        mxr[ii] = mx[(size_t)r * W + k];
        yr[ii] = y[(size_t)r * W + k];
    }
    float gEk[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) gEk[t] = 0.f;
    float gck = 0.f;
    #pragma unroll 1
    for (int j = 0; j < N; ++j) {
        if (!mrow[j]) continue;
        const float bj = brow[(size_t)j * bstride + k];
        const float* q0 = SMEM ? (spf + (size_t)j * PF)
                               : (pf + ((size_t)(b0 * N + i0) * N + j) * PF);
        float zd[TI];
        pf_dot_all<TI>(q0, N * PF, PF, Ek, zd);
        float gd = 0.f;
        #pragma unroll
        for (int ii = 0; ii < TI; ++ii) {
            if (!ok[ii] || j == i0 + ii) continue;
            const float z = ai[ii] + bj + ck + zd[ii];
            const float v = z > 0.f ? z : z * SLOPE;
            const float dz = gys[ii] * expf(v - mxr[ii]) * (1.f + v - yr[ii]) * (z > 0.f ? 1.f : SLOPE);
            gac[ii] += dz; gd += dz; gck += dz;
            pf_acc_rt(q0 + (size_t)ii * N * PF, dz, gEk, PF);
        }
        if (gd != 0.f) atomicAdd(gb + (size_t)(b0 * N + j) * gstride + k, gd);
    }
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii)
        if (i0 + ii < N) ga[(size_t)(b0 * N + i0 + ii) * gstride + k] = gac[ii];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t)
        if (t < PF && gEk[t] != 0.f) atomicAdd(gE + (size_t)t * W + k, gEk[t]);
    if (gck != 0.f) atomicAdd(gc1 + k, gck);
}

// ====================================================================================== LIST layout
// Receiver r names its own K partners: nbr[r*K + kk] is a sender ROW (already offset by its batch) or
// -1.  The index is the same for every thread of the block (threads are slices), so the "-1" branch is
// warp-uniform and costs nothing, and the b-row gather is one coalesced read per slice.  A receiver
// whose list is all -1 comes out exactly 0, which is how masked receivers are expressed.
template <int TI>
__global__ __launch_bounds__(512) void k_fwd_list(
        const float* __restrict__ a, int astride, const float* __restrict__ b, int bstride,
        const float* __restrict__ pf, const int* __restrict__ nbr, const float* __restrict__ E,
        const float* __restrict__ c1, float* __restrict__ y, float* __restrict__ mx,
        float* __restrict__ ss, int R, int K, int W, int PF) {
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= W) return;
    const int r0 = blockIdx.x * TI;
    const float ck = c1[k];
    float Ek[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) Ek[t] = (t < PF) ? E[(size_t)t * W + k] : 0.f;
    float ai[TI], m[TI], s[TI], acc[TI];
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        m[ii] = -1e30f; s[ii] = 0.f; acc[ii] = 0.f;
        ai[ii] = (r0 + ii < R) ? a[(size_t)(r0 + ii) * astride + k] : 0.f;
    }
    #pragma unroll 2
    for (int kk = 0; kk < K; ++kk) {
        #pragma unroll
        for (int ii = 0; ii < TI; ++ii) {
            const int r = r0 + ii;
            if (r >= R) continue;
            const int j = nbr[(size_t)r * K + kk];
            if (j < 0) continue;
            const float z = ai[ii] + b[(size_t)j * bstride + k] + ck
                          + pf_dot_rt(pf + ((size_t)r * K + kk) * PF, Ek, PF);
            softmax_step(z > 0.f ? z : z * SLOPE, true, m[ii], s[ii], acc[ii]);
        }
    }
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        const int r = r0 + ii;
        if (r >= R) break;
        y[(size_t)r * W + k] = s[ii] > 0.f ? acc[ii] / s[ii] : 0.f;
        mx[(size_t)r * W + k] = m[ii];
        ss[(size_t)r * W + k] = s[ii];
    }
}

// Backward.  ga is written (every receiver row is owned by one block); gb is scattered with
// atomicAdd since a sender row is a partner of many receivers.  Adjacent slices of a warp hit adjacent
// addresses, so the atomics coalesce.
template <int TI>
__global__ __launch_bounds__(512) void k_bwd_list(
        const float* __restrict__ gy, const float* __restrict__ a, int astride,
        const float* __restrict__ b, int bstride, const float* __restrict__ pf,
        const int* __restrict__ nbr, const float* __restrict__ E, const float* __restrict__ c1,
        const float* __restrict__ y, const float* __restrict__ mx, const float* __restrict__ ss,
        float* __restrict__ ga, int gastride, float* __restrict__ gb, int gbstride,
        float* __restrict__ gE, float* __restrict__ gc1, int R, int K, int W, int PF) {
    const int k = blockIdx.y * blockDim.x + threadIdx.x;
    if (k >= W) return;
    const int r0 = blockIdx.x * TI;
    const float ck = c1[k];
    float Ek[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) Ek[t] = (t < PF) ? E[(size_t)t * W + k] : 0.f;
    float ai[TI], gys[TI], mxr[TI], yr[TI], gac[TI];
    bool ok[TI];
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
        ai[ii] = gys[ii] = mxr[ii] = yr[ii] = gac[ii] = 0.f;
        ok[ii] = false;
        const int r = r0 + ii;
        if (r >= R) continue;
        const float s_ = ss[(size_t)r * W + k];
        if (!(s_ > 0.f)) continue;
        ok[ii] = true;
        ai[ii] = a[(size_t)r * astride + k];
        gys[ii] = gy[(size_t)r * W + k] / s_;
        mxr[ii] = mx[(size_t)r * W + k];
        yr[ii] = y[(size_t)r * W + k];
    }
    float gEk[MAXPF];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t) gEk[t] = 0.f;
    float gck = 0.f;
    #pragma unroll 1
    for (int kk = 0; kk < K; ++kk) {
        #pragma unroll
        for (int ii = 0; ii < TI; ++ii) {
            if (!ok[ii]) continue;
            const int r = r0 + ii;
            const int j = nbr[(size_t)r * K + kk];
            if (j < 0) continue;
            const float* q = pf + ((size_t)r * K + kk) * PF;
            const float z = ai[ii] + b[(size_t)j * bstride + k] + ck + pf_dot_rt(q, Ek, PF);
            const float v = z > 0.f ? z : z * SLOPE;
            const float dz = gys[ii] * expf(v - mxr[ii]) * (1.f + v - yr[ii]) * (z > 0.f ? 1.f : SLOPE);
            gac[ii] += dz; gck += dz;
            pf_acc_rt(q, dz, gEk, PF);
            atomicAdd(gb + (size_t)j * gbstride + k, dz);
        }
    }
    #pragma unroll
    for (int ii = 0; ii < TI; ++ii)
        if (r0 + ii < R) ga[(size_t)(r0 + ii) * gastride + k] = gac[ii];
    #pragma unroll
    for (int t = 0; t < MAXPF; ++t)
        if (t < PF && gEk[t] != 0.f) atomicAdd(gE + (size_t)t * W + k, gEk[t]);
    if (gck != 0.f) atomicAdd(gc1 + k, gck);
}

// ===================================================================================== NEIGHBOURS
// Fixed-radius k-nearest search on a uniform grid (cell edge = radius), one thread per receiver.  The
// entities were sorted by cell in torch; a receiver scans the 3x3 cells around its own, keeps the K
// nearest candidates within the radius by insertion into a sorted register/local array, and writes
// their global rows nearest-first (-1 pads).  O(N) for bounded density, against O(N^2) for cdist+topk.
#define KMAX 64
__global__ void k_knn(const float2* __restrict__ xy, const int* __restrict__ order,
                      const int* __restrict__ cell_start, const int* __restrict__ cell_of,
                      const unsigned char* __restrict__ mask, int* __restrict__ out,
                      int R, int K, int N, int Gx, int Gy, float r2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    int* o = out + (size_t)i * K;
    if (!mask[i]) { for (int k = 0; k < K; ++k) o[k] = -1; return; }
    const int ncell = Gx * Gy;
    const int b = i / N;
    const int key = cell_of[i] - b * ncell;
    const int cy = key / Gx, cx = key - cy * Gx;
    const float2 p = xy[i];
    float bd[KMAX]; int bi[KMAX];
    int cnt = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int yy = cy + dy;
        if (yy < 0 || yy >= Gy) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int xx = cx + dx;
            if (xx < 0 || xx >= Gx) continue;
            const int c = b * ncell + yy * Gx + xx;
            for (int q = cell_start[c]; q < cell_start[c + 1]; ++q) {
                const int j = order[q];
                if (j == i) continue;
                const float2 pj = xy[j];
                const float ddx = p.x - pj.x, ddy = p.y - pj.y;
                const float d2 = ddx * ddx + ddy * ddy;
                if (d2 > r2) continue;
                if (cnt == K && d2 >= bd[K - 1]) continue;
                int pos = cnt < K ? cnt : K - 1;
                while (pos > 0 && bd[pos - 1] > d2) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; --pos; }
                bd[pos] = d2; bi[pos] = j;
                if (cnt < K) ++cnt;
            }
        }
    }
    for (int k = 0; k < K; ++k) o[k] = k < cnt ? bi[k] : -1;
}

torch::Tensor les_knn(torch::Tensor xy, torch::Tensor order, torch::Tensor cell_start,
                      torch::Tensor cell_of, torch::Tensor mask, int64_t K, int64_t N,
                      int64_t Gx, int64_t Gy, double radius) {
    TORCH_CHECK(K >= 1 && K <= KMAX, "K must be 1..", KMAX);
    TORCH_CHECK(xy.is_cuda() && xy.dtype() == torch::kFloat32 && xy.is_contiguous(), "xy: contiguous fp32 cuda (R,2)");
    const int R = (int)xy.size(0);
    auto out = torch::empty({R, K}, order.options());
    auto st = at::cuda::getCurrentCUDAStream();
    k_knn<<<(R + 127) / 128, 128, 0, st>>>((const float2*)xy.data_ptr<float>(), order.data_ptr<int>(),
        cell_start.data_ptr<int>(), cell_of.data_ptr<int>(), (const unsigned char*)mask.data_ptr<bool>(),
        out.data_ptr<int>(), R, (int)K, (int)N, (int)Gx, (int)Gy, (float)(radius * radius));
    return out;
}

// ------------------------------------------------------------------------------------------- host
// Only a fixed set of tile sizes is instantiated; snap to it BEFORE anything is derived from the value
// (the shared tile is sized from `ti`, while the kernel stages min(TI, N-i0) rows from its own
// template TI, so an uninstantiated `ti` would under-allocate shared memory and corrupt the tile).
static inline int snap_ti(int v) { return (v == 1 || v == 2 || v == 8) ? v : 4; }
static const int TI_FWD = 8, TI_BWD = 4, TI_LIST = 1;   // list: tools/tune_list.py, TI=1 best at W=64..256
static const size_t SMEM_LIMIT = 48 * 1024;

static void check(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.dtype() == torch::kFloat32 && t.is_contiguous(),
                "need contiguous fp32 cuda tensor");
}
static void check_pf(int PF) {
    TORCH_CHECK(PF >= 1 && PF <= MAXPF, "pair-feature width must be 1..", MAXPF, " (got ", PF,
                "); raise MAXPF in kernels.py if you need more");
}
static int block_threads(int W, int cap) { return W >= cap ? cap : ((W + 31) / 32) * 32; }

std::vector<torch::Tensor> les_fwd(torch::Tensor ab, torch::Tensor pf, torch::Tensor E,
                                   torch::Tensor c1, torch::Tensor mask, int64_t N, int64_t TI,
                                   bool smem) {
    check(ab); check(pf); check(E); check(c1);
    const int R = (int)ab.size(0), S = (int)ab.size(1), W = S / 2, PF = (int)pf.size(3);
    check_pf(PF);
    const int B0 = R / (int)N;
    auto y = torch::empty({R, W}, ab.options());
    auto mx = torch::empty({R, W}, ab.options());
    auto ss = torch::empty({R, W}, ab.options());
    const int ti = snap_ti(TI > 0 ? (int)TI : TI_FWD);
    const int nTiles = cdiv((int)N, ti);
    const int threads = block_threads(W, 1024);
    dim3 grid(B0 * nTiles, cdiv(W, threads));
    // falls back to global loads when the tile would not fit in shared memory: correct at any N
    const size_t tile = (size_t)ti * (size_t)N * (size_t)PF * sizeof(float);
    const bool sm = smem && tile <= SMEM_LIMIT;
    const size_t shmem = sm ? tile : 0;
    const float* base = ab.data_ptr<float>();
    auto st = at::cuda::getCurrentCUDAStream();
    #define CALL(TI_, SM_) k_fwd<TI_, SM_><<<grid, threads, shmem, st>>>(base, S, base + W, S, \
        pf.data_ptr<float>(), E.data_ptr<float>(), c1.data_ptr<float>(), \
        (const unsigned char*)mask.data_ptr<bool>(), y.data_ptr<float>(), mx.data_ptr<float>(), \
        ss.data_ptr<float>(), (int)N, W, PF, nTiles)
    #define BY_TI(SM_) do { switch (ti) { case 1: CALL(1, SM_); break; case 2: CALL(2, SM_); break; \
        case 8: CALL(8, SM_); break; default: CALL(4, SM_); break; } } while (0)
    if (sm) BY_TI(true); else BY_TI(false);
    #undef BY_TI
    #undef CALL
    return {y, mx, ss};
}

std::vector<torch::Tensor> les_bwd(torch::Tensor gy, torch::Tensor ab, torch::Tensor pf,
                                   torch::Tensor E, torch::Tensor c1, torch::Tensor mask,
                                   torch::Tensor y, torch::Tensor mx, torch::Tensor ss,
                                   int64_t N, int64_t TI, bool smem) {
    check(gy); check(ab); check(pf); check(E); check(c1);
    const int R = (int)ab.size(0), S = (int)ab.size(1), W = S / 2, PF = (int)pf.size(3);
    check_pf(PF);
    const int B0 = R / (int)N;
    auto gab = torch::zeros({R, S}, ab.options());
    auto gE = torch::zeros({PF, W}, ab.options());
    auto gc1 = torch::zeros({W}, ab.options());
    const int ti = snap_ti(TI > 0 ? (int)TI : TI_BWD);
    const int nTiles = cdiv((int)N, ti);
    const int threads = block_threads(W, 1024);
    dim3 grid(B0 * nTiles, cdiv(W, threads));
    const size_t tile = (size_t)ti * (size_t)N * PF * sizeof(float);
    const bool sm = smem && tile <= SMEM_LIMIT;
    const size_t shmem = sm ? tile : 0;
    const float* base = ab.data_ptr<float>();
    float* gbase = gab.data_ptr<float>();
    auto st = at::cuda::getCurrentCUDAStream();
    #define CALL(TI_, SM_) k_bwd<TI_, SM_><<<grid, threads, shmem, st>>>(gy.data_ptr<float>(), \
        base, S, base + W, S, pf.data_ptr<float>(), E.data_ptr<float>(), c1.data_ptr<float>(), \
        (const unsigned char*)mask.data_ptr<bool>(), y.data_ptr<float>(), mx.data_ptr<float>(), \
        ss.data_ptr<float>(), gbase, gbase + W, S, gE.data_ptr<float>(), gc1.data_ptr<float>(), \
        (int)N, W, PF, nTiles)
    #define BY_TI(SM_) do { switch (ti) { case 1: CALL(1, SM_); break; case 2: CALL(2, SM_); break; \
        case 8: CALL(8, SM_); break; default: CALL(4, SM_); break; } } while (0)
    if (sm) BY_TI(true); else BY_TI(false);
    #undef BY_TI
    #undef CALL
    return {gab, gE, gc1};
}

// a (R, astride>=W), b (S_rows, bstride>=W) may be the two halves of one packed (R, 2W) matrix.
std::vector<torch::Tensor> les_fwd_list(torch::Tensor a, torch::Tensor b, torch::Tensor pf,
                                        torch::Tensor nbr, torch::Tensor E, torch::Tensor c1,
                                        int64_t TI) {
    TORCH_CHECK(a.is_cuda() && a.dtype() == torch::kFloat32 && a.stride(1) == 1, "a: fp32 cuda, unit col stride");
    TORCH_CHECK(b.is_cuda() && b.dtype() == torch::kFloat32 && b.stride(1) == 1, "b: fp32 cuda, unit col stride");
    check(pf); check(E); check(c1);
    TORCH_CHECK(nbr.is_cuda() && nbr.dtype() == torch::kInt32 && nbr.is_contiguous(), "nbr: contiguous int32 cuda");
    const int R = (int)a.size(0), W = (int)a.size(1), K = (int)nbr.size(1), PF = (int)pf.size(2);
    TORCH_CHECK(nbr.size(0) == R && pf.size(0) == R && pf.size(1) == K, "nbr/pf shape mismatch");
    check_pf(PF);
    auto y = torch::empty({R, W}, a.options());
    auto mx = torch::empty({R, W}, a.options());
    auto ss = torch::empty({R, W}, a.options());
    const int ti = snap_ti(TI > 0 ? (int)TI : TI_LIST);
    const int threads = block_threads(W, 512);
    dim3 grid(cdiv(R, ti), cdiv(W, threads));
    auto st = at::cuda::getCurrentCUDAStream();
    #define CALL(TI_) k_fwd_list<TI_><<<grid, threads, 0, st>>>(a.data_ptr<float>(), (int)a.stride(0), \
        b.data_ptr<float>(), (int)b.stride(0), pf.data_ptr<float>(), nbr.data_ptr<int>(), \
        E.data_ptr<float>(), c1.data_ptr<float>(), y.data_ptr<float>(), mx.data_ptr<float>(), \
        ss.data_ptr<float>(), R, K, W, PF)
    switch (ti) { case 1: CALL(1); break; case 2: CALL(2); break; case 8: CALL(8); break; default: CALL(4); break; }
    #undef CALL
    return {y, mx, ss};
}

// ga and gb are caller-owned, zeroed, with unit column stride (they may be halves of one packed buffer).
void les_bwd_list(torch::Tensor gy, torch::Tensor a, torch::Tensor b, torch::Tensor pf,
                  torch::Tensor nbr, torch::Tensor E, torch::Tensor c1, torch::Tensor y,
                  torch::Tensor mx, torch::Tensor ss, torch::Tensor ga, torch::Tensor gb,
                  torch::Tensor gE, torch::Tensor gc1, int64_t TI) {
    check(gy); check(pf); check(E); check(c1); check(y); check(mx); check(ss); check(gE); check(gc1);
    const int R = (int)a.size(0), W = (int)a.size(1), K = (int)nbr.size(1), PF = (int)pf.size(2);
    check_pf(PF);
    const int ti = snap_ti(TI > 0 ? (int)TI : TI_LIST);
    const int threads = block_threads(W, 512);
    dim3 grid(cdiv(R, ti), cdiv(W, threads));
    auto st = at::cuda::getCurrentCUDAStream();
    #define CALL(TI_) k_bwd_list<TI_><<<grid, threads, 0, st>>>(gy.data_ptr<float>(), \
        a.data_ptr<float>(), (int)a.stride(0), b.data_ptr<float>(), (int)b.stride(0), \
        pf.data_ptr<float>(), nbr.data_ptr<int>(), E.data_ptr<float>(), c1.data_ptr<float>(), \
        y.data_ptr<float>(), mx.data_ptr<float>(), ss.data_ptr<float>(), \
        ga.data_ptr<float>(), (int)ga.stride(0), gb.data_ptr<float>(), (int)gb.stride(0), \
        gE.data_ptr<float>(), gc1.data_ptr<float>(), R, K, W, PF)
    switch (ti) { case 1: CALL(1); break; case 2: CALL(2); break; case 8: CALL(8); break; default: CALL(4); break; }
    #undef CALL
}

// [id, registers, local (spill) bytes, blocks/SM @256 threads] per default variant.
std::vector<int64_t> les_info() {
    std::vector<int64_t> out;
    #define INFO(id, ...) do { \
        const void* fn = (const void*)(__VA_ARGS__); \
        cudaFuncAttributes at; cudaFuncGetAttributes(&at, fn); \
        int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, 256, 0); \
        out.push_back(id); out.push_back(at.numRegs); out.push_back((int64_t)at.localSizeBytes); \
        out.push_back(occ); } while (0)
    INFO(0, (k_fwd<8, true>));
    INFO(1, (k_bwd<4, true>));
    INFO(2, (k_fwd_list<1>));
    INFO(3, (k_bwd_list<1>));
    #undef INFO
    return out;
}
"""


def load_extension(verbose=False):
    global _ext, _ext_error
    if _ext is not None:
        return True
    if not torch.cuda.is_available():
        _ext_error = "CUDA not available"
        return False
    try:
        from torch.utils.cpp_extension import load_inline
        build = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "les_ext_v2")
        os.makedirs(build, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        flags = ["-O3", "--use_fast_math", "-Xptxas=-O3"]
        if os.environ.get("LES_PTXAS_V"):     # prints per-kernel register/spill counts at build time
            flags.append("-Xptxas=-v")
        _ext = load_inline(name="les_ext_v2", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["les_fwd", "les_bwd", "les_fwd_list", "les_bwd_list", "les_knn", "les_info"],
                           extra_cuda_cflags=flags, build_directory=build,
                           verbose=verbose or bool(os.environ.get("LES_VERBOSE")))
    except Exception as e:  # noqa: BLE001
        _ext_error = f"{type(e).__name__}: {str(e).strip().splitlines()[-1] if str(e).strip() else ''}"
        _ext = None
        return False
    return True


def extension_status():
    return "CUDA kernels" if _ext is not None else f"unavailable ({_ext_error})"


def require_extension(verbose=False):
    """load_extension() but loud: a build failure is raised with its message instead of surfacing
    later as a confusing AttributeError on `_ext`."""
    if not load_extension(verbose=verbose):
        raise RuntimeError(f"CUDA kernel build failed: {_ext_error}")
    return _ext


def les_info():
    """[(kernel, registers, local/spill bytes, blocks/SM @256 threads)]."""
    require_extension()
    labels = {0: "dense fwd TI=8 smem", 1: "dense bwd TI=4 smem", 2: "list fwd TI=1", 3: "list bwd TI=1"}
    raw = _ext.les_info()
    return [(labels[r[0]], r[1], r[2], r[3]) for r in (raw[i:i + 4] for i in range(0, len(raw), 4))]


# ------------------------------------------------------------------------------ raw kernel entry points
def les_fwd(ab, pf, E, c1, mask, N, TI=0, smem=True):
    return _ext.les_fwd(ab, pf, E, c1, mask, N, TI, smem)


def les_bwd(gy, ab, pf, E, c1, mask, y, mx, ss, N, TI=0, smem=True):
    return _ext.les_bwd(gy, ab, pf, E, c1, mask, y, mx, ss, N, TI, smem)


def les_fwd_list(a, b, pf, nbr, E, c1, TI=0):
    return _ext.les_fwd_list(a, b, pf, nbr, E, c1, TI)


def les_bwd_list(gy, a, b, pf, nbr, E, c1, y, mx, ss, ga, gb, gE, gc1, TI=0):
    _ext.les_bwd_list(gy, a, b, pf, nbr, E, c1, y, mx, ss, ga, gb, gE, gc1, TI)


def les_knn(xy, order, cell_start, cell_of, mask, K, N, Gx, Gy, radius):
    return _ext.les_knn(xy, order, cell_start, cell_of, mask, K, N, Gx, Gy, radius)


# ---------------------------------------------------------------------------------------------- autograd
def _split_grad(gWc, W):
    # Deliberately NOT .contiguous(): the two halves are valid strided parameter gradients, and
    # materialising them costs two global copies per layer per step.  Autograd and Adam both accept a
    # strided .grad; it only changes how the optimizer walks the elements.
    if CONTIG_GRAD:
        return gWc[:, :W].contiguous(), gWc[:, W:].contiguous()
    return gWc[:, :W], gWc[:, W:]


class _LesLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask, pf, A, B, E, c1, N, TI, TIB, need_gx):
        Bn = x.shape[0]
        xf = x.reshape(Bn * N, -1)
        Wc = torch.cat([A, B], 1)                 # (C, 2W) -- tiny; one matmul instead of two
        ab = xf @ Wc                              # (R, 2W) packed [a | b]
        y, mx, ss = _ext.les_fwd(ab, pf, E, c1, mask, N, TI, True)
        # a and b are saved rather than recomputed: two fewer matmuls per layer per backward
        ctx.save_for_backward(xf, mask, pf, A, B, E, c1, ab, y, mx, ss)
        ctx.N, ctx.TIB, ctx.need_gx = N, TIB, need_gx
        ctx.shape = x.shape
        ctx.Wc = Wc
        return y.view(Bn, N, -1)

    @staticmethod
    def backward(ctx, gy):
        # Nothing built here needs tracing: without no_grad the internal matmuls and the `cat` become
        # graph nodes that are allocated, tracked and freed every layer and never executed.
        with torch.no_grad():
            xf, mask, pf, A, B, E, c1, ab, y, mx, ss = ctx.saved_tensors
            gy = gy.reshape(-1, gy.shape[-1]).contiguous()
            gab, gE, gc1 = _ext.les_bwd(gy, ab, pf, E, c1, mask, y, mx, ss, ctx.N, ctx.TIB, True)
            gA, gB = _split_grad(xf.t() @ gab, gab.shape[1] // 2)
            # layer 0's input is raw data: no parameters upstream, so the two (R,W)x(W,C) matmuls are skipped
            gx = (gab @ ctx.Wc.t()).view(ctx.shape) if ctx.need_gx else None
        return gx, None, None, gA, gB, gE, gc1, None, None, None, None


def les_layer(x, mask, pf, A, B, E, c1, N, TI=0, TIB=0, need_gx=True):
    """Dense layer.  x (B,N,C), mask (B,N) bool, pf (B,N,N,PF) -> y (B,N,W).  Excludes i == j."""
    return _LesLayer.apply(x, mask, pf, A, B, E, c1, N, TI, TIB, need_gx)


class _LesListLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xr, xs, nbr, pf, A, B, E, c1, TI, need_gx):
        same = xs is xr
        W = A.shape[1]
        if same:
            ab = xr @ torch.cat([A, B], 1)        # packed (R, 2W)
            a, b = ab[:, :W], ab[:, W:]
        else:
            a, b = xr @ A, xs @ B
        y, mx, ss = _ext.les_fwd_list(a, b, pf, nbr, E, c1, TI)
        ctx.save_for_backward(xr, xs, nbr, pf, A, B, E, c1, a, b, y, mx, ss)
        ctx.same, ctx.TI, ctx.need_gx = same, TI, need_gx
        return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            xr, xs, nbr, pf, A, B, E, c1, a, b, y, mx, ss = ctx.saved_tensors
            W = A.shape[1]
            gy = gy.contiguous()
            gE = torch.zeros_like(E)
            gc1 = torch.zeros_like(c1)
            if ctx.same:
                gab = torch.zeros(xr.shape[0], 2 * W, device=xr.device, dtype=xr.dtype)
                ga, gb = gab[:, :W], gab[:, W:]
            else:
                ga = torch.zeros_like(a)
                gb = torch.zeros_like(b)
            _ext.les_bwd_list(gy, a, b, pf, nbr, E, c1, y, mx, ss, ga, gb, gE, gc1, ctx.TI)
            if ctx.same:
                gA, gB = _split_grad(xr.t() @ gab, W)
                gxr = (gab @ torch.cat([A, B], 1).t()) if ctx.need_gx else None
                gxs = None
            else:
                gA, gB = xr.t() @ ga, xs.t() @ gb
                gxr = (ga @ A.t()) if ctx.need_gx else None
                gxs = (gb @ B.t()) if ctx.need_gx else None
        return gxr, gxs, None, None, gA, gB, gE, gc1, None, None


def les_layer_list(xr, xs, nbr, pf, A, B, E, c1, TI=0, need_gx=True):
    """List layer.  xr (R,C) receivers, xs (S,C) senders (pass the SAME tensor for both when they are
    one set), nbr (R,K) int32 sender rows or -1, pf (R,K,PF) -> y (R,W).  Rows are flattened over the
    batch, so a sender index is already offset by its batch (see neighbors.py)."""
    return _LesListLayer.apply(xr, xs, nbr, pf, A, B, E, c1, TI, need_gx)
