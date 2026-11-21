"""Native 2-D SPH fluid simulator (weakly compressible), as a CUDA kernel: the "ground truth" and the
speed baseline the learned LESNet simulator is measured against.

Physics (standard WCSPH, Monaghan):
    density        rho_i  = sum_j m W(|x_i - x_j|, h)                   summation, fluid + wall particles
    pressure       p_i    = max(0, B ((rho_i / rho0)^7 - 1)),  B = rho0 c0^2 / 7        (Tait; no tension)
    acceleration   a_i    = -sum_j m (p_i/rho_i^2 + p_j/rho_j^2 + Pi_ij) grad W_ij  +  g_ext_i
    viscosity      Pi_ij  = -alpha c0 mu_ij / rho_bar   if v_ij . x_ij < 0,  mu = h v.x / (|x|^2 + 0.01 h^2)
    walls          static particles on a lattice, 3 layers thick; a wall neighbour has rho = rho0 and
                   p_j = p_i, v_j = 0 (so it pushes back with the fluid's own pressure and drags with viscosity)
    integration    symplectic Euler, fixed dt from the CFL condition

Every pair inside the kernel support (2h) is visited by brute force: each fluid particle loops over all
fluid and wall particles and skips those out of range.  At the particle counts used here (about 1-2k
fluid, 0.7k wall) that is cheaper than building a cell list, and it keeps the kernel short.

Units: SI-like, box [0,1]^2, g = 9.81.  `ext` is a per-particle external acceleration (gravity plus any
push), which is also the learned simulator's control input.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))

SPACING = 0.02                      # initial lattice spacing d
H = 1.3 * SPACING                   # smoothing length
RHO0 = 1000.0
MASS = RHO0 * SPACING ** 2
C0 = 40.0                           # numerical sound speed, about 10x the fastest flow
ALPHA = 0.08                        # artificial viscosity
DT = 0.25 * H / C0 * 0.8            # CFL, with margin
GRAVITY = 9.81
WALL_LAYERS = 3
SKIN = 0.03                         # cell-list margin: the grid is rebuilt every REBUILD substeps, particles move < SKIN/2 in that time
CELL = 2 * H + SKIN
ORIGIN = -0.1
NC = int(math.ceil(1.2 / CELL))
REBUILD = 8

_CPP = r"""
#include <torch/extension.h>
void sph_substeps(torch::Tensor pos, torch::Tensor vel, torch::Tensor wall, torch::Tensor ext,
                  torch::Tensor rho, torch::Tensor acc, int64_t nsub, double dt, double h, double mass,
                  double rho0, double c0, double alpha);
void sph_density(torch::Tensor pos, torch::Tensor wall, torch::Tensor rho, double h, double mass);
void sph_substeps_cl(torch::Tensor pos, torch::Tensor vel, torch::Tensor wall, torch::Tensor ext,
                     torch::Tensor rho, torch::Tensor acc, torch::Tensor order, torch::Tensor cell_id,
                     torch::Tensor cell_start, torch::Tensor wall_start, int64_t nc,
                     int64_t nsub, double dt, double h, double mass, double rho0, double c0, double alpha);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define PI_F 3.14159265358979f

__device__ __forceinline__ float kern(float r, float h) {
    const float q = r / h;
    const float s = 10.f / (7.f * PI_F * h * h);
    if (q >= 2.f) return 0.f;
    if (q < 1.f) return s * (1.f - 1.5f * q * q + 0.75f * q * q * q);
    const float t = 2.f - q;
    return s * 0.25f * t * t * t;
}
// dW/dr
__device__ __forceinline__ float dkern(float r, float h) {
    const float q = r / h;
    const float s = 10.f / (7.f * PI_F * h * h * h);
    if (q >= 2.f) return 0.f;
    if (q < 1.f) return s * (-3.f * q + 2.25f * q * q);
    const float t = 2.f - q;
    return s * (-0.75f * t * t);
}

__global__ void k_density(const float2* __restrict__ pos, const float2* __restrict__ wall,
                          float* __restrict__ rho, int nf, int nw, float h, float mass) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;
    const float2 pi = pos[i];
    const float h2x4 = 4.f * h * h;
    float s = 0.f;
    for (int j = 0; j < nf; ++j) {
        const float dx = pi.x - pos[j].x, dy = pi.y - pos[j].y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < h2x4) s += kern(sqrtf(d2), h);
    }
    for (int j = 0; j < nw; ++j) {
        const float dx = pi.x - wall[j].x, dy = pi.y - wall[j].y;
        const float d2 = dx * dx + dy * dy;
        if (d2 < h2x4) s += kern(sqrtf(d2), h);
    }
    rho[i] = mass * s;
}

__global__ void k_force(const float2* __restrict__ pos, const float2* __restrict__ vel,
                        const float2* __restrict__ wall, const float2* __restrict__ ext,
                        const float* __restrict__ rho, float2* __restrict__ acc, int nf, int nw,
                        float h, float mass, float rho0, float c0, float alpha) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;
    const float B = rho0 * c0 * c0 / 7.f;
    const float2 xi = pos[i], vi = vel[i];
    const float ri = rho[i];
    const float pi_ = fmaxf(0.f, B * (powf(ri / rho0, 7.f) - 1.f));
    const float pri = pi_ / (ri * ri);
    const float h2x4 = 4.f * h * h;
    float ax = 0.f, ay = 0.f;
    for (int j = 0; j < nf; ++j) {
        if (j == i) continue;
        const float dx = xi.x - pos[j].x, dy = xi.y - pos[j].y;
        const float d2 = dx * dx + dy * dy;
        if (d2 >= h2x4 || d2 < 1e-12f) continue;
        const float r = sqrtf(d2);
        const float rj = rho[j];
        const float pj = fmaxf(0.f, B * (powf(rj / rho0, 7.f) - 1.f));
        float term = pri + pj / (rj * rj);
        const float vx = vi.x - vel[j].x, vy = vi.y - vel[j].y;
        const float vr = vx * dx + vy * dy;
        if (vr < 0.f) {
            const float mu = h * vr / (d2 + 0.01f * h * h);
            term += -alpha * c0 * mu / (0.5f * (ri + rj));
        }
        const float g = -mass * term * dkern(r, h) / r;
        ax += g * dx; ay += g * dy;
    }
    for (int j = 0; j < nw; ++j) {
        const float dx = xi.x - wall[j].x, dy = xi.y - wall[j].y;
        const float d2 = dx * dx + dy * dy;
        if (d2 >= h2x4 || d2 < 1e-12f) continue;
        const float r = sqrtf(d2);
        float term = pri + pi_ / (rho0 * rho0);
        const float vr = vi.x * dx + vi.y * dy;
        if (vr < 0.f) {
            const float mu = h * vr / (d2 + 0.01f * h * h);
            term += -alpha * c0 * mu / (0.5f * (ri + rho0));
        }
        const float g = -mass * term * dkern(r, h) / r;
        ax += g * dx; ay += g * dy;
    }
    acc[i] = make_float2(ax + ext[i].x, ay + ext[i].y);
}

// symplectic Euler; a particle that tunnels out of the box (should not happen) is put back inside
__global__ void k_integrate(float2* __restrict__ pos, float2* __restrict__ vel,
                            const float2* __restrict__ acc, int nf, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;
    float2 v = vel[i], x = pos[i];
    v.x += acc[i].x * dt; v.y += acc[i].y * dt;
    x.x += v.x * dt; x.y += v.y * dt;
    if (x.x < 0.f) { x.x = 0.f; v.x = fabsf(v.x) * 0.3f; }
    if (x.x > 1.f) { x.x = 1.f; v.x = -fabsf(v.x) * 0.3f; }
    if (x.y < 0.f) { x.y = 0.f; v.y = fabsf(v.y) * 0.3f; }
    if (x.y > 1.f) { x.y = 1.f; v.y = -fabsf(v.y) * 0.3f; }
    vel[i] = v; pos[i] = x;
}

// ---- cell-list variant: fluid particles visited through a uniform grid (cell side >= 2h + skin), walls pre-sorted by cell.
// The grid is rebuilt by the caller every few substeps; `cell_id` is each particle's cell at build time, which
// together with the skin guarantees that every pair within 2h lies in neighbouring cells until the next rebuild.
__global__ void k_density_cl(const float2* __restrict__ pos, const float2* __restrict__ wall,
                             const int* __restrict__ order, const int* __restrict__ cell_id,
                             const int* __restrict__ cell_start, const int* __restrict__ wall_start,
                             float* __restrict__ rho, int nf, int nc, float h, float mass) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;           // threads follow the sorted order: a warp shares cells
    if (t >= nf) return;
    const int i = order[t];
    const float2 pi = pos[i];
    const float h2x4 = 4.f * h * h;
    const int c0 = cell_id[i], cx0 = c0 % nc, cy0 = c0 / nc;
    float s = 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int cy = cy0 + dy;
        if (cy < 0 || cy >= nc) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int cx = cx0 + dx;
            if (cx < 0 || cx >= nc) continue;
            const int c = cy * nc + cx;
            for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                const float2 pj = pos[order[k]];
                const float ex = pi.x - pj.x, ey = pi.y - pj.y;
                const float d2 = ex * ex + ey * ey;
                if (d2 < h2x4) s += kern(sqrtf(d2), h);
            }
            for (int k = wall_start[c]; k < wall_start[c + 1]; ++k) {
                const float ex = pi.x - wall[k].x, ey = pi.y - wall[k].y;
                const float d2 = ex * ex + ey * ey;
                if (d2 < h2x4) s += kern(sqrtf(d2), h);
            }
        }
    }
    rho[i] = mass * s;
}

__global__ void k_force_cl(const float2* __restrict__ pos, const float2* __restrict__ vel,
                           const float2* __restrict__ wall, const float2* __restrict__ ext,
                           const float* __restrict__ rho, float2* __restrict__ acc,
                           const int* __restrict__ order, const int* __restrict__ cell_id,
                           const int* __restrict__ cell_start, const int* __restrict__ wall_start,
                           int nf, int nc, float h, float mass, float rho0, float c0, float alpha) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nf) return;
    const int i = order[t];
    const float B = rho0 * c0 * c0 / 7.f;
    const float2 xi = pos[i], vi = vel[i];
    const float ri = rho[i];
    const float pi_ = fmaxf(0.f, B * (powf(ri / rho0, 7.f) - 1.f));
    const float pri = pi_ / (ri * ri);
    const float h2x4 = 4.f * h * h;
    const int cc = cell_id[i], cx0 = cc % nc, cy0 = cc / nc;
    float ax = 0.f, ay = 0.f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int cy = cy0 + dy;
        if (cy < 0 || cy >= nc) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int cx = cx0 + dx;
            if (cx < 0 || cx >= nc) continue;
            const int c = cy * nc + cx;
            for (int k = cell_start[c]; k < cell_start[c + 1]; ++k) {
                const int j = order[k];
                if (j == i) continue;
                const float ex = xi.x - pos[j].x, ey = xi.y - pos[j].y;
                const float d2 = ex * ex + ey * ey;
                if (d2 >= h2x4 || d2 < 1e-12f) continue;
                const float r = sqrtf(d2);
                const float rj = rho[j];
                const float pj = fmaxf(0.f, B * (powf(rj / rho0, 7.f) - 1.f));
                float term = pri + pj / (rj * rj);
                const float vx = vi.x - vel[j].x, vy = vi.y - vel[j].y;
                const float vr = vx * ex + vy * ey;
                if (vr < 0.f) {
                    const float mu = h * vr / (d2 + 0.01f * h * h);
                    term += -alpha * c0 * mu / (0.5f * (ri + rj));
                }
                const float g = -mass * term * dkern(r, h) / r;
                ax += g * ex; ay += g * ey;
            }
            for (int k = wall_start[c]; k < wall_start[c + 1]; ++k) {
                const float ex = xi.x - wall[k].x, ey = xi.y - wall[k].y;
                const float d2 = ex * ex + ey * ey;
                if (d2 >= h2x4 || d2 < 1e-12f) continue;
                const float r = sqrtf(d2);
                float term = pri + pi_ / (rho0 * rho0);
                const float vr = vi.x * ex + vi.y * ey;
                if (vr < 0.f) {
                    const float mu = h * vr / (d2 + 0.01f * h * h);
                    term += -alpha * c0 * mu / (0.5f * (ri + rho0));
                }
                const float g = -mass * term * dkern(r, h) / r;
                ax += g * ex; ay += g * ey;
            }
        }
    }
    acc[i] = make_float2(ax + ext[i].x, ay + ext[i].y);
}

void sph_substeps_cl(torch::Tensor pos, torch::Tensor vel, torch::Tensor wall, torch::Tensor ext,
                     torch::Tensor rho, torch::Tensor acc, torch::Tensor order, torch::Tensor cell_id,
                     torch::Tensor cell_start, torch::Tensor wall_start, int64_t nc,
                     int64_t nsub, double dt, double h, double mass, double rho0, double c0, double alpha) {
    const int nf = pos.size(0);
    auto st = at::cuda::getCurrentCUDAStream();
    const int blocks = (nf + 127) / 128;
    for (int s = 0; s < nsub; ++s) {
        k_density_cl<<<blocks, 128, 0, st>>>((const float2*)pos.data_ptr<float>(), (const float2*)wall.data_ptr<float>(),
            order.data_ptr<int>(), cell_id.data_ptr<int>(), cell_start.data_ptr<int>(), wall_start.data_ptr<int>(),
            rho.data_ptr<float>(), nf, (int)nc, (float)h, (float)mass);
        k_force_cl<<<blocks, 128, 0, st>>>((const float2*)pos.data_ptr<float>(), (const float2*)vel.data_ptr<float>(),
            (const float2*)wall.data_ptr<float>(), (const float2*)ext.data_ptr<float>(), rho.data_ptr<float>(),
            (float2*)acc.data_ptr<float>(), order.data_ptr<int>(), cell_id.data_ptr<int>(), cell_start.data_ptr<int>(),
            wall_start.data_ptr<int>(), nf, (int)nc, (float)h, (float)mass, (float)rho0, (float)c0, (float)alpha);
        k_integrate<<<blocks, 128, 0, st>>>((float2*)pos.data_ptr<float>(), (float2*)vel.data_ptr<float>(),
            (const float2*)acc.data_ptr<float>(), nf, (float)dt);
    }
}

void sph_density(torch::Tensor pos, torch::Tensor wall, torch::Tensor rho, double h, double mass) {
    const int nf = pos.size(0), nw = wall.size(0);
    auto st = at::cuda::getCurrentCUDAStream();
    k_density<<<(nf + 127) / 128, 128, 0, st>>>((const float2*)pos.data_ptr<float>(),
        (const float2*)wall.data_ptr<float>(), rho.data_ptr<float>(), nf, nw, (float)h, (float)mass);
}

void sph_substeps(torch::Tensor pos, torch::Tensor vel, torch::Tensor wall, torch::Tensor ext,
                  torch::Tensor rho, torch::Tensor acc, int64_t nsub, double dt, double h, double mass,
                  double rho0, double c0, double alpha) {
    const int nf = pos.size(0), nw = wall.size(0);
    auto st = at::cuda::getCurrentCUDAStream();
    const int blocks = (nf + 127) / 128;
    for (int s = 0; s < nsub; ++s) {
        k_density<<<blocks, 128, 0, st>>>((const float2*)pos.data_ptr<float>(),
            (const float2*)wall.data_ptr<float>(), rho.data_ptr<float>(), nf, nw, (float)h, (float)mass);
        k_force<<<blocks, 128, 0, st>>>((const float2*)pos.data_ptr<float>(),
            (const float2*)vel.data_ptr<float>(), (const float2*)wall.data_ptr<float>(),
            (const float2*)ext.data_ptr<float>(), rho.data_ptr<float>(), (float2*)acc.data_ptr<float>(),
            nf, nw, (float)h, (float)mass, (float)rho0, (float)c0, (float)alpha);
        k_integrate<<<blocks, 128, 0, st>>>((float2*)pos.data_ptr<float>(), (float2*)vel.data_ptr<float>(),
            (const float2*)acc.data_ptr<float>(), nf, (float)dt);
    }
}
"""

_ext = None


def load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline
        build = os.path.join(HERE, "build", "sph_ext2")
        os.makedirs(build, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        _ext = load_inline(name="sph_ext2", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["sph_substeps", "sph_density", "sph_substeps_cl"],
                           extra_cuda_cflags=["-O3", "--use_fast_math"], build_directory=build)
    return _ext


def _cell_ids(p):
    c = ((p - ORIGIN) / CELL).long().clamp_(0, NC - 1)
    return c[:, 1] * NC + c[:, 0]


def _starts(cid):
    st = torch.zeros(NC * NC + 1, dtype=torch.int32, device=cid.device)
    st[1:] = torch.bincount(cid, minlength=NC * NC).cumsum(0).int()
    return st


def wall_particles(device="cuda"):
    """Static wall lattice, WALL_LAYERS thick, just outside the unit box."""
    pts = []
    n = int(round(1.0 / SPACING))
    for layer in range(1, WALL_LAYERS + 1):
        off = -(layer - 0.5) * SPACING
        for i in range(-WALL_LAYERS, n + WALL_LAYERS + 1):
            c = i * SPACING
            pts += [(c, off), (c, 1.0 - off), (off, c), (1.0 - off, c)]
    w = torch.tensor(sorted(set(pts)), dtype=torch.float32)
    # drop duplicates produced where layers overlap at corners; keep only those outside or on the box edge ring
    keep = (w[:, 0] < 0.5 * SPACING) | (w[:, 0] > 1.0 - 0.5 * SPACING) | (w[:, 1] < 0.5 * SPACING) | (w[:, 1] > 1.0 - 0.5 * SPACING)
    return w[keep].contiguous().to(device)


class Sim:
    """State lives on the GPU.  pos, vel: (N,2) float32;  ext: (N,2) external acceleration."""
    def __init__(self, pos, vel=None, device="cuda", method="cell"):
        """method: 'cell' (cell list, the default speed baseline) or 'brute' (every pair; the reference)."""
        load()
        self.device = device
        self.method = method
        self.pos = pos.to(device).float().contiguous()
        self.vel = (torch.zeros_like(self.pos) if vel is None else vel.to(device).float()).contiguous()
        self.wall = wall_particles(device)
        cid = _cell_ids(self.wall)
        self.wall = self.wall[torch.argsort(cid)].contiguous()          # walls are static: sort them by cell once
        self.wall_start = _starts(cid)
        self.rho = torch.empty(len(self.pos), device=device)
        self.acc = torch.zeros_like(self.pos)
        self.ext = torch.zeros_like(self.pos)
        self.ext[:, 1] = -GRAVITY
        self.time = 0.0

    def set_gravity(self, gx, gy):
        self.ext[:] = 0
        self.ext[:, 0], self.ext[:, 1] = gx, gy

    def step(self, nsub):
        if self.method == "brute":
            _ext.sph_substeps(self.pos, self.vel, self.wall, self.ext, self.rho, self.acc, nsub, DT, H, MASS, RHO0, C0, ALPHA)
        else:
            left = nsub
            while left > 0:
                n = min(REBUILD, left)
                cid = _cell_ids(self.pos)
                order = torch.argsort(cid).int()
                _ext.sph_substeps_cl(self.pos, self.vel, self.wall, self.ext, self.rho, self.acc, order, cid.int(),
                                     _starts(cid), self.wall_start, NC, n, DT, H, MASS, RHO0, C0, ALPHA)
                left -= n
        self.time += nsub * DT

    def density(self):
        _ext.sph_density(self.pos, self.wall, self.rho, H, MASS)
        return self.rho


def lattice_block(x0, y0, w, h, spacing=SPACING):
    nx, ny = max(1, int(round(w / spacing))), max(1, int(round(h / spacing)))
    xs = x0 + (torch.arange(nx) + 0.5) * spacing
    ys = y0 + (torch.arange(ny) + 0.5) * spacing
    g = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    return g


def dam_break(width=0.4, height=0.6):
    return lattice_block(0.0, 0.0, width, height)


if __name__ == "__main__":
    # sanity: a dam break should settle with rho near rho0 and the centre of mass dropping then spreading
    import time
    sim = Sim(dam_break())
    print("N fluid", len(sim.pos), "N wall", len(sim.wall), "dt", DT)
    rho = sim.density()
    print("initial rho mean/min/max", rho.mean().item(), rho.min().item(), rho.max().item())
    torch.cuda.synchronize(); t0 = time.time()
    for f in range(40):
        sim.step(100)
        if f % 8 == 7:
            print(f"t={sim.time:.3f}s  com=({sim.pos[:,0].mean().item():.3f},{sim.pos[:,1].mean().item():.3f})"
                  f"  vmax={sim.vel.norm(dim=1).max().item():.2f}  rho mean {sim.density().mean().item():.0f}"
                  f"  in-box {((sim.pos>=0)&(sim.pos<=1)).all(1).float().mean().item():.3f}")
    torch.cuda.synchronize(); el = time.time() - t0
    print(f"{4000} substeps in {el:.2f}s = {1000*el/4000:.3f} ms/substep -> {el/(4000*DT):.1f} wall-s per sim-s")
