"""Profile the fast LESNet step: which kernels cost, how many launches, how much is overhead.

    python tools/profile_fast.py [B] [N] [cin]
"""
import os, sys
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import kernels  # noqa
import ref  # noqa
from model import LESNet  # noqa
from bench import synth  # noqa

dev = "cuda"
kernels.load_extension()
B = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
cin = int(sys.argv[3]) if len(sys.argv) > 3 else 16
x, coords, mask = synth(B, N, cin, dev)
net = LESNet(cin, width=256, depth=5).to(dev)
head = nn.Linear(256, 1).to(dev)
opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-4)
target = torch.zeros(B, device=dev)


def step():
    y = net(x, coords, mask)
    loss = ((head(ref.masked_mean(y, mask)) - target) ** 2).mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()


for _ in range(8):
    step()
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18))
