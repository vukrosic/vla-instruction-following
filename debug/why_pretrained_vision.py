"""Question: can a policy learn to point at a named object from pixels at this budget, or does it need pretrained vision?

python3 debug/why_pretrained_vision.py --seconds 7 --scratch    A small conv stem trained from pixels.
python3 debug/why_pretrained_vision.py --seconds 7              The frozen encoder.

Measured on a loaded laptop, 120k weights, batch 32, 6,000 scenes (an earlier encoder pretrained for 800 steps): from scratch, heldout hits stayed at 0.03 after 1000 steps;
with the encoder they reached 0.35 by step 600. A constant guess hits about 0.03. Longer runs were not made on that machine. These 7-second defaults only check that the script runs.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn
import models as M, vision as V, world as W

p = argparse.ArgumentParser(); p.add_argument("--spec", default="token+mse"); p.add_argument("--seconds", type=float, default=7); p.add_argument("--scratch", action="store_true"); p.add_argument("--n", type=int, default=300); a = p.parse_args()
torch.set_num_threads(1); torch.manual_seed(0); r = np.random.default_rng(0); ls = [W.make_layout(r) for _ in range(a.n)]; b = W.Batch(ls, [W.task(l, "object", a=int(r.integers(l["k"])), rng=r) for l in ls], r)
fr = b.render(); tok = torch.as_tensor(b.tok); st = torch.as_tensor(b.state()); go = torch.as_tensor(b.goal()); cut = int(a.n * .9); d, h = M.fit(a.spec, 120000, 4, 3); m = M.Policy(a.spec, d, 4, 3, h)
stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, V.E, 3, 3, 0)) if a.scratch else None
cells = None if a.scratch else V.encode(fr); px = torch.as_tensor(fr).permute(0, 3, 1, 2).float() / 255
see = lambda i: stem(px[i]) if a.scratch else cells[i]; out = lambda i: m.decode(m.context(see(i), tok[i], st[i], go[i]))[:, 0, :2]
o = torch.optim.Adam(list(m.parameters()) + (list(stem.parameters()) if a.scratch else []), lr=.002); t = time.time(); i = 0
while time.time() - t < a.seconds:
    ix = torch.as_tensor(r.integers(cut, size=32)); l = ((out(ix) - go[ix]) ** 2).mean(); o.zero_grad(); l.backward(); o.step(); i += 1
with torch.no_grad(): e = (out(torch.arange(cut, a.n)) - go[cut:]).norm(dim=1)
print(dict(steps=i, scratch=a.scratch, heldout_dist=round(e.mean().item(), 3), hits=round((e < W.R).float().mean().item(), 3)))
