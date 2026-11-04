"""Seconds per training step, one run at a time on one thread. Rule 2 of beat.py reads this. Run it on an otherwise idle machine; speeds taken while other jobs run can be off by several times.

python3 speed.py --tier 1m
"""
import argparse, time
import numpy as np, torch
import env, models as M, vision as V, world as W

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="1m"); p.add_argument("--steps", type=int, default=5); a = p.parse_args(); C = env.TIERS[a.tier]; torch.set_num_threads(1); r = np.random.default_rng(0)
    ls = [W.make_layout(r) for _ in range(C["batch"])]; b = W.Batch(ls, [W.task(l, "object", a=0, rng=r) for l in ls], r); im = V.encode(b.render()); tok = torch.as_tensor(b.tok); st = torch.as_tensor(b.state()); goal = torch.as_tensor(b.goal()); y = torch.as_tensor(np.tile(b.expert()[:, None], (1, 8, 1)))
    for sp in [f"{f}+{h}" for f in M.BASELINES for h in ("mse", "flow")]:
        d, h = M.fit(sp, C["weights"], C["heads"], C["layers"]); m = M.Policy(sp, d, C["heads"], C["layers"], h); o = torch.optim.Adam(m.parameters()); g = torch.Generator().manual_seed(0); t = time.time()
        for _ in range(a.steps): l = m.loss(im, tok, st, y, goal=goal, g=g); o.zero_grad(); l.backward(); o.step()
        print(sp.ljust(20), f"{(time.time() - t) / a.steps * 1000:.1f} ms/step", M.size(m), "weights", flush=True)
