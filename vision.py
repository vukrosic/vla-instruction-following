"""The pretrained vision encoder every policy shares, frozen, as SmolVLA freezes its vision encoder and pi0 starts from a pretrained one.

It maps a 36x36 image to 6x6 cells of 32 numbers. Pretraining is on fresh random scenes with no sentences: each cell predicts the colour and
shape of the object centred there, whether the gripper is there, and whether the object there is lit. Every colour-shape pair appears, because
colour and shape are separate labels and a pretrained encoder would have seen ordinary objects anyway. The held pairs are held from language only.

Its weights are not counted in a policy's size. Without it, from-scratch `token` stayed at chance at naming objects after 1000 steps; with it,
`token` hit 35% of named objects by step 600 (debug/why_pretrained_vision.py).

python vision.py   pretrains and writes vision.pt, which is committed so every run uses the same encoder.
"""
import os, time, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import world as W

E, CELLS, PATH = 32, 6, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision.pt")


class Encoder(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.SiLU(), nn.Conv2d(16, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, E, 3, 3, 0), nn.SiLU(), nn.Conv2d(E, E, 3, 1, 1), nn.SiLU(), nn.Conv2d(E, E, 1))

    def forward(s, im):
        return s.c(im)


_cache = {}


def encode(frames, device="cpu", batch=512):
    """Frozen cells [B,E,6,6] of uint8 frames [B,36,36,3]. Training reads these from a cache, so the encoder's cost is paid once per data set."""
    dev = str(device)
    if dev not in _cache:
        m = Encoder(); m.load_state_dict(torch.load(PATH, map_location="cpu", weights_only=True)); m.to(device).eval().requires_grad_(False); _cache[dev] = m
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch): out.append(_cache[dev](torch.as_tensor(frames[i:i + batch], device=device).permute(0, 3, 1, 2).float() / 255))
    return torch.cat(out)


def scenes(n, r):
    """Fresh frames and per-cell labels: colour, shape, gripper, lit."""
    ls = [W.make_layout(r, k=int(r.integers(2, W.K_MAX + 1))) for _ in range(n)]
    b = W.Batch(ls, [W.task(l, "object", a=0, rng=r) for l in ls], r)
    b.agent = r.random((n, 2)).astype(b.agent.dtype); b.grip = r.random(n) < .3
    on = (np.arange(W.K_MAX)[None] < b.k[:, None]); b.lit = on & (r.random((n, W.K_MAX)) < .2)
    cell = np.minimum(((b.pos * (W.IMG - 1)).round().astype(int)) // CELLS, CELLS - 1); num = cell[..., 1] * CELLS + cell[..., 0]
    col = np.full((n, CELLS ** 2), len(W.COLORS)); shp = np.full((n, CELLS ** 2), len(W.SHAPES)); lit = np.zeros((n, CELLS ** 2), np.int64)
    I, J = np.nonzero(on); col[I, num[I, J]] = b.col[I, J]; shp[I, num[I, J]] = b.shp[I, J]; lit[I, num[I, J]] = b.lit[I, J]
    a = np.minimum(((b.agent * (W.IMG - 1)).round().astype(int)) // CELLS, CELLS - 1); gr = np.zeros((n, CELLS ** 2), np.int64); gr[np.arange(n), a[:, 1] * CELLS + a[:, 0]] = 1
    im = torch.as_tensor(b.render()).permute(0, 3, 1, 2).float() / 255
    return im, *(torch.as_tensor(x) for x in (col, shp, gr, lit))


def pretrain(steps=4000, batch=32, lr=3e-3, pool=4000, stop=None):
    torch.set_num_threads(1); torch.manual_seed(0); r = np.random.default_rng(0); m = Encoder(); data = scenes(pool, r)  # a scene costs about 4 ms, so a fixed pool is cheaper than fresh batches
    h = nn.ModuleList(nn.Linear(E, n) for n in (len(W.COLORS) + 1, len(W.SHAPES) + 1, 2, 2)); opt = torch.optim.Adam([*m.parameters(), *h.parameters()], lr=lr); t0 = time.time()
    rare = (None, None, torch.tensor([1., 8.]), torch.tensor([1., 8.]))  # gripper and lit cells are about 1 in 36
    for i in range(steps):
        for p in opt.param_groups: p["lr"] = lr * .5 * (1 + np.cos(np.pi * i / steps))
        i_ = torch.as_tensor(r.integers(pool, size=batch)); im, *y = (x[i_] for x in data); f = m(im).flatten(2).transpose(1, 2)
        loss = sum(F.cross_entropy(c(f).reshape(-1, c.out_features), t.reshape(-1), weight=w) for c, t, w in zip(h, y, rare)); opt.zero_grad(); loss.backward(); opt.step()
        if stop and time.time() - t0 > stop: break
    with torch.no_grad():
        im, *y = scenes(500, np.random.default_rng(1)); f = m(im).flatten(2).transpose(1, 2); obj = y[0] < len(W.COLORS); out = [c(f).argmax(-1) for c in h]
        acc = dict(colour=(out[0] == y[0])[obj].float().mean().item(), shape=(out[1] == y[1])[obj].float().mean().item(), gripper=(out[2] == y[2])[y[2] == 1].float().mean().item(), empty=(out[0] == y[0])[~obj].float().mean().item(), lit=(out[3] == y[3])[y[3] == 1].float().mean().item())
    return m, {k: round(v, 3) for k, v in acc.items()}, i + 1, round(time.time() - t0, 1)  # seconds include making the scenes


if __name__ == "__main__":
    m, acc, n, s = pretrain(stop=float(sys.argv[1]) if len(sys.argv) > 1 else None); torch.save(m.state_dict(), PATH); print(dict(steps=n, seconds=s, weights=sum(p.numel() for p in m.parameters()), **acc))
