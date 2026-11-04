"""Policies: one image-plus-language fusion module, one action head. A new design is one Fusion class and one entry in FUSIONS.

Every fusion takes frozen image cells from vision.encode, instruction tokens and the gripper state, and returns context tokens that a head reads. Every model is
sized to the same number of weights by fit(), which sets the width and then the MLP width. All of them read the same frozen pretrained vision
encoder (vision.py), whose weights are not counted, and use RMSNorm, SwiGLU, no biases and fixed 2D position features, which differ from some
of the source papers. These are small ports that follow the papers' text, and they were not compared with authors' code:

late        Pool image and language separately, concatenate. Lynch & Sermanet 2021 (LangLfP); the plain baseline in most VLA surveys.
film        Conv residual blocks scaled by language. Perez 2018 (FiLM); BC-Z (Jang 2021) and RT-1 (Brohan 2022) condition this way.
token       Image, language and state tokens share self-attention. ViLT (Kim 2021); pi0 and pi0.5 prefix tokens work this way.
cross       Language reads the image through cross-attention. CASA (2512.19535); adapted from its Helium setup.
gated       Cross-attention with a tanh gate starting at zero. Flamingo (Alayrac 2022); used by Llama 3.2 Vision.
tokenlearner FiLM conv, then 8 learned spatial tokens, then self-attention. RT-1 (Brohan 2022), TokenLearner (Ryoo 2021).
resampler   Image tokens compressed into 8 latents before self-attention. Flamingo's Perceiver Resampler; IDEFICS.
langattn    Language forms a spatial attention map, with its soft-argmax position passed on. Adapted from CLIPort's attention stream (Shridhar 2021).

Heads:
mse    One action, regressed. Lynch & Sermanet 2021.
chunk  Next 8 actions regressed, first 4 executed, then replan. ACT (Zhao 2023) without its CVAE or temporal ensembling.
flow   Flow matching on the next 8 actions, 4 Euler steps, first 4 executed. pi0 (2410.24164), SmolVLA (2506.01844), without KI or LoRA.

Controls, built on `token`:
blind    Sentence replaced by padding. It can only read the scene.
shuffle  Sentence taken from another row of the batch, in training and at test: a useless memory of language.
oracle   State also holds the true xy of the current goal object, an upper limit that needs no vision or language skill.
"""
import functools
import torch, torch.nn as nn, torch.nn.functional as F
import world, vision

HORIZON = dict(mse=1, chunk=8, flow=8)
EXECUTE = dict(mse=1, chunk=4, flow=4)
FLOW_STEPS = 4


class RMSNorm(nn.Module):
    def __init__(s, d):
        super().__init__(); s.w = nn.Parameter(torch.ones(d))

    def forward(s, x):
        return x / (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt() * s.w


class Attn(nn.Module):
    def __init__(s, d, heads):
        super().__init__(); s.h = heads; s.q = nn.Linear(d, d, bias=False); s.k = nn.Linear(d, d, bias=False); s.v = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, d, bias=False)

    def forward(s, x, c=None, mask=None):
        c = x if c is None else c; B, N, D = x.shape; M = c.shape[1]
        q = s.q(x).view(B, N, s.h, -1).transpose(1, 2); k = s.k(c).view(B, M, s.h, -1).transpose(1, 2); v = s.v(c).view(B, M, s.h, -1).transpose(1, 2)
        m = None if mask is None else mask[:, None, None, :]
        return s.o(F.scaled_dot_product_attention(q, k, v, attn_mask=m).transpose(1, 2).reshape(B, N, D))


class MLP(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.g = nn.Linear(d, h, bias=False); s.u = nn.Linear(d, h, bias=False); s.d = nn.Linear(h, d, bias=False)

    def forward(s, x):
        return s.d(F.silu(s.g(x)) * s.u(x))


class Block(nn.Module):
    """Self-attention, optional cross-attention, SwiGLU. `gated` puts Flamingo's tanh gate on the cross-attention."""

    def __init__(s, d, heads, mlp, cross=False, gated=False):
        super().__init__()
        s.n1 = RMSNorm(d); s.a = Attn(d, heads); s.n3 = RMSNorm(d); s.m = MLP(d, mlp); s.cross = cross
        if cross:
            s.n2 = RMSNorm(d); s.nc = RMSNorm(d); s.c = Attn(d, heads); s.gate = nn.Parameter(torch.zeros(1)) if gated else None

    def forward(s, x, mem=None, mask=None, mem_mask=None):
        n = s.n1(x); x = x + s.a(n, mask=mask)
        if s.cross:
            y = s.c(s.n2(x), s.nc(mem), mask=mem_mask)
            x = x + (y if s.gate is None else torch.tanh(s.gate) * y)
        return x + s.m(s.n3(x))


class Parts(nn.Module):
    """Shared inputs: frozen vision cells projected to width d with fixed 2D positions, language tokens with one self-attention block, and a state token."""

    def __init__(s, d, heads, mlp, state=4):
        super().__init__()
        s.vis = nn.Linear(vision.E, d, bias=False); s.pos = nn.Linear(6, d, bias=False)
        c = torch.linspace(-1, 1, vision.CELLS); y, x = torch.meshgrid(c, c, indexing="ij"); pi = 3.14159265
        s.register_buffer("grid", torch.stack([x, y, torch.sin(pi * x), torch.cos(pi * x), torch.sin(pi * y), torch.cos(pi * y)], -1).reshape(vision.CELLS ** 2, 6), persistent=False)
        s.embed = nn.Embedding(len(world.VOCAB) + 1, d, padding_idx=0); s.lpos = nn.Parameter(torch.randn(1, world.L_MAX, d) * .02)
        s.lang = Block(d, heads, mlp); s.state = nn.Linear(state, d, bias=False)

    def image(s, im):
        return s.vis(im.flatten(2).transpose(1, 2)) + s.pos(s.grid)

    def stem(s, im):
        """The same tokens as a [B,d,6,6] grid, for conv fusions."""
        return s.image(im).transpose(1, 2).reshape(im.shape[0], -1, vision.CELLS, vision.CELLS)

    def language(s, tok):
        keep = tok != 0; keep = keep | (~keep.any(1, keepdim=True))  # an all-padding sentence still has tokens to attend to
        return s.lang(s.embed(tok) + s.lpos, mask=keep), keep


def pool(x, keep):
    k = keep[..., None].float(); return (x * k).sum(1) / k.sum(1).clamp_min(1)


class FiLMConv(nn.Module):
    def __init__(s, d, n):
        super().__init__(); s.a = nn.ModuleList(nn.Conv2d(d, d, 3, 1, 1, bias=False) for _ in range(n)); s.b = nn.ModuleList(nn.Conv2d(d, d, 3, 1, 1, bias=False) for _ in range(n)); s.f = nn.ModuleList(nn.Linear(d, 2 * d, bias=False) for _ in range(n)); s.n = nn.ModuleList(nn.GroupNorm(2, d) for _ in range(n))

    def forward(s, x, l):
        for a, b, f, n in zip(s.a, s.b, s.f, s.n):
            g, h = f(l).chunk(2, -1); y = n(F.silu(a(x))); y = (1 + g[..., None, None]) * y + h[..., None, None]; x = x + b(F.silu(y))
        return x


class Fusion(nn.Module):
    def __init__(s, d, heads, layers, mlp, state=4):
        super().__init__(); s.p = Parts(d, heads, mlp, state); s.build(d, heads, layers, mlp)

    def inputs(s, im, tok, st):
        l, keep = s.p.language(tok); return l, keep, s.p.state(st)[:, None]


class Late(Fusion):
    def build(s, d, heads, layers, mlp):
        s.v = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers)); s.j = nn.Sequential(nn.Linear(3 * d, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); v = s.p.image(im)
        for b in s.v: v = b(v)
        return s.j(torch.cat([v.mean(1), pool(l, keep), x[:, 0]], -1))[:, None]


class Film(Fusion):
    def build(s, d, heads, layers, mlp):
        s.c = FiLMConv(d, layers); s.t = Block(d, heads, mlp)

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); v = s.p.stem(im); v = s.c(v, pool(l, keep)).flatten(2).transpose(1, 2)
        return s.t(torch.cat([v, x], 1))


class Token(Fusion):
    def build(s, d, heads, layers, mlp):
        s.b = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); y = torch.cat([s.p.image(im), l, x], 1)
        m = torch.cat([torch.ones(l.shape[0], 36, dtype=torch.bool, device=l.device), keep, torch.ones(l.shape[0], 1, dtype=torch.bool, device=l.device)], 1)
        for b in s.b: y = b(y, mask=m)
        return y


class Cross(Fusion):
    gated = False

    def build(s, d, heads, layers, mlp):
        s.v = nn.ModuleList(Block(d, heads, mlp) for _ in range(max(1, layers // 2))); s.b = nn.ModuleList(Block(d, heads, mlp, cross=True, gated=s.gated) for _ in range(layers))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); v = s.p.image(im)
        for b in s.v: v = b(v)
        y = torch.cat([l, x], 1); m = torch.cat([keep, torch.ones(l.shape[0], 1, dtype=torch.bool, device=l.device)], 1)
        for b in s.b: y = b(y, v, mask=m)
        return y


class Gated(Cross):
    gated = True


class TokenLearner(Fusion):
    def build(s, d, heads, layers, mlp):
        s.c = FiLMConv(d, max(1, layers // 2)); s.a = nn.Conv2d(d, 8, 1, bias=False); s.b = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); v = s.c(s.p.stem(im), pool(l, keep))
        w = s.a(v).flatten(2).softmax(-1); y = torch.bmm(w, v.flatten(2).transpose(1, 2)); y = torch.cat([y, x], 1)
        for b in s.b: y = b(y)
        return y


class Resampler(Fusion):
    def build(s, d, heads, layers, mlp):
        s.lat = nn.Parameter(torch.randn(1, 8, d) * .02); s.r = Block(d, heads, mlp, cross=True); s.b = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); B = l.shape[0]; z = s.r(s.lat.expand(B, -1, -1), s.p.image(im))
        y = torch.cat([z, l, x], 1); m = torch.cat([torch.ones(B, 8, dtype=torch.bool, device=l.device), keep, torch.ones(B, 1, dtype=torch.bool, device=l.device)], 1)
        for b in s.b: y = b(y, mask=m)
        return y


class LangAttn(Fusion):
    def build(s, d, heads, layers, mlp):
        s.v = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers)); s.q = nn.Linear(d, d, bias=False); s.k = nn.Linear(d, d, bias=False); s.xy = nn.Linear(2, d, bias=False); s.temp = nn.Parameter(torch.tensor(1.))
        g = (torch.arange(6) + .5) / 6; y, x = torch.meshgrid(g, g, indexing="ij"); s.register_buffer("grid", torch.stack([x.flatten(), y.flatten()], 1))

    def forward(s, im, tok, st):
        l, keep, x = s.inputs(im, tok, st); v = s.p.image(im)
        for b in s.v: v = b(v)
        lp = pool(l, keep); w = (torch.einsum("bd,bnd->bn", s.q(lp), s.k(v)) / v.shape[-1] ** .5 * s.temp).softmax(-1)
        return torch.stack([(w[..., None] * v).sum(1), s.xy(w @ s.grid), lp, x[:, 0]], 1)


FUSIONS = dict(late=Late, film=Film, token=Token, cross=Cross, gated=Gated, tokenlearner=TokenLearner, resampler=Resampler, langattn=LangAttn)
BASELINES = tuple(FUSIONS)  # Published modules. A new design joins FUSIONS, not this tuple.
CONTROLS = ("blind", "shuffle", "oracle")
HEADS = tuple(HORIZON) + ("answer",)


class Policy(nn.Module):
    """Spec is "fusion+head", such as token+flow."""

    def __init__(s, spec, d, heads, layers, mlp):
        super().__init__(); s.spec = spec; f, s.head = spec.split("+"); s.fname = f; s.control = f if f in CONTROLS else None
        s.fusion = FUSIONS["token" if s.control else f](d, heads, layers, mlp, state=6 if f == "oracle" else 4)
        s.dec = Block(d, heads, mlp, cross=True)
        if s.head == "answer":
            s.query = nn.Parameter(torch.randn(1, 1, d) * .02); s.out = nn.Linear(d, len(world.ANSWERS), bias=False)
        else:
            s.H = HORIZON[s.head]; s.query = nn.Parameter(torch.randn(1, s.H, d) * .02); s.out = nn.Linear(d, 3, bias=False)
            if s.head == "flow": s.ain = nn.Linear(3, d, bias=False); s.tin = nn.Linear(1, d, bias=False)

    def context(s, im, tok, st, goal=None):
        if s.control == "blind": tok = torch.zeros_like(tok)
        if s.control == "shuffle": tok = tok.roll(1, 0)
        if s.control == "oracle": st = torch.cat([st, goal], 1)
        return s.fusion(im, tok, st)

    def decode(s, ctx, noisy=None, t=None):
        q = s.query.expand(ctx.shape[0], -1, -1)
        if noisy is not None: q = q + s.ain(noisy) + s.tin(t[:, None, None].expand(-1, s.H, 1))
        return s.out(s.dec(q, ctx))

    def loss(s, im, tok, st, target, goal=None, g=None):
        ctx = s.context(im, tok, st, goal)
        if s.head == "answer": return F.cross_entropy(s.decode(ctx)[:, 0], target)
        target = target[:, :s.H]
        if s.head == "flow":
            e = torch.randn(target.shape, generator=g, device=target.device); t = torch.rand(target.shape[0], generator=g, device=target.device)
            x = (1 - t[:, None, None]) * e + t[:, None, None] * target
            return F.mse_loss(s.decode(ctx, x, t), target - e)
        return F.mse_loss(s.decode(ctx), target)

    @torch.no_grad()
    def act(s, im, tok, st, goal=None, g=None):
        """Actions for the next EXECUTE steps, as [batch, steps, 3]."""
        ctx = s.context(im, tok, st, goal)
        if s.head == "flow":
            x = torch.randn(ctx.shape[0], s.H, 3, generator=g, device=ctx.device)
            for i in range(FLOW_STEPS): x = x + s.decode(ctx, x, torch.full((ctx.shape[0],), i / FLOW_STEPS, device=ctx.device)) / FLOW_STEPS
            return x[:, :EXECUTE[s.head]]
        return s.decode(ctx)[:, :EXECUTE[s.head]]

    @torch.no_grad()
    def answer(s, im, tok, st):
        return s.decode(s.context(im, tok, st))[:, 0]


def size(m):
    return sum(p.numel() for p in m.parameters())


@functools.lru_cache(maxsize=None)
def fit(spec, target, heads, layers):
    """Width and MLP width that bring this spec closest to `target` weights, so every arm has the same number of weights."""
    build = lambda d, h: size(Policy(spec, d, heads, layers, h))
    lo, hi = 1, 64
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if build(mid * heads, mid * heads) <= target: lo = mid
        else: hi = mid
    d = lo * heads; lo, hi = max(4, d // 4), d * 8
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if build(d, mid) <= target: lo = mid
        else: hi = mid
    return d, lo
