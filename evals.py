"""Scores. Every score keeps one number per item, so any two arms compare with a paired standard error. There is no combined number.

Closed loop (fraction of episodes, higher is better unless noted):
nominal.<family>   The original task of a trained layout, on new jitter. What most robot benchmarks report.
swap.<family>      Same layouts and jitter range, a held task asking for a different object. No regime trains it. 2610.06235 target replacement; LangGap's Change Object, Change Target, Spatial Description.
swap.orig          How often the policy acted on the original object when asked for another. Lower is better. 2610.06235.
nonsense.orig      How often it acted on the original object given a random sentence. Lower is better. 2610.06235 random sentence.
new.<family>       New layouts.
combo.object       Target's colour-shape pair never named in training.
wording.object     Known words in an order no training row uses.
more.object        7 objects in place of 5.
recover.object     The target jumps away at the middle of the episode, which open-loop memorizing can't follow.

Open loop:
off.mse            Action error against the expert on a trained task. Lower is better. 2609.23048 shows these can pass while closed loop fails.
off.swap           How far the first action moves when only the sentence changes.
probe.target       Linear decoding of the named object from pooled context, half of items fit, half tested. 2610.06235 finds this stays high while control fails.

Answering, for head `answer`: answer.attr, answer.count, answer.qspatial, answer.near. These are the vision-language questions asked of the same scenes.
"""
import numpy as np, torch
import world as W, models as M
import vision

LOWER = ("swap.orig", "nonsense.orig", "off.mse")
FAM = W.FAMILIES


def higher(score):
    return score not in LOWER


def rollout(model, b, device, g, jump=False):
    t = 0; goal = lambda: torch.as_tensor(b.goal(), device=device)
    E = M.EXECUTE[model.head]
    while t < W.T:
        im = vision.encode(b.render(), device)
        a = model.act(im, torch.as_tensor(b.tok, device=device), torch.as_tensor(b.state(), device=device), goal=goal(), g=g).cpu().numpy()
        for i in range(E):
            if t >= W.T: break
            b.step(a[:, i]); t += 1
            if jump and t == W.T // 2:
                r = b.rng; idx = np.arange(b.n); new = np.empty((b.n, 2), np.float32)
                for j in range(b.n):
                    while True:
                        p = r.uniform(.12, .88, 2)
                        if np.linalg.norm(p - b.pos[j, b.a[j]]) > .3 and (b.hold[j] < 0): break
                        if b.hold[j] >= 0: p = b.pos[j, b.a[j]]; break
                    new[j] = p
                b.pos[idx, b.a] = new
    return b


def repeat(xs, n):
    return [xs[i % len(xs)] for i in range(n)]


def items_nominal(table, fam, n):
    t = [x for x in table if x["orig"]["fam"] == fam] or table[:1]
    if t[0]["orig"]["fam"] != fam: return None
    return repeat(t, n)


def closed(model, name, layouts, tasks, world, role, device, texts=None, jump=False):
    if not tasks: return None
    rng = np.random.default_rng([world, sum(map(ord, name)), sum(map(ord, role))])
    g = torch.Generator(device=device).manual_seed(int(rng.integers(1 << 31)))
    return rollout(model, W.Batch(layouts, tasks, rng, texts=texts), device, g, jump=jump)


def first_frames(b, device):
    return vision.encode(b.render(), device), torch.as_tensor(b.state(), device=device)


@torch.no_grad()
def score(model, table, world, n, role="report", device="cpu", scope=None):
    """All scores for one role: report, tune or final. Roles use different held tasks and different jitter."""
    if role not in ("report", "tune", "final"): raise ValueError(role)
    model.eval(); out = {}; tab = table if scope is None else table[:scope]
    if model.head == "answer":
        return answers(model, world, n, role, device)
    r = np.random.default_rng([world, 55, sum(map(ord, role))])
    for fam in FAM:
        t = items_nominal(tab, fam, n)
        if t: out[f"nominal.{fam}"] = closed(model, f"nominal.{fam}", [x["layout"] for x in t], [x["orig"] for x in t], world, role, device).success().astype(float)
    orig_hits = []
    for fam in FAM:
        t = repeat(tab, n); tasks = [x[role][fam] for x in t]
        b = closed(model, f"swap.{fam}", [x["layout"] for x in t], tasks, world, role, device)
        out[f"swap.{fam}"] = b.success().astype(float)
        orig_hits.append(b.commit() == np.array([x["orig"]["a"] for x in t]))
    out["swap.orig"] = np.concatenate(orig_hits).astype(float)
    t = repeat(tab, n)
    b = closed(model, "nonsense", [x["layout"] for x in t], [x[role]["object"] for x in t], world, role, device, texts=W.nonsense(r, n))
    out["nonsense.orig"] = (b.commit() == np.array([x["orig"]["a"] for x in t])).astype(float)
    for fam in FAM:
        it = [W.fresh_item(r, fam, world) for _ in range(n)]
        out[f"new.{fam}"] = closed(model, f"new.{fam}", [i[0] for i in it], [i[1] for i in it], world, role, device).success().astype(float)
    it = [W.fresh_item(r, "object", world, combo=True) for _ in range(n)]
    out["combo.object"] = closed(model, "combo", [i[0] for i in it], [i[1] for i in it], world, role, device).success().astype(float)
    t = repeat(tab, n); tasks = [x[role]["object"] for x in t]
    tx = [str(r.choice(W.HELD_T)).format(a=W.phrase(x["layout"]["pairs"][tk["a"]], r)) for x, tk in zip(t, tasks)]
    out["wording.object"] = closed(model, "wording", [x["layout"] for x in t], tasks, world, role, device, texts=tx).success().astype(float)
    it = [W.fresh_item(r, "object", world, k=W.K_MAX) for _ in range(n)]
    out["more.object"] = closed(model, "more", [i[0] for i in it], [i[1] for i in it], world, role, device).success().astype(float)
    it = [W.fresh_item(r, "object", world) for _ in range(n)]
    out["recover.object"] = closed(model, "recover", [i[0] for i in it], [i[1] for i in it], world, role, device, jump=True).success().astype(float)
    out.update(open_loop(model, tab, world, n, role, device, r))
    return {k: v.astype(np.float32) for k, v in out.items()}


def open_loop(model, tab, world, n, role, device, r):
    out = {}; g = lambda: torch.Generator(device=device).manual_seed(int(world) + 11)
    t = repeat(tab, n); b = W.Batch([x["layout"] for x in t], [x["orig"] for x in t], r); err = np.zeros(n)
    for _ in range(W.T // 3):
        im = vision.encode(b.render(), device)
        a = model.act(im, torch.as_tensor(b.tok, device=device), torch.as_tensor(b.state(), device=device), goal=torch.as_tensor(b.goal(), device=device), g=g()).cpu().numpy()[:, 0]
        e = b.expert(); err += ((a - e) ** 2).mean(1) / (W.T // 3); b.step(e)
    out["off.mse"] = err
    b = W.Batch([x["layout"] for x in t], [x["orig"] for x in t], r); im, st = first_frames(b, device); goal = torch.as_tensor(b.goal(), device=device)
    a1 = model.act(im, torch.as_tensor(b.tok, device=device), st, goal=goal, g=g()).cpu().numpy()[:, 0, :2]
    tok2 = np.stack([W.encode(x[role]["object"]["text"]) for x in t])
    a2 = model.act(im, torch.as_tensor(tok2, device=device), st, goal=goal, g=g()).cpu().numpy()[:, 0, :2]
    out["off.swap"] = np.linalg.norm(a1 - a2, axis=1)
    m = max(n * 4, 16); t = repeat(tab, m); tasks = [W.task(x["layout"], "object", a=int(r.integers(W.K)), rng=r) for x in t]
    b = W.Batch([x["layout"] for x in t], tasks, r); im, st = first_frames(b, device)
    c = model.context(im, torch.as_tensor(b.tok, device=device), st, goal=torch.as_tensor(b.goal(), device=device)).mean(1).cpu().numpy()
    y = np.array([W.PAIRS.index(x["layout"]["pairs"][tk["a"]]) for x, tk in zip(t, tasks)])
    out["probe.target"] = probe(c, y)
    return out


def probe(x, y, ridge=1.0):
    """Ridge on half the items, tested on the other half: 2610.06235's question of whether the target is readable. Per-item hits come back."""
    np.seterr(all="ignore")  # numpy's Accelerate matmul raises spurious warnings on macOS; check.py verifies finite results
    h = len(x) // 2; mu, sd = x[:h].mean(0), np.maximum(x[:h].std(0), 1e-3); a = np.c_[(x - mu) / sd, np.ones(len(x))].astype(np.float64)
    Y = np.eye(len(W.PAIRS))[y[:h]]
    w = np.linalg.solve(a[:h].T @ a[:h] + ridge * np.eye(a.shape[1]), a[:h].T @ Y)
    return ((a[h:] @ w).argmax(1) == y[h:]).astype(float)


def answer_rows(world, n, role_seed, kinds=tuple(W.QUESTION_T)):
    r = np.random.default_rng([world, 77, role_seed]); out = []
    for kind in kinds:
        rows = []
        while len(rows) < n:
            lay = W.make_layout(r); q = W.question(lay, kind, r)
            if q: rows.append((lay, q[0], W.ANSWERS.index(q[1])))
        out.append((kind, rows))
    return out, r


def answers(model, world, n, role, device):
    out = {}; rows, r = answer_rows(world, n, 100 + sum(map(ord, role)))
    for kind, it in rows:
        b = W.Batch([x[0] for x in it], [W.task(x[0], "object", a=0, rng=r) for x in it], r, texts=[x[1] for x in it]); im, st = first_frames(b, device)
        p = model.answer(im, torch.as_tensor(b.tok, device=device), st).argmax(1).cpu().numpy()
        out[f"answer.{kind}"] = (p == np.array([x[2] for x in it])).astype(np.float32)
    return out
