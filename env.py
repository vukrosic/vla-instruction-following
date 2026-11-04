"""The environment: train one policy on one kind of data at one budget, and write the same scores for every arm.

python3 env.py --spec token+flow --regime coupled --tier 1m --lr 1e-3 [--world 1] [--role report|tune|final] [--tag x] [--save] [--device cpu]

--tier sets weights, demonstrations and steps, so every arm trains on the same frames in the same order. --regime sets which layout-task pairs are demonstrated:
coupled  One task per layout, like LIBERO.
free     Several tasks per layout, the held ones excluded, with the same number of demonstrations. LangGap's same-scene data.
random   A new layout for every demonstration, with the same number of demonstrations.
small    Free data on half the layouts only.
dilute   `small` plus coupled data on the other half: the same demonstrations on the first half, four times more elsewhere. LangGap's dilution setting.

--world changes layouts, tasks, held colour-shape pairs and starting weights. Rows go to results.jsonl with a key per setting, so a finished setting never reruns. Per-item scores go to runs/.
"""
import argparse, fcntl, hashlib, json, math, os, signal, time
import numpy as np, torch
import world as W, models as M, evals, vision as V

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs"); CKPT = os.path.join(HERE, "ckpt"); RESULTS = os.path.join(HERE, "results.jsonl")
TEST_VERSION = 1  # bump after any change to the world or a score, so older rows are never reused; old rows then move to results_vN.jsonl
TIERS = {
    "tiny": dict(weights=8000, heads=2, layers=1, layouts=4, demos=8, steps=6, batch=4, items=1),  # checks code only, its scores mean nothing
    "5s": dict(weights=30000, heads=4, layers=2, layouts=8, demos=40, steps=120, batch=16, items=4),
    "1m": dict(weights=60000, heads=4, layers=2, layouts=8, demos=200, steps=1500, batch=32, items=20),
    "5m": dict(weights=200000, heads=4, layers=3, layouts=16, demos=800, steps=5000, batch=64, items=50),
    "1h": dict(weights=1000000, heads=8, layers=4, layouts=32, demos=4000, steps=40000, batch=64, items=150),
    "5h": dict(weights=4000000, heads=8, layers=6, layouts=64, demos=16000, steps=120000, batch=128, items=300),
}
REGIMES = ("coupled", "free", "random", "small", "dilute")
DART = 0.15  # noise on executed demonstration actions, so the demos visit states off the expert's line


def assignment(regime, table, demos, world):
    """Which layout and task each demonstration uses."""
    r = np.random.default_rng([world, 3, sum(map(ord, regime))]); n = len(table); half = max(1, n // 2); out = []
    def free(t, i): 
        tasks = t["free"] + [t["orig"]]; return t["layout"], tasks[i % len(tasks)]
    if regime == "coupled": out = [(table[i % n]["layout"], table[i % n]["orig"]) for i in range(demos)]
    elif regime == "free": out = [free(table[i % n], i // n) for i in range(demos)]
    elif regime == "random": out = [W.fresh_item(r, W.FAMILIES[i % 4], world) for i in range(demos)]
    elif regime == "small": out = [free(table[i % half], i // half) for i in range(max(half, demos // 5))]
    elif regime == "dilute":
        out = [free(table[i % half], i // half) for i in range(max(half, demos // 5))]
        rest = table[half:] or table
        out += [(rest[i % len(rest)]["layout"], rest[i % len(rest)]["orig"]) for i in range(demos - len(out))]
    else: raise ValueError(regime)
    return out


def scope_of(regime, table):
    return max(1, len(table) // 2) if regime in ("small", "dilute") else len(table)


def demonstrations(regime, table, demos, world):
    """Expert rollouts with noise on executed actions. Labels are the expert's action at the state actually visited."""
    a = assignment(regime, table, demos, world); r = np.random.default_rng([world, 4, sum(map(ord, regime))])
    b = W.Batch([x[0] for x in a], [x[1] for x in a], r); H = max(M.HORIZON.values())
    F, S, G, A = [], [], [], []
    for _ in range(W.T):
        F.append(b.render()); S.append(b.state()); G.append(b.goal()); e = b.expert(); A.append(e.copy()); x = e.copy(); x[:, :2] += r.normal(0, DART, x[:, :2].shape); b.step(x)
    A = np.stack(A); chunks = np.stack([A[np.minimum(t + np.arange(H), W.T - 1)] for t in range(W.T)])  # [T, H, n, 3]
    n = len(a)
    return dict(frames=np.concatenate(F), state=np.concatenate(S), goal=np.concatenate(G), tok=np.tile(b.tok, (W.T, 1)), act=chunks.transpose(0, 2, 1, 3).reshape(W.T * n, H, 3))


def answer_data(world, rows):
    """Questions about fresh scenes, read by head `answer`. No held colour-shape pair is named."""
    r = np.random.default_rng([world, 5]); banned = W.held_pairs(world); F, T, Y, S = [], [], [], []
    per = 256; lay = []; tx = []; ans = []
    while len(lay) < rows:
        l = W.make_layout(r); kind = list(W.QUESTION_T)[int(r.integers(4))]; q = W.question(l, kind, r)
        if q is None or any(p in banned for p in l["pairs"]): continue
        lay.append(l); tx.append(q[0]); ans.append(W.ANSWERS.index(q[1]))
    for i in range(0, rows, per):
        b = W.Batch(lay[i:i + per], [W.task(l, "object", a=0, rng=r) for l in lay[i:i + per]], r, texts=tx[i:i + per]); F.append(b.render()); S.append(b.state()); T.append(b.tok)
    return dict(frames=np.concatenate(F), state=np.concatenate(S), goal=np.zeros((rows, 2), np.float32), tok=np.concatenate(T), act=np.array(ans))


def key_of(a):
    s = json.dumps(dict(v=TEST_VERSION, spec=a.spec, regime=a.regime, tier=a.tier, world=a.world, lr=a.lr, role=a.role, tag=a.tag, extra=a.extra), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--spec", default="token+mse"); p.add_argument("--regime", default="coupled"); p.add_argument("--tier", default="tiny"); p.add_argument("--world", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--role", default="report"); p.add_argument("--tag", default=""); p.add_argument("--extra", default=""); p.add_argument("--device", default="cpu"); p.add_argument("--save", action="store_true"); p.add_argument("--force", action="store_true"); p.add_argument("--save-every", type=int, default=0, help="save resume state every N steps"); p.add_argument("--stop-at", type=int, default=0, help="stop at this step, for testing resume")
    p.add_argument("--results", default=RESULTS); return p


def build(a):
    C = TIERS[a.tier]; torch.manual_seed(1000 * a.world + 7); d, mlp = M.fit(a.spec, C["weights"], C["heads"], C["layers"])
    torch.manual_seed(1000 * a.world + 7); return M.Policy(a.spec, d, C["heads"], C["layers"], mlp).to(a.device), d, mlp


def run(a):
    C = TIERS[a.tier]; key = key_of(a)
    if not a.force and os.path.exists(a.results) and any(json.loads(l)["key"] == key for l in open(a.results)):
        print("already done", key); return None
    if a.device == "cpu": torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 1)))
    table = W.layout_table(a.world, C["layouts"]); answer = a.spec.endswith("+answer")
    data = answer_data(a.world, C["demos"] * W.T) if answer else demonstrations(a.regime, table, C["demos"], a.world)
    cells = V.encode(data.pop("frames"), a.device); N = len(cells)
    model, d, mlp = build(a); opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(.9, .95))
    order = np.random.default_rng([a.world, 6, C["steps"]]); g = torch.Generator(device=a.device).manual_seed(1000 * a.world + 8)
    t0 = time.time(); ema = None; first = None; start = 0; state = os.path.join(CKPT, key + ".resume.pt")
    if a.save_every and os.path.exists(state):
        z = torch.load(state, map_location=a.device, weights_only=False); model.load_state_dict(z["model"]); opt.load_state_dict(z["opt"]); order.bit_generator.state = z["order"]; g.set_state(z["g"].cpu()) if a.device == "cpu" else g.set_state(z["g"]); start, first, ema, spent = z["step"], z["first"], z["ema"], z["spent"]; t0 -= spent; torch.set_rng_state(z["rng"])
    model.train()
    for step in range(start, C["steps"]):
        if a.stop_at and step == a.stop_at: return None
        if a.save_every and step and step % a.save_every == 0:
            os.makedirs(CKPT, exist_ok=True); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), order=order.bit_generator.state, g=g.get_state(), step=step, first=first, ema=ema, spent=time.time() - t0, rng=torch.get_rng_state()), state + ".tmp"); os.replace(state + ".tmp", state)
        for p in opt.param_groups: p["lr"] = a.lr * min(1, (step + 1) / 50) * .5 * (1 + math.cos(math.pi * step / C["steps"]))
        i = order.integers(N, size=C["batch"])
        im = cells[torch.as_tensor(i, device=a.device)]
        tok = torch.as_tensor(data["tok"][i], device=a.device); st = torch.as_tensor(data["state"][i], device=a.device); goal = torch.as_tensor(data["goal"][i], device=a.device); y = torch.as_tensor(data["act"][i], device=a.device)
        loss = model.loss(im, tok, st, y, goal=goal, g=g); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if not torch.isfinite(loss): raise SystemExit(f"loss went non-finite at step {step}")
        first = loss.item() if first is None else first; ema = loss.item() if ema is None else .98 * ema + .02 * loss.item()
    train_s = time.time() - t0; t1 = time.time()
    scores = evals.score(model, table, a.world, C["items"], a.role, a.device, scope=scope_of(a.regime, table))
    row = dict(key=key, v=TEST_VERSION, spec=a.spec, regime=a.regime, tier=a.tier, world=a.world, lr=a.lr, role=a.role, tag=a.tag, extra=a.extra, width=d, mlp=mlp, weights=M.size(model), loss0=round(first, 4), loss=round(ema, 4), train_s=round(train_s, 1), eval_s=round(time.time() - t1, 1), scores={k: round(float(v.mean()), 4) for k, v in scores.items()})
    if a.save_every and os.path.exists(state): os.unlink(state)
    os.makedirs(RUNS, exist_ok=True); np.savez_compressed(os.path.join(RUNS, key + ".npz"), **scores)
    with open(a.results, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(json.dumps(row) + "\n")
    if a.save: os.makedirs(CKPT, exist_ok=True); torch.save(model.state_dict(), os.path.join(CKPT, key + ".pt"))
    return row


if __name__ == "__main__":
    r = run(parser().parse_args())
    if r: print(json.dumps({k: v for k, v in r.items() if k != "scores"})); print(json.dumps(r["scores"]))
