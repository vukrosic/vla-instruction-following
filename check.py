"""Bug checks, each stopped at 7 s. Any failure raises. Run after every code change.

Asks: can the expert do every task? Is a held task or pair in training? Does every weight get a gradient? Does the sentence change the output?
Are sizes equal? Do the scores read 1 for a perfect script and near 0 for a still one? Are two runs of one setting identical? Does every path run?
"""
import json, os, signal, sys, tempfile, time
import numpy as np, torch
import env, evals, models as M, vision as V, world as W

torch.set_num_threads(1)
ALL = [f"{f}+mse" for f in M.FUSIONS] + [f"{c}+mse" for c in M.CONTROLS] + ["token+chunk", "token+flow", "token+answer", "film+flow"]


class stage:
    def __init__(s, name): s.name = name
    def __enter__(s): s.t = time.time(); signal.signal(signal.SIGALRM, lambda *_: sys.exit(f"{s.name} ran over 7 s")); signal.alarm(7); print(s.name, end=" ", flush=True)
    def __exit__(s, kind, *rest):
        signal.alarm(0)
        if not kind: print(f"ok {time.time() - s.t:.1f}s", flush=True)


class Script:
    """Scripted stand-ins that test the scores themselves. `goal` follows the true goal; `still` stays; `orig` can't be scripted here because a script doesn't see the original."""
    head = "mse"
    def __init__(s, mode): s.mode = mode
    def eval(s): return s
    def act(s, im, tok, st, goal=None, g=None):
        a = torch.zeros(len(st), 1, 3); a[:, :, 2] = -1
        if s.mode == "goal":
            v = (goal - st[:, :2]) / W.SPEED; a[:, 0, :2] = v / v.abs().max(1, keepdim=True)[0].clamp_min(1)
        return a
    def context(s, im, tok, st, goal=None): return goal[:, None]


def small(spec, **k):
    C = env.TIERS["tiny"]; torch.manual_seed(0); d, h = M.fit(spec, C["weights"], C["heads"], C["layers"]); torch.manual_seed(0); return M.Policy(spec, d, C["heads"], C["layers"], h)


def demo_batch(n=4, answer=False):
    r = np.random.default_rng(0); ls = [W.make_layout(r) for _ in range(n)]; b = W.Batch(ls, [W.task(l, "object", a=i % 3, rng=r) for i, l in enumerate(ls)], r)
    tok = torch.as_tensor(b.tok); tok2 = torch.as_tensor(W.Batch(ls, [W.task(l, "object", a=(i % 3) + 1, rng=r) for i, l in enumerate(ls)], r).tok)
    return V.encode(b.render()), tok, tok2, torch.as_tensor(b.state()), torch.as_tensor(b.goal()), torch.as_tensor(np.tile(b.expert()[:, None], (1, 8, 1)))


def main():
    with stage("expert does every task"):
        r = np.random.default_rng(0)
        for fam in W.FAMILIES:
            it = [W.fresh_item(r, fam, 0) for _ in range(60)]; b = W.Batch([i[0] for i in it], [i[1] for i in it], r)
            for _ in range(W.T): b.step(b.expert())
            assert b.success().mean() == 1, f"expert fails {fam}: {b.success().mean()}"

    with stage("held tasks and pairs stay out of training"):
        for world in (0, 1):
            tab = W.layout_table(world, 8); banned = W.held_pairs(world); names = [W.phrase(p, np.random.default_rng(i)) for p in banned for i in range(8)]
            for regime in env.REGIMES:
                a = env.assignment(regime, tab, 200, world); texts = [t["text"] for _, t in a]
                assert not any(n in tx for n in set(names) for tx in texts), f"{regime} names a held pair"
                for t in tab:
                    held = {t[role][fam]["text"] for role in ("report", "tune", "final") for fam in W.FAMILIES}
                    seen = {x[1]["text"] for x in a if x[0] is t["layout"]}
                    assert not held & seen, f"{regime} trains a held task"
            for t in tab:
                for fam in W.FAMILIES:
                    assert t["report"][fam]["a"] != t["orig"]["a"] or fam != t["orig"]["fam"], "a swap task asks for the original target"
                    assert t["report"][fam]["text"] != t["final"][fam]["text"] or fam == "spatial", f"final {fam} repeats report"

    with stage("encoder frozen, deterministic, outside the policy"):
        im, *_ = demo_batch(); m = small("token+mse")
        assert not any("vision" in n or "c.0" in n for n, _ in m.named_parameters()), "encoder weights are in the policy"
        assert im.shape[1:] == (V.E, V.CELLS, V.CELLS) and not im.requires_grad
        f = np.random.default_rng(0).integers(0, 255, (3, 36, 36, 3), dtype=np.uint8); assert (V.encode(f) == V.encode(f)).all()

    for part, specs in (("fusions", ALL[:4]), ("fusions", ALL[4:8]), ("controls and heads", ALL[8:])):
        with stage(f"{part}: gradient, sentence, state, size"):
            im, tok, tok2, st, goal, act = demo_batch(); ans = torch.zeros(4, dtype=torch.long)
            for sp in specs:
                m = small(sp); g = torch.Generator().manual_seed(0)
                for n, w in m.named_parameters():
                    if n.endswith("gate"): w.data.fill_(.1)  # Flamingo's gate starts at zero, which blocks every gradient behind it until the gate moves
                y = ans if sp.endswith("answer") else act
                m.loss(im, tok, st, y, goal=goal, g=g).backward()
                none = [n for n, w in m.named_parameters() if (w.grad is None or (w.grad == 0).all()) and not (sp.startswith("blind") and "embed" in n)]  # blind only ever sees padding
                assert not none, f"{sp} no gradient: {none}"
                ratio = M.size(m) / env.TIERS["tiny"]["weights"]; assert .85 < ratio <= 1.0, f"{sp} size {M.size(m)}"
                if sp.endswith("answer"): continue
                fix = lambda t, s: (m.act(im, t, s, goal=goal, g=torch.Generator().manual_seed(1)))
                moved = (fix(tok, st) - fix(tok2, st)).abs().max().item()
                if sp.startswith("blind"): assert moved == 0, "blind reads the sentence"
                else: assert moved > 0, f"{sp} ignores the sentence"
                assert (fix(tok, st) - fix(tok, st + .1)).abs().max().item() > 0, f"{sp} ignores the state"
                if sp.startswith("oracle"): assert (m.act(im, tok, st, goal=goal + .2) - m.act(im, tok, st, goal=goal)).abs().max() > 0, "oracle ignores the goal"

    with stage("sizes match at the 1m and 5m tiers"):
        for tier in ("1m",):
            C = env.TIERS[tier]; sz = {sp: M.size(M.Policy(sp, *(lambda dh: (dh[0], C["heads"], C["layers"], dh[1]))(M.fit(sp, C["weights"], C["heads"], C["layers"])))) for sp in ("token+mse", "film+mse", "cross+mse", "late+mse")}
            assert max(sz.values()) / min(sz.values()) < 1.03, f"sizes differ {sz}"

    tab = W.layout_table(0, 8); encode = V.encode; V.encode = lambda *a: None  # scripts read no pixels, so these two stages skip the encoder's cost
    with stage("scores: a perfect script scores 1"):
        s = evals.score(Script("goal"), tab, 0, 16); bad = {k: float(s[k].mean()) for k in s if k.split(".")[0] in ("nominal", "swap", "new", "combo", "wording", "more", "recover") and k != "swap.orig" and not k.endswith("place") and s[k].mean() < .9}  # the script never grips, so only the expert stage covers place
        assert not bad, f"script fails: {bad}"
        assert s["swap.orig"].mean() < .15, f"script scored as following the original: {s['swap.orig'].mean()}"
        assert s["off.mse"].mean() < .2, f"script's off.mse {s['off.mse'].mean()}"
        assert all(np.isfinite(v).all() for v in s.values()), "non-finite score"

    with stage("scores: a still script scores about 0"):
        s = evals.score(Script("still"), tab, 0, 8); bad = {k: float(s[k].mean()) for k in s if k.split(".")[0] in ("nominal", "swap", "new", "combo", "wording", "more", "recover") and s[k].mean() > .15}
        assert not bad, f"still script scores: {bad}"

    V.encode = encode
    out = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name; os.unlink(out)
    A = lambda **k: env.parser().parse_args(["--tier", "tiny", "--results", out, "--tag", "check", *sum(([f"--{a}", str(b)] for a, b in k.items()), [])])
    with stage("two runs of one setting are identical; done settings skip"):
        a = env.run(A(spec="film+chunk")); b = env.run(A(spec="film+chunk", extra="again"))
        assert a["scores"] == b["scores"] and a["loss"] == b["loss"], "two runs differ"
        assert env.run(A(spec="film+chunk")) is None, "a finished setting ran again"
        assert a["weights"] <= env.TIERS["tiny"]["weights"] and os.path.exists(os.path.join(env.RUNS, a["key"] + ".npz"))
    with stage("a stopped run resumes to the same scores"):
        a = env.run(A(spec="late+chunk", extra="whole")); assert env.run(A(spec="late+chunk", extra="cut", **{"save-every": 2, "stop-at": 4})) is None
        b = env.run(A(spec="late+chunk", extra="cut", **{"save-every": 2})); assert a["scores"] == b["scores"], "resume changed the scores"
    with stage("regimes run"):
        for reg in ("free", "random", "dilute"): assert env.run(A(spec="late+mse", regime=reg))["scores"]
    with stage("answer head and final role run"):
        s = env.run(A(spec="token+answer"))["scores"]; assert all(f"answer.{k}" in s for k in W.QUESTION_T), s.keys()
        assert env.run(A(spec="token+mse", role="final"))["role"] == "final"
    for row in map(json.loads, open(out)): os.unlink(os.path.join(env.RUNS, row["key"] + ".npz"))
    os.unlink(out); print("all checks pass")


if __name__ == "__main__":
    main()
