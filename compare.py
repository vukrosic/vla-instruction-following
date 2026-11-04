"""Tables from results.jsonl. There is no combined number: read one score at a time.

python3 compare.py --tier 1m --regime free --ref token+flow    Mean of each score per arm, and the paired z of each arm against --ref.
Each arm uses the lr that tuning rows picked. Rows with no tuning rows use the lr they have, and the table says so.
"""
import argparse, json, os
import numpy as np
import env, evals

JUDGE = ("swap.object", "swap.spatial", "swap.place", "swap.seq", "new.object", "new.spatial", "new.place", "new.seq", "combo.object", "wording.object", "more.object", "recover.object")


def rows(path=env.RESULTS):
    if not os.path.exists(path): return []
    return [r for r in map(json.loads, open(path)) if r["v"] == env.TEST_VERSION and r["tag"] == ""]


def pick_lr(all_rows, spec, regime, tier, world=0):
    """The lr with the best mean rank over JUDGE on tune rows, using the same search for every arm. None if the arm has no tune rows."""
    t = [r for r in all_rows if (r["spec"], r["regime"], r["tier"], r["world"], r["role"]) == (spec, regime, tier, world, "tune")]
    if not t: return None
    keys = [k for k in JUDGE if k in t[0]["scores"]] or list(t[0]["scores"])
    rank = {r["lr"]: 0. for r in t}
    for k in keys:
        for i, r in enumerate(sorted(t, key=lambda r: -r["scores"][k])): rank[r["lr"]] += i
    return min(rank, key=rank.get)


def chosen(all_rows, regime, tier, world=0, role="report"):
    """spec -> the report row at the lr tuning picked. An arm with no tune rows keeps its only report row."""
    out = {}
    for spec in sorted({r["spec"] for r in all_rows if (r["regime"], r["tier"], r["world"], r["role"]) == (regime, tier, world, role)}):
        c = [r for r in all_rows if (r["spec"], r["regime"], r["tier"], r["world"], r["role"]) == (spec, regime, tier, world, role)]; lr = pick_lr(all_rows, spec, regime, tier, world)
        h = [r for r in c if r["lr"] == lr] or (c if lr is None else [])
        if h: out[spec] = h[-1]
    return out


def items(row):
    return np.load(os.path.join(env.RUNS, row["key"] + ".npz"))


def paired(a, b, score):
    """Mean difference a - b on one score, and its z over paired items. Both arms saw the same items."""
    x, y = items(a)[score], items(b)[score]; d = x - y; se = d.std() / max(1, len(d) - 1) ** .5 if len(d) > 1 else 0
    return float(d.mean()), (float(d.mean() / se) if se > 0 else 0.)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="1m"); p.add_argument("--regime", default="free"); p.add_argument("--ref", default=None); p.add_argument("--world", type=int, default=0); a = p.parse_args()
    t = chosen(rows(), a.regime, a.tier, a.world)
    if not t: raise SystemExit(f"no report rows for {a.regime} at {a.tier}")
    specs = sorted(t); scores = sorted({k for r in t.values() for k in r["scores"]})
    print("score".ljust(18) + "".join(s[:15].rjust(16) for s in specs))
    for k in scores:
        line = k.ljust(18)
        for s in specs:
            v = t[s]["scores"].get(k); txt = "-" if v is None else f"{v:.3f}"
            if a.ref and v is not None and a.ref in t and s != a.ref and k in t[a.ref]["scores"]: txt += f" z{paired(t[s], t[a.ref], k)[1]:+.1f}"
            line += txt.rjust(16)
        print(line)
    print("lr".ljust(18) + "".join(f"{t[s]['lr']:g}".rjust(16) for s in specs)); print("weights".ljust(18) + "".join(str(t[s]["weights"]).rjust(16) for s in specs)); print("train s".ljust(18) + "".join(str(t[s]["train_s"]).rjust(16) for s in specs))
