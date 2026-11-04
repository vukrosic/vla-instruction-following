"""Which scores can judge a design? Cut-offs are set here, before any table is read. A score that fails can't support a claim.

python3 audit.py --tier 1m
moves      Published modules score above the blind control by at least MOVE on a higher-is-better score.
separates  The spread across published modules exceeds SPREAD standard errors.
ceiling    The oracle control is above FLOOR, so failure is not the world's fault.
agrees     With another world or tier present, rank correlation across modules is at least AGREE.
"""
import argparse, os
import numpy as np
import compare as C, env, evals, models as M

MOVE, SPREAD, FLOOR, AGREE = .1, 2., .5, .5


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if np.std(ra) and np.std(rb) else 0.


def audit(all_rows, regime, tier, world=0, other=None):
    t = C.chosen(all_rows, regime, tier, world); pub = [s for s in t if s.split("+")[0] in M.FUSIONS]; res = {}
    if not pub: return res
    o = C.chosen(all_rows, regime, other[1], other[0]) if other else {}
    for k in sorted({k for s in pub for k in t[s]["scores"]}):
        v = np.array([t[s]["scores"][k] for s in pub]); sign = 1 if evals.higher(k) else -1; r = {}
        blind = t.get("blind+mse", {}).get("scores", {}).get(k); r["moves"] = None if blind is None else bool(sign * (v - blind).max() >= MOVE)
        orc = t.get("oracle+mse", {}).get("scores", {}).get(k); r["ceiling"] = None if orc is None or not evals.higher(k) else bool(orc >= FLOOR)
        x = [C.items(t[s])[k] for s in pub if os.path.exists(os.path.join(env.RUNS, t[s]["key"] + ".npz"))]
        se = np.mean([i.std() / max(1, len(i) - 1) ** .5 for i in x]) if x else 0; r["separates"] = bool(v.max() - v.min() > SPREAD * se) if se else False
        both = [s for s in pub if s in o and k in o[s]["scores"]]; r["agrees"] = None if len(both) < 3 else spearman([t[s]["scores"][k] for s in both], [o[s]["scores"][k] for s in both]) >= AGREE
        res[k] = r
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="1m"); p.add_argument("--regime", default="free"); p.add_argument("--world", type=int, default=0); p.add_argument("--other-world", type=int, default=None); a = p.parse_args()
    res = audit(C.rows(), a.regime, a.tier, a.world, (a.other_world, a.tier) if a.other_world is not None else None)
    if not res: raise SystemExit("no rows")
    judge = [k for k, r in res.items() if all(v is not False for v in r.values()) and r["separates"]]
    for k, r in res.items(): print(k.ljust(18), " ".join(f"{n}={'-' if v is None else 'yes' if v else 'NO'}" for n, v in r.items()))
    print(f"{len(judge)} of {len(res)} scores can judge:", " ".join(judge))
