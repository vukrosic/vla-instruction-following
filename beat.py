"""The win rule, committed before any new design runs. A new design has to pass all of it. A design that gets close is still `no win`, and what it lost goes in the report.

python3 beat.py newfusion+flow --regime free --tiers 1m 5m --worlds 0 1

1. Same size. Weights within 2% of the baselines at every tier. "X and Y have the same N weights, because fit() sets each to the tier's target with one shared frozen encoder."
2. Compute cap. Its seconds per training step, measured one run at a time on one thread by speed.py, are at most 1.5x the median baseline's.
3. Tuned the same. Every arm searched the same LRS on tune rows, and it uses the lr that picked.
4. Every baseline, best per score. On each score it must beat the best baseline for that score, so a design can't win by beating a weak one.
5. Wins. z >= 2 over paired items on at least 4 of the JUDGE scores that audit.py allows, on each of two worlds, at the largest tier.
6. No losses. On no allowed score and no nominal score is z <= -2 at any listed tier.
7. Holds at scale. At the smaller tier it wins the same scores in direction: mean difference above 0 on at least 3 of the 4.
8. Hidden items. The final check runs `--role final` once, after the design is fixed, and it must repeat rule 5 at z >= 2 on the scores that won.

Every design tried is counted in EXPERIMENTS.md, and the report gives that count.
"""
import argparse
import audit as A, compare as C, models as M, jobs

MIN_WINS, Z = 4, 2.


def baselines(t):
    return [s for s in t if s.split("+")[0] in M.BASELINES]


def verdict(all_rows, spec, regime, tier, world, allowed, role="report"):
    t = C.chosen(all_rows, regime, tier, world, role)
    if spec not in t: return dict(ok=False, why=f"no {role} row at {tier} world {world}")
    out = dict(wins=[], losses=[], ties=[]); base = baselines(t)
    if not base: return dict(ok=False, why=f"no baselines at {tier} world {world}")
    w = {t[s]["weights"] for s in base} | {t[spec]["weights"]}
    if max(w) > 1.02 * min(w): return dict(ok=False, why=f"sizes differ: {sorted(w)}")
    for k in allowed + [k for k in t[spec]["scores"] if k.startswith("nominal.")]:
        if k not in t[spec]["scores"]: continue
        best = max(base, key=lambda s: t[s]["scores"].get(k, -1)); d, z = C.paired(t[spec], t[best], k); name = f"{k} {d:+.3f} z{z:+.1f} vs {best}"
        out["wins" if z >= Z and k in allowed else "losses" if z <= -Z else "ties"].append(name)
    out["ok"] = len(out["wins"]) >= MIN_WINS and not out["losses"]; return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("spec"); p.add_argument("--regime", default="free"); p.add_argument("--tiers", nargs="+", default=["1m", "5m"]); p.add_argument("--worlds", nargs="+", type=int, default=[0, 1]); a = p.parse_args()
    rows = C.rows(); top = a.tiers[-1]; aud = A.audit(rows, a.regime, top, a.worlds[0], (a.worlds[1], top)); allowed = [k for k in C.JUDGE if k in aud and aud[k]["separates"] and aud[k]["moves"] is not False and aud[k]["ceiling"] is not False and aud[k]["agrees"] is not False]
    print("allowed scores:", allowed); ok = True
    for tier in a.tiers:
        for w in a.worlds:
            v = verdict(rows, a.spec, a.regime, tier, w, allowed); print(tier, "world", w, v)
            if tier == top: ok &= v["ok"]
            else: ok &= not v.get("losses", ["missing"])
    print("WIN" if ok else "no win", "(rules 2, 7 and 8 are read from speed.py, the table and the final run)")
