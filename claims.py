"""Does this test reproduce what papers report? Each claim is one function of the result rows. A mismatch needs a reason from a targeted run, not a change to the claim.

python3 claims.py --tier 1m    Prints pass, FAIL or `no rows` per claim. These are small ports in a small world, so only the direction and rough size can match.
"""
import argparse
import compare as C


def score(t, spec, k):
    return t[spec]["scores"].get(k) if spec in t else None


def claims(all_rows, tier, world=0, spec="token+flow"):
    c = C.chosen(all_rows, "coupled", tier, world); f = C.chosen(all_rows, "free", tier, world); s = C.chosen(all_rows, "small", tier, world); d = C.chosen(all_rows, "dilute", tier, world)
    def gap(t, a, b, m): 
        x, y = score(t, a[0], a[1]), score(t, b[0], b[1]); return None if x is None or y is None else x - y >= m
    out = [
        ("Trained tasks high, a different object in the same scene low. LangGap 93.8% vs 29.3%; 2610.06235 current-target 0-11.5%.", gap(c, (spec, "nominal.object"), (spec, "swap.object"), .4)),
        ("Asked for another object, the policy goes to the original. 2610.06235: 86.5-98.3% to the original.", None if score(c, spec, "swap.orig") is None else score(c, spec, "swap.orig") >= .5),
        ("A random sentence still reaches the original less often than the right sentence. 2610.06235: 98.0% to 11.7% on pi0.5.", gap(c, (spec, "nominal.object"), (spec, "nonsense.orig"), .1)),
        ("The named target can be decoded from context while control fails. 2610.06235.", None if score(c, spec, "probe.target") is None or score(c, spec, "swap.object") is None else score(c, spec, "probe.target") - score(c, spec, "swap.object") >= .3),
        ("Offline error can look fine while closed-loop swap fails. 2609.23048: 0/72 vs 40/72.", None if score(c, spec, "off.mse") is None or score(c, spec, "swap.object") is None else score(c, spec, "off.mse") < .15 and score(c, spec, "swap.object") < .3),
        ("Several tasks per scene help on the same demonstration count. LangGap: 6-task training 0% to 28.0%.", None if score(f, spec, "swap.object") is None or score(c, spec, "swap.object") is None else score(f, spec, "swap.object") - score(c, spec, "swap.object") >= .1),
        ("Adding single-task data elsewhere dilutes it. LangGap: 28.0% to 4.0%.", gap(s, (spec, "swap.object"), (spec, "swap.object"), 0) if not d else (None if score(s, spec, "swap.object") is None or score(d, spec, "swap.object") is None else score(s, spec, "swap.object") - score(d, spec, "swap.object") >= .05)),
        ("A policy with no sentence gets at most about 1/k. LangGap's bound; k=5 here.", None if score(f, "blind+mse", "new.object") is None else score(f, "blind+mse", "new.object") <= .3),
        ("Cross-attention is close to token insertion, slightly lower from scratch. CASA: about 1.5 points on VLM benchmarks, not on control.", None if score(f, "cross+mse", "new.object") is None or score(f, "token+mse", "new.object") is None else abs(score(f, "cross+mse", "new.object") - score(f, "token+mse", "new.object")) <= .15),
    ]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--tier", default="1m"); p.add_argument("--world", type=int, default=0); p.add_argument("--spec", default="token+flow"); a = p.parse_args()
    for text, ok in claims(C.rows(), a.tier, a.world, a.spec): print(("no rows" if ok is None else "pass" if ok else "FAIL").ljust(9), text)
