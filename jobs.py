"""Job lists, run one after another. A finished setting is skipped, so a stopped list continues where it stopped.

python3 jobs.py kill --tier 1m --device cuda    The kill test in EXPERIMENTS.md.
python3 jobs.py baselines --tier 5m             Every published module, each tuned on the same LRS.
python3 jobs.py kill --list                     Prints the settings and runs nothing.
"""
import argparse, itertools
import env, models as M

LRS = (3e-4, 1e-3, 3e-3)
HEADS_MAIN = ("mse", "chunk", "flow")


def kill():
    """Controls and four modules at one lr each: enough to see whether the world can tell them apart."""
    out = [dict(spec=f"{f}+mse", regime=r) for r in ("coupled", "free", "random") for f in ("token", "film", "cross", "langattn", "blind", "shuffle", "oracle")]
    out += [dict(spec="token+mse", regime=r) for r in ("small", "dilute")] + [dict(spec=f"{f}+answer", regime="random") for f in ("token", "film", "cross", "langattn")]
    return out


def baselines():
    """Each module with each action head, on each lr, on tune rows, then report rows at every lr. compare.py uses only the lr tuning picked."""
    out = []
    for f, h, r, lr, role in itertools.product(M.BASELINES, HEADS_MAIN, ("free", "random"), LRS, ("tune", "report")): out.append(dict(spec=f"{f}+{h}", regime=r, lr=lr, role=role))
    for c, r, lr, role in itertools.product(M.CONTROLS, ("coupled", "free", "random"), LRS, ("tune", "report")): out.append(dict(spec=f"{c}+mse", regime=r, lr=lr, role=role))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("name", choices=("kill", "baselines")); p.add_argument("--tier", default="1m"); p.add_argument("--world", type=int, default=0); p.add_argument("--device", default="cpu"); p.add_argument("--list", action="store_true"); a = p.parse_args()
    js = dict(kill=kill, baselines=baselines)[a.name]()
    print(len(js), "settings")
    for j in js:
        if a.list: print(j); continue
        x = env.parser().parse_args([]); x.tier, x.world, x.device, x.save_every = a.tier, a.world, a.device, 1000
        for k, v in j.items(): setattr(x, k, v)
        r = env.run(x)
        if r: print(j, r["train_s"], "s", {k: r["scores"][k] for k in list(r["scores"])[:4]}, flush=True)
