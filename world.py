"""The tabletop world: scenes, instructions, a scripted expert, physics and rendering, all batched in numpy with no simulator.

A scene holds 5 objects (colour + shape, positions on a unit table), a gripper, and a background tint plus two dim rectangles that
identify the layout. A policy sees a 36x36 image, a tokenized sentence and the gripper state, and outputs (dx, dy, grip).

The design copies the property LangGap (2603.00592) and "Encoded but Not in Control" (2610.06235) rely on: several valid tasks share one
scene, so a policy that ignores language can reach at most about 1/k of them. Each layout has one original task, which the `coupled`
regime trains on alone, as LIBERO has one task per layout. It also has held tasks that ask for a different object, which no regime trains.
"""
import numpy as np

IMG = 36
K, K_MAX = 5, 7
T = 30
SPEED, NOISE, R, GRASP = 0.10, 0.003, 0.08, 0.045
COLORS = ["red", "green", "blue", "yellow", "purple", "cyan"]
RGB = np.array([[.90, .12, .12], [.10, .72, .18], [.15, .25, .95], [.95, .85, .05], [.60, .15, .80], [.05, .80, .85]], dtype=np.float32)
SHAPES = ["circle", "square", "triangle", "cross"]
DIRS = ["leftmost", "rightmost", "topmost", "lowest"]
FAMILIES = ("object", "spatial", "place", "seq")
PAIRS = [(c, s) for c in range(len(COLORS)) for s in range(len(SHAPES))]
TRAIN_T = dict(
    object=["go to the {a}", "move to the {a}", "reach the {a}"],
    spatial=["go to the {d} object", "move to the {d} object", "reach the {d} object"],
    place=["put the {a} on the {b}", "move the {a} onto the {b}"],
    seq=["touch the {a} then the {b}", "touch the {a} and after that the {b}"],
)
HELD_T = ["to the {a} go", "the {a} reach"]  # known words in orders no training row uses
QUESTION_T = dict(
    attr="what shape is the {c} object", count="how many {c} objects",
    qspatial="what color is the {d} object", near="what color is the object nearest the {a}",
)
WORDS = sorted(set(" ".join(sum(TRAIN_T.values(), []) + HELD_T + list(QUESTION_T.values())).replace("{a}", "").replace("{b}", "").replace("{d}", "").replace("{c}", "").split())
               | set(COLORS) | set(SHAPES) | set(DIRS) | {"that", "is"})
VOCAB = {w: i + 1 for i, w in enumerate(WORDS)}
ANSWERS = COLORS + SHAPES + ["0", "1", "2", "3", "4", "5", "6", "7"]
L_MAX = 16


def _stamp(shape):
    g = np.arange(-4, 5); y, x = np.meshgrid(g, g, indexing="ij")
    if shape == "circle": m = (x * x + y * y <= 17) & (x * x + y * y >= 5)  # a ring, because a solid disc this small looks like the square
    elif shape == "square": m = (abs(x) <= 3) & (abs(y) <= 3)
    elif shape == "triangle": m = (y >= -3) & (y <= 3) & (abs(x) <= (y + 3) * 2 // 3 + 1)
    else: m = (abs(x) <= 1) & (abs(y) <= 4) | (abs(y) <= 1) & (abs(x) <= 4)
    return y[m], x[m]


STAMPS = [_stamp(s) for s in SHAPES]


def phrase(pair, rng, held=False):
    c, s = COLORS[pair[0]], SHAPES[pair[1]]
    return f"{c} {s}" if rng.random() < .5 else f"{s} that is {c}"


def encode(text):
    ids = [VOCAB[w] for w in text.split()]
    assert len(ids) <= L_MAX, text
    return np.array(ids + [0] * (L_MAX - len(ids)), dtype=np.int64)


def held_pairs(world):
    """Colour-shape pairs no instruction names in training, per world."""
    r = np.random.default_rng(1000 + world); return {PAIRS[i] for i in r.choice(len(PAIRS), 5, replace=False)}


def make_layout(rng, k=K, avoid=None, must=None):
    """Objects with unique colour-shape pairs, spaced apart, plus the background that identifies the layout."""
    avoid = avoid or set()
    while True:
        p = rng.uniform(.12, .88, (k, 2))
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(k)
        if d.min() > .24: break
    ids = rng.choice(len(PAIRS), k, replace=False); pairs = [PAIRS[i] for i in ids]
    if must is not None and must not in pairs: pairs[0] = must
    deco = rng.uniform(0, 1, (2, 4)); tint = rng.uniform(.35, .65, 3)
    return dict(pos=p.astype(np.float32), pairs=pairs, k=k, tint=tint.astype(np.float32), deco=deco.astype(np.float32))


def target_of_dir(layout, d):
    p = layout["pos"]
    return int({0: p[:, 0].argmin(), 1: p[:, 0].argmax(), 2: p[:, 1].argmin(), 3: p[:, 1].argmax()}[d])


def task(layout, fam, a=None, b=None, d=None, rng=None, tpl=None):
    """A task is a family, the object the policy must act on first (a), a second object (b) and its sentence."""
    rng = rng or np.random.default_rng(0)
    if fam == "spatial":
        a = target_of_dir(layout, d); text = (tpl or rng.choice(TRAIN_T[fam])).format(d=DIRS[d])
    elif fam == "object":
        text = (tpl or rng.choice(TRAIN_T[fam])).format(a=phrase(layout["pairs"][a], rng))
    else:
        text = (tpl or rng.choice(TRAIN_T[fam])).format(a=phrase(layout["pairs"][a], rng), b=phrase(layout["pairs"][b], rng))
    return dict(fam=fam, a=a, b=-1 if b is None else b, d=d, tpl=None, text=text)


def layout_table(world, n_layouts):
    """Per layout: one original task; 2 free tasks per family; report, tune and final held tasks per family, each on a target other than the original's."""
    rng = np.random.default_rng(2000 + world); out = []; banned = held_pairs(world)
    for i in range(n_layouts):
        while True:
            lay = make_layout(rng)
            if any(p in banned for p in lay["pairs"]): continue
            dirt = [target_of_dir(lay, d) for d in range(4)]
            fam0 = FAMILIES[i % 4]; order = rng.permutation(K).tolist(); a0 = order[0]
            if fam0 == "spatial":
                d0 = int(rng.integers(4)); a0 = dirt[d0]
                others = [d for d in range(4) if dirt[d] != a0]
                if len(others) < 3: continue
            else:
                d0 = None; others = [d for d in range(4) if dirt[d] != a0]
                if len(others) < 3: continue
            break
        rest = [x for x in range(K) if x != a0]; rng.shuffle(rest); rest = [int(x) for x in rest]
        t = dict(layout=lay)
        if fam0 == "spatial": t["orig"] = task(lay, "spatial", d=d0, rng=rng)
        elif fam0 == "object": t["orig"] = task(lay, "object", a=a0, rng=rng)
        else: t["orig"] = task(lay, fam0, a=a0, b=rest[0], rng=rng)
        # Held roles use targets 0..2 of `rest`; the free regime trains on rest[3] and on the original target.
        d_order = rng.permutation(others).tolist()
        for j, role in enumerate(("report", "tune", "final")):
            t[role] = dict(
                object=task(lay, "object", a=rest[j], rng=rng),
                spatial=task(lay, "spatial", d=int(d_order[j]), rng=rng),
                place=task(lay, "place", a=rest[j], b=rest[(j + 1) % 4], rng=rng),
                seq=task(lay, "seq", a=rest[j], b=rest[(j + 2) % 4], rng=rng),
            )
        held_d = {int(x) for x in d_order[:3]}
        free_d = [d for d in range(4) if d not in held_d]
        t["free"] = [task(lay, "object", a=a0, rng=rng), task(lay, "object", a=rest[3], rng=rng),
                     task(lay, "place", a=a0, b=rest[3], rng=rng), task(lay, "place", a=rest[3], b=a0, rng=rng),
                     task(lay, "seq", a=a0, b=rest[3], rng=rng), task(lay, "seq", a=rest[3], b=a0, rng=rng)]
        t["free"] += [task(lay, "spatial", d=d, rng=rng) for d in free_d]
        out.append(t)
    return out


def fresh_item(rng, fam, world, k=K, combo=False):
    """A new layout with a task whose named pairs were seen in training, or with a target from the held pairs when combo is set."""
    banned = held_pairs(world)
    while True:
        must = sorted(banned)[int(rng.integers(len(banned)))] if combo else None
        lay = make_layout(rng, k, must=must)
        if combo:
            a = lay["pairs"].index(must); rest = [i for i in range(k) if i != a]; b = int(rng.choice(rest))
            return lay, task(lay, "object", a=a, rng=rng)
        order = [int(x) for x in rng.permutation(k)]
        named = [o for o in order if lay["pairs"][o] not in banned]
        if len(named) < 2: continue
        if fam == "spatial": return lay, task(lay, "spatial", d=int(rng.integers(4)), rng=rng)
        if fam == "object": return lay, task(lay, "object", a=named[0], rng=rng)
        return lay, task(lay, fam, a=named[0], b=named[1], rng=rng)


class Batch:
    """N episodes running in parallel."""

    def __init__(s, layouts, tasks, rng, texts=None, jitter=.02):
        n = len(layouts); s.n, s.rng = n, rng
        s.k = np.array([l["k"] for l in layouts])
        s.pos = np.zeros((n, K_MAX, 2), np.float32); s.col = np.zeros((n, K_MAX), np.int64); s.shp = np.zeros((n, K_MAX), np.int64)
        s.tint = np.stack([l["tint"] for l in layouts]); s.deco = np.stack([l["deco"] for l in layouts])
        for i, l in enumerate(layouts):
            s.pos[i, :l["k"]] = np.clip(l["pos"] + rng.uniform(-jitter, jitter, (l["k"], 2)), .05, .95)
            s.col[i, :l["k"]] = [p[0] for p in l["pairs"]]; s.shp[i, :l["k"]] = [p[1] for p in l["pairs"]]
        s.agent = rng.uniform(.1, .9, (n, 2)).astype(np.float32)
        s.grip = np.zeros(n, bool); s.hold = np.full(n, -1); s.lit = np.zeros((n, K_MAX), bool); s.order = np.full((n, K_MAX), -1); s.nlit = np.zeros(n, np.int64)
        s.fam = np.array([FAMILIES.index(t["fam"]) for t in tasks]); s.a = np.array([t["a"] for t in tasks]); s.b = np.array([t["b"] for t in tasks])
        s.tok = np.stack([encode(x) for x in (texts if texts is not None else [t["text"] for t in tasks])])
        s.first_hold = np.full(n, -1); s.first_lit = np.full(n, -1); s.seq_done = np.zeros(n, bool)

    def state(s):
        return np.concatenate([s.agent, s.grip[:, None] * 2. - 1, (s.hold >= 0)[:, None] * 2. - 1], 1).astype(np.float32)

    def goal(s):
        """True xy of the object the task needs next, read only by the oracle control."""
        idx = np.arange(s.n); g = s.pos[idx, s.a].copy()
        carrying = (s.fam == 2) & (s.hold == s.a); g[carrying] = s.pos[idx, s.b][carrying]
        go_b = (s.fam == 3) & s.lit[idx, s.a]; g[go_b] = s.pos[idx, s.b][go_b]
        return g

    def expert(s):
        idx = np.arange(s.n); act = np.zeros((s.n, 3), np.float32); act[:, 2] = -1
        goal = s.pos[idx, s.a].copy()
        place, seq = s.fam == 2, s.fam == 3
        close = np.zeros(s.n, bool); open_ = np.zeros(s.n, bool)
        # place: fetch a, then carry to b and let go
        carrying = place & (s.hold == s.a)
        goal[carrying] = s.pos[idx, s.b][carrying]
        d = np.linalg.norm(goal - s.agent, axis=1)
        placed = place & (s.hold < 0) & (s.first_hold == s.a) & (np.linalg.norm(s.pos[idx, s.a] - s.pos[idx, np.maximum(s.b, 0)], axis=1) < R)
        goal[placed] = s.agent[placed]
        close |= place & ~carrying & ~placed & (d < GRASP) & (s.hold < 0)
        open_ |= place & (s.hold >= 0) & (s.hold != s.a)
        hold_on = carrying & (d >= GRASP)
        # seq: touch a, then touch b; touching an object lights it
        a_lit = s.lit[idx, s.a]
        go_b = seq & a_lit
        goal[go_b] = s.pos[idx, s.b][go_b]
        d = np.linalg.norm(goal - s.agent, axis=1)
        v = (goal - s.agent) / SPEED; m = np.maximum(np.abs(v).max(1, keepdims=True), 1)
        act[:, :2] = v / m
        act[close | hold_on, 2] = 1
        act[open_, 2] = -1
        return act

    def step(s, act):
        idx = np.arange(s.n)
        v = np.clip(act[:, :2], -1, 1) * SPEED + s.rng.normal(0, NOISE, (s.n, 2))
        s.agent = np.clip(s.agent + v, 0, 1).astype(np.float32)
        now = act[:, 2] > 0; rising = now & ~s.grip; s.grip = now
        s.hold[~now] = -1
        for i in np.nonzero(rising)[0]:
            ki = s.k[i]; d = np.linalg.norm(s.pos[i, :ki] - s.agent[i], axis=1); j = int(d.argmin())
            if d[j] < R:
                s.hold[i] = j
                if s.first_hold[i] < 0: s.first_hold[i] = j
        near = (np.linalg.norm(s.pos - s.agent[:, None], axis=-1) < R) & (np.arange(K_MAX)[None] < s.k[:, None]) & ~s.lit
        for i, j in zip(*np.nonzero(near)):
            s.lit[i, j] = True; s.order[i, s.nlit[i]] = j; s.nlit[i] += 1
            if s.first_lit[i] < 0: s.first_lit[i] = j
        h = s.hold >= 0
        s.pos[idx[h], s.hold[h]] = s.agent[h]
        seq = (s.fam == 3) & s.lit[idx, s.a] & (np.linalg.norm(s.pos[idx, np.maximum(s.b, 0)] - s.agent, axis=1) < R)
        s.seq_done |= seq

    def success(s):
        idx = np.arange(s.n); to_a = np.linalg.norm(s.pos[idx, s.a] - s.agent, axis=1) < R
        reach = to_a & (s.fam <= 1)
        place = (s.fam == 2) & (np.linalg.norm(s.pos[idx, s.a] - s.pos[idx, np.maximum(s.b, 0)], axis=1) < R * 1.5) & (s.hold != s.a) & (s.first_hold == s.a)
        seq = (s.fam == 3) & s.seq_done
        return reach | place | seq

    def commit(s):
        """The object the policy acted on: the object it ended at for reach tasks, the first held for place, the first touched for seq; -1 if none."""
        out = np.full(s.n, -1)
        for i in range(s.n):
            if s.fam[i] <= 1:
                d = np.linalg.norm(s.pos[i, :s.k[i]] - s.agent[i], axis=1); j = int(d.argmin())
                if d[j] < R: out[i] = j
            elif s.fam[i] == 2: out[i] = s.first_hold[i]
            else: out[i] = s.first_lit[i]
        return out

    def render(s):
        n = s.n; im = np.empty((n, IMG, IMG, 3), np.float32); im[:] = s.tint[:, None, None]
        for r in range(2):
            x0 = (s.deco[:, r, 0] * (IMG - 8)).astype(int); y0 = (s.deco[:, r, 1] * (IMG - 8)).astype(int); w = 3 + (s.deco[:, r, 2] * 5).astype(int); h = 3 + (s.deco[:, r, 3] * 5).astype(int)
            for i in range(n): im[i, y0[i]:y0[i] + h[i], x0[i]:x0[i] + w[i]] *= .82
        px = (s.pos * (IMG - 1)).round().astype(int)
        mask = np.arange(K_MAX)[None] < s.k[:, None]
        for sh, (dy, dx) in enumerate(STAMPS):
            I, J = np.nonzero((s.shp == sh) & mask)
            if not len(I): continue
            Y = px[I, J, 1, None] + dy; X = px[I, J, 0, None] + dx
            ok = (Y >= 0) & (Y < IMG) & (X >= 0) & (X < IMG)
            II = np.broadcast_to(I[:, None], Y.shape)[ok]
            im[II, Y[ok], X[ok]] = np.broadcast_to(RGB[s.col[I, J]][:, None], Y.shape + (3,))[ok]
        I, J = np.nonzero(s.lit)
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)): im[I, np.clip(px[I, J, 1] + dy, 0, IMG - 1), np.clip(px[I, J, 0] + dx, 0, IMG - 1)] = 1.
        a = (s.agent * (IMG - 1)).round().astype(int)
        for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2), (0, 0)):
            y, x = np.clip(a[:, 1] + dy, 0, IMG - 1), np.clip(a[:, 0] + dx, 0, IMG - 1); im[np.arange(n), y, x] = 0.
        g = np.nonzero(s.grip)[0]
        im[g, a[g, 1], a[g, 0]] = 1.
        return (im * 255).astype(np.uint8)


def nonsense(rng, n):
    w = list(VOCAB); return [" ".join(rng.choice(w, 6)) for _ in range(n)]


def question(lay, kind, rng):
    p = lay["pos"]; pairs = lay["pairs"]; k = lay["k"]
    if kind == "attr":
        cs = [c for c, _ in pairs]; uniq = [i for i in range(k) if cs.count(cs[i]) == 1]
        if not uniq: return None
        i = int(rng.choice(uniq)); return QUESTION_T[kind].format(c=COLORS[pairs[i][0]]), SHAPES[pairs[i][1]]
    if kind == "count":
        c = int(rng.integers(len(COLORS))); return QUESTION_T[kind].format(c=COLORS[c]), str(sum(x == c for x, _ in pairs))
    if kind == "qspatial":
        d = int(rng.integers(4)); return QUESTION_T[kind].format(d=DIRS[d]), COLORS[pairs[target_of_dir(lay, d)][0]]
    i = int(rng.integers(k)); dist = np.linalg.norm(p - p[i], axis=1); dist[i] = 9; j = int(dist.argmin())
    return QUESTION_T[kind].format(a=phrase(pairs[i], rng)), COLORS[pairs[j][0]]
