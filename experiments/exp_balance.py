"""Experiments 3-4 (v2): the gimbal fly, one brain in 100 bodies, and evolving a body for a fixed brain.

Body: flybody (MuJoCo, 67 parts, adhesive claws) on a platform rocking side to side (roll), simulated exactly by
rotating gravity in the platform frame. Goal: keep the thorax level with true vertical.

Brain: the real MaleCNS ventral nerve cord + descending + ascending neurons + all proprioceptors (24,073 LIF neurons).
Senses (every 10 ms): leg load -> campaniform sensilla, femur-tibia angle -> chordotonal / hair-plate afferents,
body roll rate -> haltere afferents (halteres are the fly's gyroscopes; the left/right populations take the two signs).

Control loop:  senses -> VNC (frozen wiring) -> linear readout of tilt from ~20k NON-sensory VNC neurons
               (interneurons, motor neurons, ascending/descending)  ->  tilt estimate x leveling pattern -> 42 leg joints.
What is learned: (A) the leveling pattern (42 numbers, ES on physics with the true tilt), (B) the tilt readout
(PCA -> ridge, DAgger). The connectome is never edited.

    python exp_balance.py teacher     # (A)
    python exp_balance.py readout     # (B)
    python exp_balance.py eval        # passive vs teacher (true tilt) vs brain-driven vs blind-brain controls
    python exp_balance.py bodies      # one brain, 100 random bodies
    python exp_balance.py evolve      # GA over body genes, brain + readout fixed
    python exp_balance.py video
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylab.core import OUT, ES, conn, log
from flylab.physfly import VNCBrain, JOINTS, LEGS, DEFAULT_GENES, GENE_RANGE
from flylab.pool import FlyPool, roll_deg

OUTD = os.path.join(OUT, "balance")
NJ = len(JOINTS)
TRAIN_PROFS = [dict(amp=25, freq=0.35, phase=0.0), dict(amp=25, freq=0.5, phase=np.pi), dict(amp=18, freq=0.7, phase=1.0),
               dict(amp=28, freq=0.25, phase=2.0)]
TEST_PROF = dict(amp=25, freq=0.4, phase=0.0)


def roll_rate(t, prof):
    a, f, ph, ramp = prof["amp"], prof["freq"], prof["phase"], prof.get("ramp", 0.5)
    s = min(1.0, t / ramp)
    return np.deg2rad(a * (s * 2 * np.pi * f * np.cos(2 * np.pi * f * t + ph) + (t < ramp) / ramp * np.sin(2 * np.pi * f * t + ph)))


# ------------------------------------------------------------------------------------------- the brain's senses
class Senses:
    def __init__(self, vb: VNCBrain):
        self.vb = vb
        c, n = vb.fb.c, vb.fb.c.neurons
        from flybrain.senses import laterality
        sub = n.subclass.fillna("").to_numpy().astype(str)
        pro = (n["class"] == "mechanosensory_proprioceptive").to_numpy()
        hal = np.flatnonzero(pro & (sub == "haltere"))
        inst = n.instance.fillna("").to_numpy().astype(str)[hal]
        side = np.where(np.char.endswith(inst, "_L"), 1, np.where(np.char.endswith(inst, "_R"), -1, 0))
        ref = c.reference; rs = ref.neurons.somaSide.to_numpy()
        lat = laterality(ref, ref.index_of(n.bodyId.to_numpy()[hal]), np.flatnonzero(rs == "L"), np.flatnonzero(rs == "R"))
        side = np.where(side != 0, side, np.where(lat > 0.1, 1, np.where(lat < -0.1, -1, 0)))
        self.hal_L, self.hal_R = hal[side > 0], hal[side < 0]
        sens = np.zeros(c.n, bool); sens[vb.sens_idx] = True; sens[hal] = True
        sup = n.superclass.fillna("").to_numpy().astype(str)
        sens |= np.char.find(sup, "sensory") >= 0
        self.readout_idx = np.flatnonzero(~sens)
        self.readout_t = torch.as_tensor(self.readout_idx, device=vb.dev)

    def describe(self):
        return (f"halteres: {len(self.hal_L)} L + {len(self.hal_R)} R afferents; antennal JO: {0 if self.vb.fb.wind_sense is None else len(self.vb.fb.wind_sense.joC) + len(self.vb.fb.wind_sense.joE)} cells; readout listens to {len(self.readout_idx):,} "
                f"non-sensory neurons")

    def feed(self, loads, flex, omega, blind=False, tone=10.0, grav=None):
        """grav: signed tilt of the body from true vertical (rad) -> antennal (JO-C/E) deflection, opposite per side."""
        vb, B = self.vb, len(loads)
        if grav is None:
            grav = np.zeros(B)
        if blind:
            loads, flex, omega, grav = np.ones_like(loads), np.zeros_like(flex), np.zeros_like(omega), np.zeros(B)
        if vb.fb.wind_sense is not None:
            d = np.clip(2.0 * np.sin(grav), -1, 1)
            vb.fb.wind(d, -d)
        rates = []
        for li, (nl, nf, ne) in enumerate(vb.sens_layout):
            l = np.clip(20 + 120 * (loads[:, li] - 1), 0, 250)
            f = np.clip(10 + 150 * np.maximum(flex[:, li], 0), 0, 250)
            e = np.clip(10 + 150 * np.maximum(-flex[:, li], 0), 0, 250)
            rates += [np.repeat(l[:, None], nl, 1), np.repeat(f[:, None], nf, 1), np.repeat(e[:, None], ne, 1)]
        vb.fb._input("legsense", vb.sens_idx, np.concatenate(rates, 1).astype(np.float32))
        hl = np.clip(30 + 90 * omega, 0, 250); hr = np.clip(30 - 90 * omega, 0, 250)
        vb.fb._input("hal_L", self.hal_L, np.repeat(hl[:, None], len(self.hal_L), 1).astype(np.float32))
        vb.fb._input("hal_R", self.hal_R, np.repeat(hr[:, None], len(self.hal_R), 1).astype(np.float32))
        vb.fb._input("dn_tone", vb.dn, np.full((B, len(vb.dn)), tone, np.float32))

    def rates(self):
        return self.vb.fb.brain.rate[:, self.readout_t].float().cpu().numpy()


# ------------------------------------------------------------------------------------------------ episodes
def run(pool, prof, seconds, V=None, tilt_fn=None, brain=None, record=False, blind=False):
    """V: (6, NJ) leveling pattern (per radian of tilt). tilt_fn(state) -> tilt estimate (rad) per body, or None for
    the true platform roll (teacher). brain: (vb, senses) to keep the VNC running (needed for brain-driven control)."""
    loads, flex, tilt, fallen, om, up = pool.reset()
    n = pool.n
    flat = dict(amp=0.0, freq=0.0, phase=0.0)
    if brain is not None:
        vb, se = brain
        vb.fb.reset()
        for _ in range(30):
            se.feed(loads, flex, om, blind, grav=-np.arctan2(-up[:, 1], up[:, 2])); vb.step(10.0)
            loads, flex, tilt, fallen, om, up = pool.step(None, flat)
    tilts, X, Y, est_hist = [], [], [], []
    est = np.zeros(n)
    for k in range(int(seconds / 0.01)):
        t = k * 0.01
        true = np.deg2rad(roll_deg(t, prof))
        omega = roll_rate(t, prof) + om
        grav = true - np.arctan2(-up[:, 1], up[:, 2])          # body tilt from true vertical (signed)
        if brain is not None:
            se.feed(loads, flex, omega, blind, grav=grav); vb.step(10.0)
            if record or tilt_fn is not None:
                x = se.rates()
                if record:
                    X.append(x.astype(np.float16)); Y.append(np.full(n, true))
                if tilt_fn is not None:
                    est = tilt_fn(x, est)
        u = np.full(n, true) if tilt_fn is None else est
        est_hist.append(u.copy())
        off = None if V is None else np.clip(u[:, None, None] * V[None], -1.0, 1.0)
        loads, flex, tilt, fallen, om, up = pool.step(off, prof, t)
        tilts.append(tilt)
    T = np.array(tilts)[50:]
    out = dict(tilt=T.mean(0) + 30 * fallen, trace=np.array(tilts), est=np.array(est_hist))
    if record:
        out["X"], out["Y"] = np.concatenate(X), np.concatenate(Y)
    return out


def teacher(gens=25, pop=32):
    """(A) ES over the 42-number leveling pattern, physics only, true tilt."""
    os.makedirs(OUTD, exist_ok=True)
    pool = FlyPool([DEFAULT_GENES] * pop)
    passive = run(pool, TEST_PROF, 2.5)["tilt"].mean()
    log(f"passive tilt {passive:.2f} deg")
    es = ES(6 * NJ, pop, sigma=0.4, seed=1)
    t0 = time.time()
    for g in range(gens):
        cand = es.ask()
        prof = TRAIN_PROFS[g % len(TRAIN_PROFS)]
        tilt = np.zeros(pop)
        loads, flex, tl, fallen, om, up = pool.reset()
        tr = []
        for k in range(250):
            t = k * 0.01
            true = np.deg2rad(roll_deg(t, prof))
            off = np.clip(true * cand.reshape(pop, 6, NJ), -1.0, 1.0)
            loads, flex, tl, fallen, om, up = pool.step(off, prof)
            tr.append(tl)
        fit = -(np.array(tr)[50:].mean(0) + 30 * fallen)
        es.tell(cand, fit)
        log(f"teacher gen {g:2d}: best {-fit.max():5.2f}  centre {-fit[0]:5.2f}  mean {-fit.mean():5.2f} deg ({time.time() - t0:.0f}s)")
        np.save(os.path.join(OUTD, "V.npy"), es.best.reshape(6, NJ))
        np.save(os.path.join(OUTD, "V_mean.npy"), es.mean.reshape(6, NJ))
    pool.close()


def pick_V():
    """Best of the ES best / ES mean on the test profile."""
    cands = [np.load(os.path.join(OUTD, f)) for f in ("V.npy", "V_mean.npy")]
    pool = FlyPool([DEFAULT_GENES] * 2, workers=2)
    scores = []
    for V in cands:
        loads, flex, tl, fallen, om, up = pool.reset(); tr = []
        for k in range(400):
            true = np.deg2rad(roll_deg(k * 0.01, TEST_PROF))
            loads, flex, tl, fallen, om, up = pool.step(np.clip(true * V[None].repeat(2, 0), -1, 1), TEST_PROF); tr.append(tl)
        scores.append(float(np.array(tr)[50:].mean()))
    pool.close()
    V = cands[int(np.argmin(scores))]
    np.save(os.path.join(OUTD, "V_final.npy"), V)
    log("teacher candidates on test profile:", scores)
    return V


class TiltReadout:
    def __init__(self, d):
        self.mu, self.sd, self.P, self.w, self.b = d["mu"], d["sd"], d["P"], d["w"], float(d["b"])
        self.alpha = float(d.get("alpha", 0.3))

    @classmethod
    def fit(cls, X, y, k=200, lams=(1, 10, 100, 1e3, 1e4)):
        mu, sd = X.mean(0), X.std(0) + 0.05
        Z = torch.from_numpy((X - mu) / sd)
        _, _, P = torch.pca_lowrank(Z, q=min(k, len(Z) - 1), center=False, niter=4)
        A = (Z @ P).numpy(); n = len(A)
        folds = np.array_split(np.arange(n), 5); best = None
        for lam in lams:
            r2 = []
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f); ym = y[tr].mean()
                w = np.linalg.solve(A[tr].T @ A[tr] + lam * np.eye(A.shape[1]), A[tr].T @ (y[tr] - ym))
                p = A[f] @ w + ym
                r2.append(1 - ((y[f] - p) ** 2).sum() / max(((y[f] - y[f].mean()) ** 2).sum(), 1e-9))
            if best is None or np.mean(r2) > best[0]:
                best = (float(np.mean(r2)), lam)
        ym = y.mean()
        w = np.linalg.solve(A.T @ A + best[1] * np.eye(A.shape[1]), A.T @ (y - ym))
        log(f"  tilt readout: {n} frames x {X.shape[1]} neurons -> {A.shape[1]} PCs, ridge {best[1]:g}, CV R2 {best[0]:.3f}")
        return cls(dict(mu=mu, sd=sd, P=P.numpy(), w=w.astype(np.float32), b=ym)), best[0]

    def __call__(self, x, prev):
        y = ((x - self.mu) / self.sd) @ self.P @ self.w + self.b
        return (1 - self.alpha) * prev + self.alpha * y          # light smoothing of the decoded tilt (part of the readout)

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, P=self.P, w=self.w, b=self.b, alpha=self.alpha)


def readout(rounds=4, B=16):
    """(B) DAgger: record VNC activity while the body is levelled by the teacher / by the current readout."""
    V = pick_V()
    vb = VNCBrain(conn(), batch=B); se = Senses(vb)
    log(vb.summary()); log(se.describe())
    pool = FlyPool([DEFAULT_GENES] * B)
    Xs, Ys, ro, rep = [], [], None, []
    t0 = time.time()
    for r in range(rounds):
        for i, prof in enumerate(TRAIN_PROFS[:2] if r else TRAIN_PROFS[:2]):
            prof = dict(prof, phase=prof["phase"] + r)
            fn = None if (ro is None or (r == 1 and i == 0)) else ro
            out = run(pool, prof, 3.0, V=V, tilt_fn=fn, brain=(vb, se), record=True)
            Xs.append(out["X"]); Ys.append(out["Y"])
            log(f"  round {r} profile {i}: {'teacher' if fn is None else 'brain'} drives, tilt {out['tilt'].mean():.2f} deg")
        ro, r2 = TiltReadout.fit(np.concatenate(Xs).astype(np.float32), np.concatenate(Ys))
        ro.save(os.path.join(OUTD, "tilt_readout.npz"))
        rep.append(dict(round=r, cv_r2=r2, wall=time.time() - t0))
        json.dump(rep, open(os.path.join(OUTD, "readout.json"), "w"), indent=1)
    pool.close()


def load_controller():
    return np.load(os.path.join(OUTD, "V_final.npy")), TiltReadout(np.load(os.path.join(OUTD, "tilt_readout.npz")))


def evaluate(B=4, seconds=6.0):
    V, ro = load_controller()
    vb = VNCBrain(conn(), batch=B); se = Senses(vb)
    pool = FlyPool([DEFAULT_GENES] * B, workers=B)
    res = {}
    for name, kw in [("legs locked (passive)", dict(V=None)), ("teacher (knows the true tilt)", dict(V=V)),
                     ("real VNC brain", dict(V=V, tilt_fn=ro, brain=(vb, se))),
                     ("brain with senses cut", dict(V=V, tilt_fn=ro, brain=(vb, se), blind=True))]:
        vals = []
        for prof in (TEST_PROF, dict(TEST_PROF, freq=0.3, phase=1.5)):
            vals += list(run(pool, prof, seconds, **kw)["tilt"])
        res[name] = dict(mean_tilt_deg=float(np.mean(vals)), sd=float(np.std(vals)))
        log(name, res[name])
    p = res["legs locked (passive)"]["mean_tilt_deg"]
    for k in list(res):
        res[k]["tilt_removed_pct"] = 100 * (1 - res[k]["mean_tilt_deg"] / p)
    json.dump(res, open(os.path.join(OUTD, "eval.json"), "w"), indent=1)
    pool.close()


def random_genes(rng):
    return {k: float(rng.uniform(*GENE_RANGE[k])) for k in GENE_RANGE}


def score_bodies(G, seconds=3.0):
    V, ro = load_controller()
    n = len(G)
    vb = VNCBrain(conn(), batch=n); se = Senses(vb)
    pool = FlyPool(G)
    br = run(pool, TEST_PROF, seconds, V=V, tilt_fn=ro, brain=(vb, se))["tilt"]
    pas = run(pool, TEST_PROF, seconds)["tilt"]
    pool.close(); del vb; torch.mps.empty_cache()
    return br, pas


def bodies(n=100, seed=7):
    rng = np.random.default_rng(seed)
    G = [dict(DEFAULT_GENES)] + [random_genes(rng) for _ in range(n - 1)]
    t0 = time.time()
    br, pas = score_bodies(G)
    rows = [dict(genes=g, tilt_brain=float(b), tilt_passive=float(p), removed_pct=float(100 * (1 - b / p)))
            for g, b, p in zip(G, br, pas)]
    json.dump(rows, open(os.path.join(OUTD, "bodies100.json"), "w"), indent=1)
    log(f"{n} bodies in {time.time() - t0:.0f}s; default body removes {rows[0]['removed_pct']:.1f}% of tilt")
    order = np.argsort(br)
    for i in list(order[:5]) + list(order[-3:]):
        log(f"  tilt {br[i]:5.2f} (passive {pas[i]:5.2f})  " + ", ".join(f"{k}={v:.2f}" for k, v in G[i].items()))
    keys = list(GENE_RANGE); X = np.array([[g[k] for k in keys] for g in G])
    log("gene vs brain-tilt correlation:", {k: round(float(np.corrcoef(X[:, i], br)[0, 1]), 2) for i, k in enumerate(keys)})


def evolve(gens=12, pop=48, seed=11):
    rng = np.random.default_rng(seed)
    keys = list(GENE_RANGE)
    lo = np.array([GENE_RANGE[k][0] for k in keys]); hi = np.array([GENE_RANGE[k][1] for k in keys])
    P = np.array([[DEFAULT_GENES[k] for k in keys]] + [[random_genes(rng)[k] for k in keys] for _ in range(pop - 1)])
    hist, t0 = [], time.time()
    for g in range(gens):
        genes = [dict(zip(keys, map(float, row))) for row in P]
        br, pas = score_bodies(genes)
        order = np.argsort(br)
        hist.append(dict(gen=g, best=float(br[order[0]]), median=float(np.median(br)), best_genes=genes[order[0]],
                         best_passive=float(pas[order[0]]), pop=[dict(genes=genes[i], tilt=float(br[i]), passive=float(pas[i])) for i in range(pop)]))
        log(f"gen {g:2d}: best tilt {br[order[0]]:5.2f} (passive {pas[order[0]]:5.2f}), median {np.median(br):5.2f}  "
            + ", ".join(f"{k}={v:.2f}" for k, v in genes[order[0]].items()) + f"  ({time.time() - t0:.0f}s)")
        json.dump(hist, open(os.path.join(OUTD, "evolution.json"), "w"), indent=1)
        elite = P[order[:pop // 4]]
        kids = [P[order[0]], P[order[1]]]
        while len(kids) < pop:
            a, b = elite[rng.integers(len(elite), size=2)]
            child = np.where(rng.random(len(keys)) < 0.5, a, b) + rng.normal(0, 0.07, len(keys)) * (hi - lo)
            kids.append(np.clip(child, lo, hi))
        P = np.array(kids)


def video(seconds=10.0, out="balance.mp4"):
    import mujoco
    from PIL import Image
    from flylab.core import Video, text, paste, sparkline
    from flylab.physfly import Body
    from flylab.pool import tilt_gravity
    V, ro = load_controller()
    evo = os.path.join(OUTD, "evolution.json")
    evolved = json.load(open(evo))[-1]["best_genes"] if os.path.exists(evo) else DEFAULT_GENES
    cfg = [("legs locked (no brain)", DEFAULT_GENES, False), ("real nerve-cord brain", DEFAULT_GENES, True),
           ("evolved body + same brain", evolved, True)]
    vb = VNCBrain(conn(), batch=3); se = Senses(vb)
    bodies_ = [Body(g) for _, g, _ in cfg]
    rends = [mujoco.Renderer(b.m, 300, 400) for b in bodies_]
    prof = dict(TEST_PROF)
    st = lambda: (np.array([b.leg_loads() for b in bodies_]), np.array([b.flex() for b in bodies_]),
                  np.array([float(b.d.qvel[3]) for b in bodies_]))
    flat = dict(amp=0.0, freq=0.0, phase=0.0)
    loads, flex, om = st()
    for _ in range(30):
        se.feed(loads, flex, om); vb.step(10.0)
        for b in bodies_:
            b.step(10.0, lambda t: tilt_gravity(t, flat), 0)
        loads, flex, om = st()
    vid = Video(os.path.join(OUTD, out))
    hist = [[], [], []]; est = np.zeros(3)
    for k in range(int(seconds / 0.01)):
        t = k * 0.01
        ups = np.array([b.up() for b in bodies_])
        se.feed(loads, flex, roll_rate(t, prof) + om, grav=np.deg2rad(roll_deg(t, prof)) - np.arctan2(-ups[:, 1], ups[:, 2])); vb.step(10.0)
        est = ro(se.rates(), est)
        for i, (b, (_, _, active)) in enumerate(zip(bodies_, cfg)):
            if active:
                off = np.clip(est[i] * V, -1, 1)
                b.set_targets({(li, j): off[li, ji] for li in range(6) for ji, j in enumerate(JOINTS)})
            b.step(10.0, lambda tt: tilt_gravity(tt, prof), t)
            hist[i].append(b.tilt_vs_gravity())
        loads, flex, om = st()
        roll = roll_deg(t, prof)
        if k % 3:
            continue
        can = np.zeros((720, 1280, 3), np.uint8); can[:] = (10, 11, 16)
        text(can, (20, 14), "A real fly nerve cord balancing a physics-simulated fly", 28, bold=True)
        text(can, (20, 50), f"floor rocking +-{prof['amp']} deg (now {roll:+5.1f}); goal: keep the body level", 16, (255, 190, 60))
        for i, ((name, genes, active), r, b) in enumerate(zip(cfg, rends, bodies_)):
            r.update_scene(b.d, camera="cam_back")
            img = Image.fromarray(r.render()).rotate(roll, resample=Image.BILINEAR, fillcolor=(10, 11, 16))
            x0 = 20 + i * 420
            paste(can, np.asarray(img), x0, 110)
            text(can, (x0, 86), name, 17, bold=True)
            text(can, (x0, 520), f"body tilt {hist[i][-1]:4.1f} deg   (avg {np.mean(hist[i][50:] or [0]):4.1f})", 15)
            paste(can, sparkline(hist[i][-400:], 400, 70, color=(255, 190, 60) if active else (150, 150, 160), lo=0, hi=30), x0, 545)
        if os.path.exists(os.path.join(OUTD, "eval.json")):
            ev = json.load(open(os.path.join(OUTD, "eval.json")))
            text(can, (20, 625), f"measured: real brain removes {ev['real brain (nerve cord + antennae)']['tilt_removed_pct']:.0f}% of the tilt; "
                 f"with its senses cut {ev['brain with senses cut']['tilt_removed_pct']:.0f}%", 16, (255, 190, 60))
        text(can, (20, 652), "Body: flybody (MuJoCo). Brain: 26,925 MaleCNS neurons (VNC + antennal gravity pathway); leg load, joint angle, haltere + antenna -> tilt", 14, (150, 150, 165))
        text(can, (20, 672), "read from 18k non-sensory neurons -> leveling reflex on 42 leg joints. Wiring frozen. Evolved body: " + ", ".join(f"{k} {v:.2f}" for k, v in evolved.items()), 14, (150, 150, 165))
        vid.add(can)
    vid.close(); log("wrote", vid.path, [round(float(np.mean(h[50:])), 2) for h in hist])


if __name__ == "__main__" and (len(sys.argv) < 2 or not sys.argv[1].endswith("4")) and sys.argv[1:2] != ["tune4"]:
    {"teacher": teacher, "readout": readout, "eval": evaluate, "bodies": bodies, "evolve": evolve,
     "video": video}[sys.argv[1] if len(sys.argv) > 1 else "teacher"]()


# ------------------------------------------------------------- v4: brain-sensed tilt error -> PI leveling reflex
class Reflex:
    """Decoded body tilt from vertical (from the brain) -> proportional-integral leveling command u (rad).
    u multiplies the leveling pattern V. The brain only reports the error it senses; the PI rule is the reflex arc."""

    def __init__(self, ro, kp=0.6, ki=6.0):
        self.ro, self.kp, self.ki = ro, kp, ki
        self.u_int = None; self.e = None

    def __call__(self, x, prev):
        e = ((x - self.ro.mu) / self.ro.sd) @ self.ro.P @ self.ro.w + self.ro.b
        self.e = e if self.e is None else 0.6 * self.e + 0.4 * e
        if self.u_int is None:
            self.u_int = np.zeros_like(e)
        self.u_int = np.clip(self.u_int + self.ki * self.e * 0.01, -0.8, 0.8)
        return np.clip(self.u_int + self.kp * self.e, -0.9, 0.9)

    def reset(self):
        self.u_int = None; self.e = None


def run4(pool, prof, seconds, V, brain, reflex=None, record=False, explore=0.0, rng=None, blind=False):
    """Like run(), but labels are the body's signed tilt from vertical, and control is the brain reflex (or the
    teacher u = true roll + exploration drift when reflex is None)."""
    vb, se = brain
    loads, flex, tilt, fallen, om, up = pool.reset()
    n = pool.n; flat = dict(amp=0.0, freq=0.0, phase=0.0)
    vb.fb.reset()
    for _ in range(30):
        se.feed(loads, flex, om, blind, grav=-np.arctan2(-up[:, 1], up[:, 2])); vb.step(10.0)
        loads, flex, tilt, fallen, om, up = pool.step(None, flat)
    if reflex is not None:
        reflex.reset()
    rng = rng or np.random.default_rng(0)
    drift = np.zeros(n); tilts, X, Y = [], [], []
    for k in range(int(seconds / 0.01)):
        t = k * 0.01
        true = np.deg2rad(roll_deg(t, prof))
        grav = true - np.arctan2(-up[:, 1], up[:, 2])
        se.feed(loads, flex, roll_rate(t, prof) + om, blind, grav=grav); vb.step(10.0)
        x = se.rates() if (record or reflex is not None) else None
        if record:
            X.append(x.astype(np.float16)); Y.append(grav.copy())
        drift = 0.98 * drift + explore * rng.standard_normal(n)
        u = (true + drift) if reflex is None else reflex(x, None) + drift
        off = np.clip(u[:, None, None] * V[None], -1.0, 1.0)
        loads, flex, tilt, fallen, om, up = pool.step(off, prof, t)
        tilts.append(tilt)
    T = np.array(tilts)[50:]
    out = dict(tilt=T.mean(0) + 30 * fallen, trace=np.array(tilts))
    if record:
        out["X"], out["Y"] = np.concatenate(X), np.concatenate(Y)
    return out


def readout4(rounds=4, B=16):
    V = np.load(os.path.join(OUTD, "V_final.npy"))
    vb = VNCBrain(conn(), batch=B); se = Senses(vb)
    log(se.describe())
    pool = FlyPool([DEFAULT_GENES] * B)
    Xs, Ys, rep, reflex = [], [], [], None
    rng = np.random.default_rng(3); t0 = time.time()
    for r in range(rounds):
        for i, prof in enumerate(TRAIN_PROFS[:2]):
            prof = dict(prof, phase=prof["phase"] + r)
            use = reflex if (reflex is not None and (r >= 2 or i == 1)) else None
            out = run4(pool, prof, 3.0, V, (vb, se), reflex=use, record=True, explore=0.02, rng=rng)
            Xs.append(out["X"]); Ys.append(out["Y"])
            log(f"  round {r} profile {i}: {'teacher+explore' if use is None else 'brain reflex'} drives, tilt {out['tilt'].mean():.2f} deg")
        ro, r2 = TiltReadout.fit(np.concatenate(Xs).astype(np.float32), np.concatenate(Ys))
        ro.save(os.path.join(OUTD, "tilt_error_readout.npz"))
        reflex = Reflex(ro)
        rep.append(dict(round=r, cv_r2=r2, wall=time.time() - t0))
        json.dump(rep, open(os.path.join(OUTD, "readout4.json"), "w"), indent=1)
    # tune the reflex gains on the training profiles (2 numbers, grid)
    best = None
    for kp in (0.3, 0.6, 1.0):
        for ki in (3.0, 6.0, 10.0):
            rf = Reflex(ro, kp, ki)
            tl = run4(pool, TRAIN_PROFS[0], 3.0, V, (vb, se), reflex=rf)["tilt"].mean()
            log(f"  reflex kp {kp} ki {ki}: tilt {tl:.2f}")
            if best is None or tl < best[0]:
                best = (tl, kp, ki)
    json.dump(dict(kp=best[1], ki=best[2], tilt=best[0]), open(os.path.join(OUTD, "reflex_gains.json"), "w"))
    pool.close()


def load_reflex():
    g = json.load(open(os.path.join(OUTD, "reflex_gains.json")))
    return np.load(os.path.join(OUTD, "V_final.npy")), Reflex(TiltReadout(np.load(os.path.join(OUTD, "tilt_error_readout.npz"))), g["kp"], g["ki"])


def evaluate4(B=4, seconds=6.0):
    V, rf = load_reflex()
    vb = VNCBrain(conn(), batch=B); se = Senses(vb)
    pool = FlyPool([DEFAULT_GENES] * B, workers=B)
    res = {}
    profs = (TEST_PROF, dict(TEST_PROF, freq=0.3, phase=1.5))
    vals = []
    for prof in profs:
        vals += list(run(pool, prof, seconds)["tilt"])
    res["legs locked (passive)"] = dict(mean_tilt_deg=float(np.mean(vals)), sd=float(np.std(vals)))
    vals = []
    for prof in profs:
        vals += list(run(pool, prof, seconds, V=V)["tilt"])
    res["teacher (knows the true tilt)"] = dict(mean_tilt_deg=float(np.mean(vals)), sd=float(np.std(vals)))
    for name, blind in (("real brain (nerve cord + antennae)", False), ("brain with senses cut", True)):
        vals = []
        for prof in profs:
            vals += list(run4(pool, prof, seconds, V, (vb, se), reflex=rf, blind=blind)["tilt"])
        res[name] = dict(mean_tilt_deg=float(np.mean(vals)), sd=float(np.std(vals)))
    p = res["legs locked (passive)"]["mean_tilt_deg"]
    for k in res:
        res[k]["tilt_removed_pct"] = 100 * (1 - res[k]["mean_tilt_deg"] / p)
        log(k, res[k])
    json.dump(res, open(os.path.join(OUTD, "eval.json"), "w"), indent=1)
    pool.close()


def score_bodies4(G, seconds=3.0):
    V, rf = load_reflex()
    n = len(G)
    vb = VNCBrain(conn(), batch=n); se = Senses(vb)
    pool = FlyPool(G)
    br = run4(pool, TEST_PROF, seconds, V, (vb, se), reflex=rf)["tilt"]
    pas = run(pool, TEST_PROF, seconds)["tilt"]
    pool.close(); del vb; torch.mps.empty_cache()
    return br, pas


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("readout4", "eval4", "bodies4", "evolve4", "video4"):
    if sys.argv[1] in ("bodies4", "evolve4", "video4"):
        score_bodies = score_bodies4                                     # bodies()/evolve() use the v4 reflex
        load_controller = lambda: (load_reflex()[0], load_reflex()[1])
    {"readout4": readout4, "eval4": evaluate4, "bodies4": bodies, "evolve4": evolve, "video4": video}[sys.argv[1]]()


def tune4():
    """Wider PI gain search: every (kp, ki) pair is its own body/brain row, scored on two training profiles."""
    V = np.load(os.path.join(OUTD, "V_final.npy"))
    ro = TiltReadout(np.load(os.path.join(OUTD, "tilt_error_readout.npz")))
    grid = [(kp, ki) for kp in (0.6, 1.0, 1.6) for ki in (8.0, 12.0, 18.0, 26.0)]
    n = len(grid)
    vb = VNCBrain(conn(), batch=n); se = Senses(vb)
    pool = FlyPool([DEFAULT_GENES] * n)
    rf = Reflex(ro, np.array([g[0] for g in grid]), np.array([g[1] for g in grid]))
    score = np.zeros(n)
    for prof in TRAIN_PROFS[:2]:
        score += run4(pool, prof, 3.0, V, (vb, se), reflex=rf)["tilt"] / 2
    for g, s_ in zip(grid, score):
        log(f"  kp {g[0]} ki {g[1]}: {s_:.2f} deg")
    i = int(np.argmin(score))
    json.dump(dict(kp=grid[i][0], ki=grid[i][1], tilt=float(score[i])), open(os.path.join(OUTD, "reflex_gains.json"), "w"))
    log("chosen", grid[i], score[i])
    pool.close()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "tune4":
    tune4()
