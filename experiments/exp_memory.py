"""Experiment 5: teach a real fly brain to love the smell of petrol, and to remember people by their smell.

Real mushroom-body learning inside the MaleCNS connectome:
  * Smells drive the fly's olfactory receptor neurons (ORNs) glomerulus by glomerulus -> antennal lobe -> ~900 of
    4,064 Kenyon cells fire for a given smell (a sparse "smell code").
  * Training pairs a smell with the fly's own dopamine neurons: PAM (316 cells, the sugar-reward neurons) or PPL1
    (16 cells, the punishment neurons), exactly like lab conditioning (odour + sugar / odour + shock).
  * The ONLY synapses that change are Kenyon cell -> MBON synapses (where flies store these memories). Rule:
    dopamine-gated depression (Hige et al. 2015; Cohn et al. 2015): a KC->MBON synapse weakens when the KC fires while
    the dopamine neurons of that MBON's compartment fire. Compartments are read from the connectome (DAN->MBON synapses).
    The other ~25.6M synapses stay frozen.
  * Readout: MBON valence = mean rate of approach-type MBONs (cholinergic/GABAergic) minus avoidance-type MBONs
    (glutamatergic) (Aso et al. 2014 rule of thumb).
  * Test: a virtual T-maze, the standard assay. A fly at the junction samples arm 1, then arm 2, and walks into the arm
    whose smell gave the higher valence (+ noise). Preference index PI = (n_arm1 - n_arm2) / n.

Smells (glomerulus sets are stand-ins, NOT measured receptor data for these mixtures):
  petrol   aromatic-hydrocarbon-like stand-in: DL1, DM5, VM7d, VC4, DC2
  Alex     sweaty-acid body odour stand-in (ionotropic acid/amine glomeruli): DC4, DP1m, DL2d, VM1, DC1
  Sam      ester/perfume-ish body odour stand-in: DM2, DM3, VA2, DL3, VM2
  banana   control fruit smell: DM1, DM4, VA5, DM6

Brain: everything except the optic lobes and visual projection neurons (the eyes are not needed to smell):
antennal lobe + mushroom body + all central brain + descending neurons + VNC = 53,446 MaleCNS neurons.

    python exp_memory.py run       # petrol + people experiments, paired vs sham -> out/memory/results.json
    python exp_memory.py video     # T-maze animation, before/after
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylab.core import OUT, Video, brain, log, paste, resize, sparkline, text, Timer

OUTD = os.path.join(OUT, "memory")
MODULES = ["antennal_lobe", "mushroom_body", "central", "descending", "vnc"]
SMELLS = {
    "petrol": ["DL1", "DM5", "VM7d", "VC4", "DC2"],
    "Alex": ["DC4", "DP1m", "DL2d", "VM1", "DC1"],
    "Sam": ["DM2", "DM3", "VA2", "DL3", "VM2"],
    "banana": ["DM1", "DM4", "VA5", "DM6"],
}
CONC = 0.8


def dan_mbon_counts(mbon_ids, dan_ids):
    """Raw DAN -> MBON synapse counts (MBON x DAN) from the MaleCNS weights file. flybrain's LIF matrix gives
    dopamine synapses sign 0 (no fast effect), so compartments must come from the raw counts. Cached."""
    os.makedirs(OUTD, exist_ok=True)
    cache = os.path.join(OUTD, "dan_mbon_counts.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        if np.array_equal(d["m"], mbon_ids) and np.array_equal(d["d"], dan_ids):
            return d["C"]
    import pyarrow.compute as pc
    import pyarrow.feather as pf
    from flylab.core import FLYBRAIN
    t = pf.read_table(os.path.join(FLYBRAIN, "data", "connectome-weights-male-cns-v1.0-minconf-0.5.feather"))
    t = t.filter(pc.is_in(t["body_pre"], value_set=__import__("pyarrow").array(dan_ids)))
    t = t.filter(pc.is_in(t["body_post"], value_set=__import__("pyarrow").array(mbon_ids)))
    pre, post, w = (t[k].to_numpy() for k in ("body_pre", "body_post", "weight"))
    mi = {b: i for i, b in enumerate(mbon_ids)}; di = {b: i for i, b in enumerate(dan_ids)}
    C = np.zeros((len(mbon_ids), len(dan_ids)), np.float32)
    for a, b, x in zip(pre, post, w):
        C[mi[b], di[a]] += x
    np.savez(cache, m=mbon_ids, d=dan_ids, C=C)
    return C


class MushroomBody:
    def __init__(self, fb, eta=0.02):
        self.fb, self.eta = fb, eta
        b = fb.brain
        assert getattr(b, "_out_w", None) is not None, "needs the event-driven weight layout (Metal/CPU)"
        c = fb.c
        t = c.neurons.type.fillna("").to_numpy().astype(str)
        self.kc = np.flatnonzero(np.char.startswith(t, "KC"))
        self.mbon = np.flatnonzero(np.char.startswith(t, "MBON"))
        self.ppl1 = np.flatnonzero(np.char.startswith(t, "PPL1"))
        self.pam = np.flatnonzero(np.char.startswith(t, "PAM"))
        dans = np.r_[self.ppl1, self.pam]
        Wc = sp.csc_matrix(b._W_cpu); Wc.sum_duplicates(); Wc.sort_indices()
        assert Wc.nnz == b._out_w.numel(), "weight layout mismatch"
        is_mbon = np.zeros(c.n, bool); is_mbon[self.mbon] = True
        slot = np.full(c.n, -1); slot[self.mbon] = np.arange(len(self.mbon))
        kslot = np.full(c.n, -1); kslot[self.kc] = np.arange(len(self.kc))
        pos, pre, post = [], [], []
        for k in self.kc:
            a, z = Wc.indptr[k], Wc.indptr[k + 1]
            rows = Wc.indices[a:z]; m = is_mbon[rows]
            pos.append(np.arange(a, z)[m]); pre.append(np.full(m.sum(), kslot[k])); post.append(slot[rows[m]])
        dev = fb.device
        self.pos = torch.as_tensor(np.concatenate(pos), device=dev)
        self.e_kc = torch.as_tensor(np.concatenate(pre), device=dev)
        self.e_mb = torch.as_tensor(np.concatenate(post), device=dev)
        self.w0 = b._out_w[self.pos].clone()
        C = dan_mbon_counts(c.neurons.bodyId.to_numpy()[self.mbon], c.neurons.bodyId.to_numpy()[dans])
        Craw = C.copy()
        self.plastic_mbons = int((C.sum(1) > 0).sum())
        C /= np.maximum(C.sum(1, keepdims=True), 1e-9)
        self.C = torch.as_tensor(C, device=dev)
        self.dans_t = torch.as_tensor(dans, device=dev)
        self.kc_t = torch.as_tensor(self.kc, device=dev)
        self.mbon_t = torch.as_tensor(self.mbon, device=dev)
        # valence of each MBON from its own compartment (opponent model, Aso et al. 2014): MBONs in PPL1 (punishment)
        # compartments promote approach (punishment depresses them -> avoidance); MBONs in PAM (reward) compartments
        # promote avoidance (reward depresses them -> approach). Read from the raw DAN->MBON synapse counts.
        share_ppl1 = Craw[:, :len(self.ppl1)].sum(1) / np.maximum(Craw.sum(1), 1e-9)
        self.sign = torch.as_tensor(np.where(share_ppl1 > 0.5, 1.0, -1.0), device=dev, dtype=torch.float32)
        self.n_approach = int((share_ppl1 > 0.5).sum())
        self.kc_base = None

    def describe(self):
        return (f"{len(self.kc):,} Kenyon cells, {len(self.mbon)} MBONs ({self.plastic_mbons} receive dopamine), "
                f"{len(self.ppl1)} PPL1 punishment + {len(self.pam)} PAM reward dopamine neurons, "
                f"{len(self.pos):,} plastic KC->MBON synapses")

    def plasticity(self):
        r = self.fb.brain.rate
        kc = r[:, self.kc_t]
        kc = torch.relu(kc - self.kc_base) / 20.0 if self.kc_base is not None else kc / 20.0   # smell-evoked part only
        dm = (r[:, self.dans_t] / 50.0) @ self.C.T
        drive = (kc[:, self.e_kc] * dm[:, self.e_mb]).mean(0)
        w = self.fb.brain._out_w
        w[self.pos] = torch.maximum(w[self.pos] * torch.exp(-self.eta * drive), 0.05 * self.w0)

    def valence(self):
        return (self.fb.brain.rate[:, self.mbon_t] * self.sign).mean(1).cpu().numpy()

    def kc_rates(self):
        return self.fb.brain.rate[:, self.kc_t].cpu().numpy()

    def relative_weights(self):
        return (self.fb.brain._out_w[self.pos] / self.w0).cpu().numpy()


class Nose:
    """B flies (batch rows of one brain; the plastic synapses are shared, so they are B trials of the same fly)."""

    def __init__(self, B=16, seed=0):
        self.B = B
        self.fb = brain(B, modules=MODULES, seed=seed)
        self.mb = MushroomBody(self.fb)
        self.rng = np.random.default_rng(seed)
        self.glom = sorted({g for v in SMELLS.values() for g in v})
        self.hook = None

    def smell(self, names):
        """names: one smell name (or None) per row, or a single name for all rows."""
        if names is None or isinstance(names, str):
            names = [names] * self.B
        conc = {g: np.array([CONC if (n and g in SMELLS[n]) else 0.0 for n in names]) for g in self.glom}
        self.fb.smell(conc, conc)

    def dopamine(self, kind):
        mb = self.mb
        self.fb._input("ppl1", mb.ppl1, np.full((self.B, len(mb.ppl1)), 150.0 if kind == "punish" else 0.0, np.float32))
        self.fb._input("pam", mb.pam, np.full((self.B, len(mb.pam)), 60.0 if kind == "reward" else 0.0, np.float32))

    def run(self, seconds, smell, dop=None, learn=False, tag=""):
        self.smell(smell); self.dopamine(dop)
        vals = []
        for k in range(int(round(seconds / 0.01))):
            self.fb.step(10.0)
            if learn:
                self.mb.plasticity()
            v = self.mb.valence(); vals.append(v)
            if self.hook:
                self.hook(self, k, tag, smell, dop, v)
        self.dopamine(None)
        if tag == "settle":
            self.mb.kc_base = self.fb.brain.rate[:, self.mb.kc_t].clone()
        return np.array(vals)

    def sample(self, smell, seconds=1.2):
        v = self.run(seconds, smell, tag=f"sniff {smell}")
        self.run(0.4, None, tag="air")
        return v[40:].mean(0)                             # (B,)

    def tmaze(self, arm1, arm2, noise=None):
        """Every fly samples both arms; returns preference index for arm1, valences and choices."""
        v1, v2 = self.sample(arm1), self.sample(arm2)
        if noise is None:
            noise = getattr(self, "noise", 0.3)
        choice = (v1 - v2 + noise * self.rng.standard_normal(self.B)) > 0
        pi = float((choice.sum() - (~choice).sum()) / self.B)
        return dict(arm1=arm1, arm2=arm2, PI=pi, v1=float(v1.mean()), v2=float(v2.mean()), choices=choice.tolist())

    def train(self, pairs, trials=6):
        """pairs: [(smell, 'reward'|'punish'|None)], presented in alternation (differential conditioning)."""
        for i in range(trials):
            for smell, dop in pairs:
                self.run(0.4, smell, tag=f"training {i + 1}: {smell}")
                self.run(1.0, smell, dop, learn=True, tag=f"training {i + 1}: {smell} + {dop or 'nothing'}")
                self.run(0.6, None, tag="air")


def experiment(name, pairs, tests, paired=True, B=16):
    nose = Nose(B, seed=0)
    nose.run(1.0, None, tag="settle")
    # calibrate decision noise from smell-to-smell valence variability of the untrained fly
    before = [nose.tmaze(a, b) for a, b in tests]
    nose.train([(s, d if paired else None) for s, d in pairs])
    nose.run(1.0, None, tag="air")
    after = [nose.tmaze(a, b) for a, b in tests]
    w = nose.mb.relative_weights()
    return dict(experiment=name, paired=paired, mushroom_body=nose.mb.describe(),
                before=before, after=after, weakened_synapses=int((w < 0.8).sum()), plastic_synapses=int(len(w)))


EXPERIMENTS = {
    "petrol": dict(pairs=[("petrol", "reward"), ("banana", None)],
                   tests=[("petrol", "banana"), ("petrol", "Sam"), ("petrol", "Alex")]),
    "people": dict(pairs=[("Sam", "reward"), ("Alex", "punish")],
                   tests=[("Sam", "Alex"), ("Sam", "banana"), ("Alex", "banana")]),
}


def run():
    os.makedirs(OUTD, exist_ok=True)
    T = Timer(); out = []
    for name, spec in EXPERIMENTS.items():
        for paired in (True, False):
            r = experiment(name, spec["pairs"], spec["tests"], paired)
            out.append(r)
            log(f"{name} {'paired' if paired else 'SHAM'}: " + "; ".join(
                f"{b['arm1']} vs {b['arm2']}: PI {b['PI']:+.2f} -> {a['PI']:+.2f} (valence {b['v1'] - b['v2']:+.2f} -> {a['v1'] - a['v2']:+.2f})"
                for b, a in zip(r["before"], r["after"])) + f"  weakened {r['weakened_synapses']}  ({T():.0f}s)")
            json.dump(out, open(os.path.join(OUTD, "results.json"), "w"), indent=1)


# ----------------------------------------------------------------------------------------------- video
def maze_img(flies, arms, W=560, H=420, colors=((255, 170, 60), (120, 200, 255))):
    """Top-down T-maze. flies: list of (x, y) in [0,1]^2 maze coords; arms: (left label, right label)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (14, 16, 22)); dr = ImageDraw.Draw(img)
    cx = W // 2
    dr.rectangle([cx - 30, 150, cx + 30, H - 20], fill=(40, 44, 56))               # stem
    dr.rectangle([30, 90, W - 30, 150], fill=(40, 44, 56))                         # arms
    for i, (lab, col) in enumerate(zip(arms, colors)):
        x0 = 30 if i == 0 else W - 150
        dr.rectangle([x0, 90, x0 + 120, 150], fill=tuple(int(c * 0.35) for c in col))
        for k in range(6):                                                         # smell plume dots
            dr.ellipse([x0 + 10 + 18 * k, 60 - 6 * (k % 2), x0 + 18 + 18 * k, 68 - 6 * (k % 2)], fill=tuple(int(c * 0.6) for c in col))
    for (x, y, c) in flies:
        px, py = 30 + x * (W - 60), 20 + y * (H - 40)
        dr.ellipse([px - 6, py - 4, px + 6, py + 4], fill=c)
        dr.ellipse([px - 9, py - 7, px - 2, py - 2], fill=(200, 200, 230)); dr.ellipse([px + 2, py - 7, px + 9, py - 2], fill=(200, 200, 230))
    return np.asarray(img).copy()


def video(out="petrol_memory.mp4", exp="petrol"):
    spec = EXPERIMENTS[exp]
    a1, a2 = spec["tests"][0]
    nose = Nose(16, seed=0)
    vid = Video(os.path.join(OUTD, out))
    state = dict(phase="", hist=[], dop=None, tick=0)

    def hook(n, k, tag, smell, dop, v):
        state["hist"].append(float(v.mean())); state["dop"] = dop; state["tag"] = tag; state["smell"] = smell
        state["tick"] += 1
        if state["tick"] % (6 if tag.startswith("training") else 3):
            return
        frame(None)

    def frame(flies, walking=None):
        can = np.zeros((720, 1280, 3), np.uint8); can[:] = (10, 11, 16)
        text(can, (20, 14), "Teaching a real fly brain to love the smell of petrol" if exp == "petrol" else
             "A real fly brain that remembers people by their smell", 30, bold=True)
        text(can, (20, 54), state["phase"], 18, (255, 190, 60))
        if flies is None:
            flies = [(0.5 + 0.03 * np.sin(i), 0.82 + 0.02 * (i % 4), (230, 230, 240)) for i in range(16)]
        paste(can, maze_img(flies, (a1, a2)), 20, 100)
        text(can, (40, 120), f"{a1}", 18, bold=True); text(can, (470, 120), f"{a2}", 18, bold=True)
        rates = nose.mb.kc_rates()[0]
        g = np.zeros(64 * 64, np.float32); g[:len(rates)] = np.sqrt(np.clip(rates / 6.0, 0, 1))
        g = g.reshape(64, 64)
        paste(can, resize(np.stack([g * 255, g * 220, g * 120 + 20], -1).astype(np.uint8), 240, 240), 620, 100)
        text(can, (620, 80), "Kenyon cells (4,064): the smell code", 14, (150, 150, 165))
        d = state.get("dop")
        if d:
            text(can, (880, 120), "PAM reward dopamine" if d == "reward" else "PPL1 punishment dopamine", 20,
                 (90, 255, 140) if d == "reward" else (255, 90, 90), bold=True)
            text(can, (880, 150), "(the fly's own 'sugar' neurons)" if d == "reward" else "(the fly's own 'shock' neurons)", 15)
        if state.get("smell"):
            text(can, (880, 190), f"smelling: {state['smell']}", 18)
        wr = nose.mb.relative_weights()
        hist, _ = np.histogram(np.clip(wr, 0, 1.0001), bins=20, range=(0, 1.0001))
        bars = np.zeros((120, 340, 3), np.uint8); bars[:] = (20, 22, 30)
        hmax = max(1, np.sort(hist)[-2])
        for i, hcount in enumerate(hist):
            hh = int(min(1, hcount / hmax) * 110)
            bars[120 - hh:, i * 17:i * 17 + 14] = (255, 190, 60) if i < 19 else (90, 90, 110)
        paste(can, bars, 880, 260)
        text(can, (880, 240), f"KC->MBON synapse strength (weakened: {(wr < 0.8).sum():,} of {len(wr):,})", 14, (150, 150, 165))
        paste(can, sparkline(state["hist"][-500:], 560, 80, lo=-3, hi=3), 620, 420)
        text(can, (620, 400), "mushroom-body valence (approach + / avoid -)", 14, (150, 150, 165))
        if walking:
            text(can, (20, 540), walking, 20, bold=True)
        text(can, (20, 600), "Only Kenyon cell -> MBON synapses can change (dopamine-gated, as in real flies). The other", 15, (150, 150, 165))
        text(can, (20, 622), "~11M synapses of this 53,446-neuron MaleCNS brain (all but the eyes) are frozen. Petrol = aromatic", 15, (150, 150, 165))
        text(can, (20, 644), "stand-in glomeruli (approximate). 16 flies = 16 runs of one brain. Apple M2.", 15, (150, 150, 165))
        vid.add(can)

    def maze_test(label):
        state["phase"] = label
        r = nose.tmaze(a1, a2)
        choices = np.array(r["choices"])
        pts = []
        for i in range(16):
            tx = 0.08 + 0.12 * (i % 4) / 3 if choices[i] else 0.80 + 0.12 * (i % 4) / 3
            pts.append((tx, 0.20 + 0.06 * (i // 4) / 3))
        n1 = int(choices.sum())
        for s in np.linspace(0, 1, 45):
            fl = []
            for i in range(16):
                sx, sy = 0.5 + 0.03 * np.sin(i), 0.82 + 0.02 * (i % 4)
                if s < 0.5:
                    x, y = sx, sy + (0.2 - sy) * (s / 0.5)
                else:
                    x, y = sx + (pts[i][0] - sx) * ((s - 0.5) / 0.5), 0.2
                fl.append((x, y, (255, 170, 60) if choices[i] else (120, 200, 255)))
            frame(fl, f"{n1} of 16 flies chose {a1}, {16 - n1} chose {a2}   (preference index {r['PI']:+.2f})")
        for _ in range(110):
            frame(fl, f"{n1} of 16 flies chose {a1}, {16 - n1} chose {a2}   (preference index {r['PI']:+.2f})")
        return r

    nose.hook = hook
    state["phase"] = "1 · Naive fly: smells both arms, then picks"
    nose.run(1.0, None, tag="settle")
    nose.hook = hook
    before = maze_test("1 · Naive fly: smells both arms of the T-maze, then picks one")
    state["phase"] = f"2 · Training: {a1} + the fly's reward dopamine, {spec['pairs'][1][0]} + nothing ({'x6' })" if exp == "petrol" else "2 · Training: Sam + reward dopamine, Alex + punishment dopamine (x6)"
    nose.train([(s, d) for s, d in spec["pairs"]])
    nose.run(1.0, None, tag="air")
    after = maze_test("3 · Memory test: same T-maze, no dopamine any more")
    vid.close()
    log("video", vid.path, before["PI"], "->", after["PI"])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "video":
        video(*(sys.argv[2:4] if len(sys.argv) > 3 else []))
