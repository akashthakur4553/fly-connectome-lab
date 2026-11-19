"""Experiment 1: a real fruit-fly brain drives a car.

The car carries the fly's two compound eyes (the real MaleCNS retina geometry) 2 cm above a race track lined with
poles (black inside, white outside). Every 10 ms the scene is ray-traced into ~1,500 ommatidia, fed to the full
167,106-neuron connectome (optic lobe + central brain + descending neurons + VNC), and a steering command is read
out of the brain's visual projection and descending neurons.

The connectome is never edited. Only a linear steering readout (PCA -> ridge) is learned, by DAgger imitation:
drive with a mix of a track-following "driving instructor" and the fly's own readout, record the fly's neurons,
label every frame with what the instructor would have done, refit, repeat.

    python exp_car.py train            # DAgger rounds -> out/car/readout.npz
    python exp_car.py test             # trained fly vs native steering vs blind fly, on the training + unseen track
    python exp_car.py video            # the LinkedIn clip
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylab.core import (OUT, Eye, BrainPanel, Video, brain, conn, log, paste, resize, sparkline, text, yaw_R,
                         to_rgb8, Timer)
from flylab.track import Track, stadium, kidney

OUTD = os.path.join(OUT, "car")
SPEED = 0.25            # m/s  (about 100 fly body-lengths per second: a fast RC car for a fly)
W_MAX = 2.2             # rad/s full lock
EYE_H = 0.02
DT = 0.01


def visual_cells(c):
    from flybrain import regions
    lab = regions.labels(c)
    return np.flatnonzero((lab == "visual_projection") | (lab == "descending"))


class Car:
    def __init__(self, track: Track, B: int, rng):
        self.track, self.B, self.rng = track, B, rng
        self.xy = np.zeros((B, 2)); self.h = np.zeros(B)
        self.progress = np.zeros(B); self.crashes = np.zeros(B, int)
        self.u = np.zeros(B)
        for b in range(B):
            self.respawn(b, rng.integers(len(track.c)))

    def respawn(self, b, i=None):
        t = self.track
        if i is None:
            i = t.locate(self.xy[b:b + 1])[0][0]
        self.xy[b] = t.c[i]; self.h[b] = t.heading[i]

    def expert(self):
        i, off, th, k = self.track.locate(self.xy)
        look = (i + 25) % len(self.track.c)                                    # aim ~10 cm ahead
        tgt = self.track.c[look]
        want = np.arctan2(tgt[:, 1] - self.xy[:, 1], tgt[:, 0] - self.xy[:, 0])
        err = np.angle(np.exp(1j * (want - self.h)))
        return np.clip(3.0 * err / W_MAX + 0.5 * (-off) / (self.track.width / 2) * 0.3, -1, 1)

    def step(self, u):
        self.u = np.clip(u, -1, 1)
        i0, _, th, _ = self.track.locate(self.xy)
        self.h = self.h + self.u * W_MAX * DT
        v = np.c_[np.cos(self.h), np.sin(self.h)] * SPEED * DT
        self.xy = self.xy + v
        self.progress += (v * self.track.tan[i0]).sum(-1)
        _, off, _, _ = self.track.locate(self.xy)
        crashed = np.abs(off) > self.track.width / 2 - 0.006
        for b in np.flatnonzero(crashed):
            self.crashes[b] += 1; self.respawn(b)
        return crashed

    def eye(self):
        return np.c_[self.xy, np.full(self.B, EYE_H)], yaw_R(self.h)


class Pilot:
    """Linear steering readout from ~11k visual-projection + descending neurons (z-score -> PCA -> ridge)."""

    def __init__(self, d):
        self.mu, self.sd, self.V, self.w, self.b = d["mu"], d["sd"], d["V"], d["w"], float(d["b"])

    @classmethod
    def fit(cls, X, y, k=256, lams=(1, 10, 100, 1000, 1e4)):
        X = X.astype(np.float32)
        mu, sd = X.mean(0), X.std(0) + 0.05
        Z = (X - mu) / sd
        n = len(Z)
        Zt = torch.from_numpy(Z)
        U, S, V = torch.pca_lowrank(Zt, q=min(k, n - 1), center=False, niter=4)
        P = (Zt @ V).numpy()
        rng = np.random.default_rng(0)
        blocks = np.array_split(np.arange(n), 5)                                   # contiguous folds: honest CV
        best = None
        for lam in lams:
            r2 = []
            for f in blocks:
                tr = np.setdiff1d(np.arange(n), f)
                A = P[tr]; ym = y[tr].mean()
                w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - ym))
                pred = P[f] @ w + ym
                r2.append(1 - ((y[f] - pred) ** 2).sum() / max(((y[f] - y[f].mean()) ** 2).sum(), 1e-9))
            if best is None or np.mean(r2) > best[0]:
                best = (float(np.mean(r2)), lam)
        lam = best[1]; ym = y.mean()
        w = np.linalg.solve(P.T @ P + lam * np.eye(P.shape[1]), P.T @ (y - ym))
        log(f"  readout fit: {n} frames, {X.shape[1]} neurons -> {P.shape[1]} PCs, ridge {lam:g}, CV R2 {best[0]:.3f}")
        return cls(dict(mu=mu, sd=sd, V=V.numpy(), w=w.astype(np.float32), b=ym)), best[0]

    def __call__(self, X):
        return ((X - self.mu) / self.sd) @ self.V @ self.w + self.b

    def save(self, path):
        np.savez(path, mu=self.mu, sd=self.sd, V=self.V, w=self.w, b=self.b)


class Rig:
    """One brain (batch B) + eye + cars on a track."""

    def __init__(self, track, B=4, seed=0):
        self.track, self.world = track, track.world()
        self.fb = brain(B, seed=seed)
        self.eye = Eye(self.fb)
        self.cells = torch.as_tensor(visual_cells(self.fb.c), device=self.fb.device)
        self.car = Car(track, B, np.random.default_rng(seed))
        self.native = __import__("flybrain.body", fromlist=["Locomotion"]).Locomotion()

    def sense(self, blind=False):
        eye, R = self.car.eye()
        rad = self.eye.look(self.world, eye, R)
        if blind:
            rad = torch.zeros_like(rad)
        self.rad = rad
        self.fb.vision(rad)
        self.fb.step(10.0)
        return self.fb.brain.rate[:, self.cells].float().cpu().numpy()

    def native_u(self):
        m = self.fb.motor()
        out = []
        for b in range(self.car.B):
            cmd = self.native.readout(m.row(b) if self.car.B > 1 else m, dt_s=DT)
            out.append(cmd["yaw"] / W_MAX)
        return np.array(out)


def collect(rig, frames, pilot=None, beta=1.0, noise=0.25, rng=None, warm=40):
    rng = rng or np.random.default_rng(0)
    X, Y = [], []
    for k in range(warm + frames):
        x = rig.sense()
        y = rig.car.expert()
        if k >= warm:
            X.append(x.astype(np.float16)); Y.append(y)
        if pilot is None or rng.random() < beta:
            u = y + noise * rng.standard_normal(rig.car.B)
        else:
            u = pilot(x)
        rig.car.step(u)
    return np.concatenate(X), np.concatenate(Y)


def train(rounds=4, frames=1500, B=4):
    os.makedirs(OUTD, exist_ok=True)
    rig = Rig(Track(stadium()), B=B)
    log(f"brain: {rig.fb.c.n} neurons on {rig.fb.device}; readout listens to {len(rig.cells)} neurons; {rig.eye.n_col} eye columns")
    T = Timer(); Xs, Ys, pilot, report = [], [], None, []
    for r in range(rounds):
        beta = [1.0, 0.5, 0.25, 0.0][min(r, 3)]
        X, Y = collect(rig, frames, pilot, beta=beta, rng=np.random.default_rng(r))
        Xs.append(X); Ys.append(Y)
        crashes = int(rig.car.crashes.sum()); rig.car.crashes[:] = 0
        pilot, r2 = Pilot.fit(np.concatenate(Xs).astype(np.float32), np.concatenate(Ys))
        report.append(dict(round=r, beta=beta, frames=int(len(np.concatenate(Ys))), cv_r2=r2, crashes_during=crashes,
                           fly_seconds=frames * B * DT, wall_s=T()))
        log(f"round {r}: beta {beta}, crashes while collecting {crashes}, CV R2 {r2:.3f}  ({T():.0f}s)")
        pilot.save(os.path.join(OUTD, "readout.npz"))
    json.dump(report, open(os.path.join(OUTD, "train.json"), "w"), indent=1)


def drive(track, mode, seconds=30, B=2, seed=1):
    rig = Rig(track, B=B, seed=seed)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    for _ in range(40):
        rig.sense(blind=(mode == "blind"))
    rig.car.crashes[:] = 0; rig.car.progress[:] = 0
    for k in range(int(seconds / DT)):
        x = rig.sense(blind=(mode == "blind"))
        u = rig.native_u() if mode == "native" else pilot(x)
        rig.car.step(u)
    laps = rig.car.progress / track.P
    return dict(mode=mode, seconds=seconds, cars=B, crashes=rig.car.crashes.tolist(),
                crashes_per_min=float(rig.car.crashes.mean() * 60 / seconds), laps=laps.round(2).tolist())


def test(seconds=20):
    res = []
    for name, tr in [("training oval", Track(stadium())), ("unseen bean track", Track(kidney()))]:
        for mode in ("trained", "native", "blind"):
            r = drive(tr, mode, seconds); r["track"] = name; res.append(r)
            log(json.dumps(r))
    json.dump(res, open(os.path.join(OUTD, "test.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------- video
def topdown(track, xy, h, trail, W=460, H=460, crashes=0):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (14, 16, 22))
    dr = ImageDraw.Draw(img)
    lo, hi = track.c.min(0) - 0.15, track.c.max(0) + 0.15
    s = min((W - 20) / (hi - lo)[0], (H - 20) / (hi - lo)[1])
    def P(p): return (10 + (p[0] - lo[0]) * s, H - 10 - (p[1] - lo[1]) * s)
    road = [P(p) for p in track.c] + [P(track.c[0])]
    dr.line(road, fill=(46, 50, 62), width=int(track.width * s))
    dr.line(road, fill=(80, 80, 60), width=1)
    for p, side in track.posts():
        x, y = P(p); r = 2.5
        dr.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 240) if side < 0 else (90, 90, 100))
    if len(trail) > 1:
        dr.line([P(p) for p in trail], fill=(255, 190, 60), width=2)
    x, y = P(xy)
    c, si = np.cos(h), np.sin(h)
    pts = [(x + 11 * c, y - 11 * si), (x - 7 * c + 6 * si, y + 7 * si + 6 * c), (x - 7 * c - 6 * si, y + 7 * si - 6 * c)]
    dr.polygon(pts, fill=(255, 80, 80))
    return np.asarray(img).copy()


def wheel(u, w=200, h=46):
    img = np.zeros((h, w, 3), np.uint8); img[:] = (20, 22, 30)
    mid = w // 2; x = int(mid - u * (w // 2 - 6))
    img[h // 2 - 2:h // 2 + 2, 6:w - 6] = (60, 64, 80)
    lo, hi = sorted((mid, x))
    img[h // 2 - 8:h // 2 + 8, lo:hi + 1] = (90, 200, 255)
    img[:, mid - 1:mid + 1] = (200, 200, 200)
    return img


def video(seconds=22, track_name="stadium", out="car_drive.mp4", mode="trained"):
    tr = Track(stadium() if track_name == "stadium" else kidney())
    rig = Rig(tr, B=1, seed=3)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    panel = BrainPanel(rig.fb, 420, 300)
    vid = Video(os.path.join(OUTD, out), fps=30)
    for _ in range(40):
        rig.sense(blind=mode == "blind")
    rig.car.crashes[:] = 0; rig.car.progress[:] = 0
    trail, us, n = [], [], int(seconds / DT)
    stats = json.load(open(os.path.join(OUTD, "test.json"))) if os.path.exists(os.path.join(OUTD, "test.json")) else []
    for k in range(n):
        x = rig.sense(blind=mode == "blind")
        u = rig.native_u() if mode == "native" else pilot(x)
        rig.car.step(u); us.append(float(np.clip(u[0], -1, 1)))
        trail.append(rig.car.xy[0].copy()); trail = trail[-400:]
        if k % 3:
            continue
        can = np.zeros((720, 1280, 3), np.uint8); can[:] = (10, 11, 16)
        eye, R = rig.car.eye()
        f = R[0][:, 0]; up = np.array([0, 0, 1.0])
        cam = rig.world.render_camera(eye[0] - 0.05 * f + np.array([0, 0, 0.03]), f - np.array([0, 0, 0.18]), up,
                                      width=420, height=236, fov_deg=100)
        paste(can, to_rgb8(cam, exposure=1.4), 20, 70)
        paste(can, resize(rig.eye.picture(rig.rad[0].cpu().numpy()), 420, 194), 20, 360)
        paste(can, topdown(tr, rig.car.xy[0], rig.car.h[0], trail, 380, 380), 460, 70)
        paste(can, panel.draw(0), 850, 70)
        paste(can, wheel(us[-1], 380, 46), 460, 480)
        paste(can, sparkline(us[-300:], 380, 70, lo=-1, hi=1), 460, 560)
        text(can, (20, 18), "A real fruit-fly brain driving a car" + ("" if track_name == "stadium" else "  ·  on a track it has never seen") + ("  ·  UNTRAINED (its own steering neurons)" if mode == "native" else ""), 26, bold=True)
        text(can, (20, 50), "chase cam", 14, (150, 150, 165))
        text(can, (20, 340), "what the fly's compound eyes see (real retina, 1,466 columns)", 14, (150, 150, 165))
        text(can, (460, 50), f"track  ·  laps {rig.car.progress[0] / tr.P:4.2f}  ·  crashes {rig.car.crashes[0]}", 14, (150, 150, 165))
        text(can, (850, 50), "167,106 neurons of the MaleCNS connectome (glow = firing)", 14, (150, 150, 165))
        text(can, (460, 460), "steering read from the fly's neurons", 14, (150, 150, 165))
        text(can, (460, 540), "steering, last 9 s", 14, (150, 150, 165))
        text(can, (850, 390), "brain: frozen wiring, never edited", 18, (255, 190, 60))
        text(can, (850, 418), "learned: one linear readout (steering wheel)", 18, (255, 190, 60))
        text(can, (850, 446), f"listening to {len(rig.cells):,} visual + descending neurons", 16)
        text(can, (850, 474), "simulated on an Apple M2 laptop GPU", 16)
        text(can, (850, 520), f"t = {k * DT:5.2f} s (fly time)", 16, (150, 150, 165))
        for i, r in enumerate([r for r in stats if r["track"] == ("training oval" if track_name == "stadium" else "unseen bean track")]):
            text(can, (850, 570 + 24 * i), f"{r['mode']:>8}: {r.get('crash_free_pct', 0):5.1f}% crash-free  ({r['crashes_per_min']:.1f} crashes/min)", 15)
        vid.add(can)
    vid.close()
    log("wrote", vid.path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "test":
        test()
    elif cmd == "video":
        video(track_name=sys.argv[2] if len(sys.argv) > 2 else "stadium",
              out=sys.argv[3] if len(sys.argv) > 3 else "car_drive.mp4",
              mode=sys.argv[4] if len(sys.argv) > 4 else "trained")


# ------------------------------------------------------------------- v2: train on several tracks, test on the unseen one
def crash_free_pct(crash_times, seconds, window=2.0):
    """% of `window`-second segments of driving with no crash."""
    n = int(seconds / window)
    bad = {int(t // window) for t in crash_times}
    return 100.0 * (n - len(bad)) / max(n, 1)


def train2(rounds=4, frames=900, B=2):
    from flylab.track import wavy
    tracks = {"oval ccw": Track(stadium()), "oval cw": Track(stadium()[::-1].copy()), "wavy": Track(wavy())}
    rigs = {k: Rig(t, B=B, seed=i) for i, (k, t) in enumerate(tracks.items())}
    pilot = None                                  # from scratch: the v1 readout had learned room cues
    T, Xs, Ys, report = Timer(), [], [], []
    for r in range(rounds):
        beta = [1.0, 0.5, 0.25, 0.0][min(r, 3)]
        crashes = {}
        for name, rig in rigs.items():
            rig.car.crashes[:] = 0
            X, Y = collect(rig, frames, pilot, beta=beta, rng=np.random.default_rng(100 + r))
            Xs.append(X); Ys.append(Y); crashes[name] = int(rig.car.crashes.sum())
        pilot, r2 = Pilot.fit(np.concatenate(Xs).astype(np.float32), np.concatenate(Ys))
        pilot.save(os.path.join(OUTD, "readout.npz"))
        report.append(dict(round=r, beta=beta, cv_r2=r2, crashes=crashes, wall_s=T()))
        log(f"v2 round {r}: beta {beta}, crashes {crashes}, CV R2 {r2:.3f} ({T():.0f}s)")
        json.dump(report, open(os.path.join(OUTD, "train2.json"), "w"), indent=1)


def drive2(track, mode, seconds=30, B=2, seed=1):
    rig = Rig(track, B=B, seed=seed)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    for _ in range(40):
        rig.sense(blind=(mode == "blind"))
    rig.car.crashes[:] = 0; rig.car.progress[:] = 0
    times = [[] for _ in range(B)]
    for k in range(int(seconds / DT)):
        x = rig.sense(blind=(mode == "blind"))
        u = rig.native_u() if mode == "native" else pilot(x)
        for b in np.flatnonzero(rig.car.step(u)):
            times[b].append(k * DT)
    cf = [crash_free_pct(t, seconds) for t in times]
    return dict(mode=mode, seconds=seconds, cars=B, crashes=rig.car.crashes.tolist(),
                crashes_per_min=float(rig.car.crashes.mean() * 60 / seconds),
                crash_free_pct=float(np.mean(cf)), laps=(rig.car.progress / track.P).round(2).tolist())


def test2(seconds=20):
    res = []
    for name, tr in [("unseen bean track", Track(kidney())), ("training oval", Track(stadium()))]:
        for mode in ("trained", "native", "blind"):
            r = drive2(tr, mode, seconds); r["track"] = name; res.append(r)
            log(json.dumps(r))
            json.dump(res, open(os.path.join(OUTD, "test.json"), "w"), indent=1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("train2", "test2"):
    {"train2": train2, "test2": test2}[sys.argv[1]]()
