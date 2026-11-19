"""Experiment 2: can a real fruit-fly brain play Pong?

The fly rides on the paddle at the near end of a court and looks up the court with its real compound eyes. A black
ball bounces off the side walls and the far wall. The paddle's sideways velocity is read from ~11k of the fly's
neurons (visual projection + descending) by a linear readout trained exactly like the car (DAgger imitation of a
coach that tracks the ball). The 167,106-neuron connectome is never edited.

    python exp_pong.py train
    python exp_pong.py test       # trained fly vs blind fly vs no-paddle-motion, hit rate
    python exp_pong.py video      # single fly
    python exp_pong.py duel       # fly vs fly (two independent brains)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flylab.core import (OUT, Eye, BrainPanel, Video, brain, log, paste, resize, sparkline, text, yaw_R, to_rgb8,
                         World, Sphere, Plane, Box, add_material, Timer)
from exp_car import Pilot, visual_cells

COLUMNAR = r"^(Tm\d+[a-z]?|TmY\d+[a-z]?|Mi\d+|T2a?|T3)$"


class Features:
    """Visual-projection + descending LIF rates, plus retinotopic optic-lobe columnar cells (Tm, TmY, Mi, T2/T3):
    one cell per facet column, so they keep *where* the ball is in the visual field."""

    def __init__(self, fb):
        self.fb = fb
        self.cells = torch.as_tensor(visual_cells(fb.c), device=fb.device)
        t = fb.c.neurons.type.fillna("").to_numpy().astype(str)[fb.optic.rate_idx]
        import re
        sel = np.flatnonzero([bool(re.match(COLUMNAR, x)) for x in t])
        # each columnar cell -> its facet column -> viewing direction -> (type, azimuth bin, elevation bin)
        nrn = fb.c.neurons.iloc[fb.optic.rate_idx[sel]]
        r = fb.retina
        col = {(s_, int(a), int(b)): i for i, (s_, (a, b)) in enumerate(zip(r.col_side, r.col_hex))}
        az = np.full(len(sel), np.nan); el = np.full(len(sel), np.nan)
        for i, (s_, a, b) in enumerate(zip(nrn.hex_side.to_numpy(), nrn.hex1.to_numpy(), nrn.hex2.to_numpy())):
            if isinstance(s_, str) and np.isfinite(a) and np.isfinite(b) and (s_, int(a), int(b)) in col:
                az[i], el[i] = r.col_az_el[col[(s_, int(a), int(b))]]
        ok = np.isfinite(az)
        sel, az, el, tt = sel[ok], az[ok], el[ok], t[sel][ok]
        types = sorted(set(tt)); ti = np.array([types.index(x) for x in tt])
        ab = np.clip(((az + 120) / 10).astype(int), 0, 23); eb = np.clip(((el + 80) / 40).astype(int), 0, 3)
        key = (ti * 24 + ab) * 4 + eb
        uniq, inv = np.unique(key, return_inverse=True)
        self.opt = torch.as_tensor(sel, device=fb.device)
        self.bin = torch.as_tensor(inv, device=fb.device)
        self.nbin = len(uniq)
        self.cnt = torch.bincount(self.bin, minlength=self.nbin).float().clamp(min=1)
        self.n = len(self.cells) + self.nbin
        self.n_columnar = int(len(sel))

    def __call__(self):
        a = self.fb.brain.rate[:, self.cells].float()
        o = self.fb.optic.rates()[:, self.opt].float()
        b = torch.zeros(o.shape[0], self.nbin, device=o.device).index_add_(1, self.bin, o) / self.cnt * 40.0
        return torch.cat([a, b], 1).cpu().numpy()

OUTD = os.path.join(OUT, "pong")
LEN, HALF_W = 0.4, 0.2           # court length (x) and half width (y), metres
BALL_R, BALL_V = 0.06, 0.28
PAD_HALF, PAD_V = 0.015, 0.45
EYE_H = 0.012
DT = 0.01


def court_world(ball_xy=(0.25, 0.0), two=False):
    add_material("court", (0.03, 0.07, 0.09, 0.10), pattern=1, refl2=(0.015, 0.035, 0.05, 0.06), scale=0.05)
    add_material("rail", (0.8, 1.0, 1.0, 1.0), pattern=2, refl2=(0.2, 0.3, 0.3, 0.3), scale=0.04)
    add_material("ball", (0.01, 0.015, 0.015, 0.015))
    w = World(detail=0.5, light_pos=(LEN / 2, 0, 1.2), light_color=(1.2, 2.2, 2.2, 2.2), ambient=(0.3, 0.35, 0.35, 0.35))
    w.planes.append(Plane((0, 0, 0), (0, 0, 1), "court"))
    w.planes.append(Plane((0, 0, 1.6), (0, 0, -1), "ceiling"))
    for a, n in [((2.5, 0, 0), (-1, 0, 0)), ((-2.5, 0, 0), (1, 0, 0)), ((0, 2.5, 0), (0, -1, 0)), ((0, -2.5, 0), (0, 1, 0))]:
        w.planes.append(Plane(a, n, "wall"))
    h = 0.05
    w.boxes.append(Box((-0.1, HALF_W, 0), (LEN + 0.05, HALF_W + 0.02, h), "rail"))
    w.boxes.append(Box((-0.1, -HALF_W - 0.02, 0), (LEN + 0.05, -HALF_W, h), "rail"))
    if not two:
        w.boxes.append(Box((LEN, -HALF_W, 0), (LEN + 0.02, HALF_W, h), "rail"))
    w.spheres.append(Sphere((ball_xy[0], ball_xy[1], BALL_R), (BALL_R,) * 3, "ball"))
    return w


class Game:
    """B independent single-fly courts (paddle at x=0) or, with two=True, one court with paddles at both ends."""

    def __init__(self, B, rng, two=False):
        self.B, self.rng, self.two = B, rng, two
        self.ball = np.zeros((B, 2)); self.vel = np.zeros((B, 2))
        self.pad = np.zeros(B)                     # paddle y (near end); in duel mode row 1 is the far paddle
        self.hits = np.zeros(B, int); self.misses = np.zeros(B, int)
        for b in range(B):
            self.serve(b)

    def serve(self, b, toward_near=True):
        ang = self.rng.uniform(-0.6, 0.6)
        self.ball[b] = (LEN * (0.85 if toward_near else 0.15), self.rng.uniform(-0.12, 0.12))
        d = -1 if toward_near else 1
        self.vel[b] = BALL_V * np.array([d * np.cos(ang), np.sin(ang)])

    def intercept(self, b, x_line=0.0):
        """Ball y where it will cross x_line, with side-wall reflections folded."""
        bx, by = self.ball[b]; vx, vy = self.vel[b]
        if (x_line - bx) * vx <= 0:
            return 0.0
        t = (x_line - bx) / vx
        y = by + vy * t
        L = HALF_W - BALL_R
        y = (y + L) % (4 * L)
        return (y if y < 2 * L else 4 * L - y) - L

    def offset(self):
        return self.ball[:, 1] - self.pad

    def coach(self):
        """Move towards where the ball is now (a purely visual target: its bearing from the fly)."""
        return np.clip((self.ball[:, 1] - self.pad) / 0.03, -1, 1)

    def step(self, u):
        self.pad = np.clip(self.pad + np.clip(u, -1, 1) * PAD_V * DT, -HALF_W + PAD_HALF, HALF_W - PAD_HALF)
        events = []
        for b in range(self.B):
            self.ball[b] += self.vel[b] * DT
            if abs(self.ball[b, 1]) > HALF_W - BALL_R:
                self.vel[b, 1] *= -1
                self.ball[b, 1] = np.sign(self.ball[b, 1]) * (HALF_W - BALL_R)
            if self.ball[b, 0] > LEN - BALL_R and not self.two:
                self.vel[b, 0] *= -1; self.ball[b, 0] = LEN - BALL_R
            if self.ball[b, 0] < BALL_R + 0.005 and self.vel[b, 0] < 0:
                if abs(self.ball[b, 1] - self.pad[b]) < PAD_HALF + BALL_R:
                    self.hits[b] += 1
                    off = (self.ball[b, 1] - self.pad[b]) / (PAD_HALF + BALL_R)
                    ang = 0.7 * off + self.rng.uniform(-0.2, 0.2)
                    self.vel[b] = BALL_V * np.array([np.cos(ang), np.sin(ang)])
                    events.append((b, "hit"))
                else:
                    self.misses[b] += 1; self.serve(b); events.append((b, "miss"))
        return events

    def eyes(self):
        eye = np.c_[np.zeros(self.B), self.pad, np.full(self.B, EYE_H)]
        return eye, yaw_R(np.zeros(self.B))


class Rig:
    def __init__(self, B=4, seed=0):
        self.fb = brain(B, seed=seed)
        self.eye = Eye(self.fb)
        self.feat = Features(self.fb)
        self.cells = range(self.feat.n)
        self.game = Game(B, np.random.default_rng(seed))
        self.worlds = [court_world() for _ in range(B)]

    def sense(self, blind=False):
        rads = []
        eye, R = self.game.eyes()
        for b in range(self.game.B):
            w = self.worlds[b]
            w.move_sphere(len(w.spheres) - 1, (self.game.ball[b, 0], self.game.ball[b, 1], BALL_R))
            rads.append(self.eye.look(w, eye[b:b + 1], R[b:b + 1]))
        rad = torch.cat(rads)
        if blind:
            rad = torch.zeros_like(rad)
        self.rad = rad
        self.fb.vision(rad); self.fb.step(10.0)
        return self.feat()


def train(rounds=4, frames=1500, B=4):
    os.makedirs(OUTD, exist_ok=True)
    rig = Rig(B)
    T, Xs, Ys, pilot, report = Timer(), [], [], None, []
    rng = np.random.default_rng(0)
    for r in range(rounds):
        beta = [1.0, 0.5, 0.25, 0.0][min(r, 3)]
        rig.game.hits[:] = 0; rig.game.misses[:] = 0
        drift = np.zeros(B)
        for k in range(40 + frames):
            x = rig.sense()
            y = rig.game.offset()                                   # label: where the ball is relative to the paddle
            if k >= 40:
                Xs.append(x.astype(np.float16)); Ys.append(y)
            # exploration: a wandering paddle (smooth random drift) mixed with the coach, so the fly sees the
            # ball at every bearing (a perfect coach keeps the offset ~0 and there is nothing to learn)
            drift = 0.97 * drift + 0.25 * rng.standard_normal(B)
            if pilot is None or rng.random() < beta:
                u = 0.5 * rig.game.coach() + drift
            else:
                u = steer(pilot(x)) + 0.3 * drift
            rig.game.step(u)
        X = np.concatenate(Xs).astype(np.float32); Y = np.concatenate(Ys)
        pilot, r2 = Pilot.fit(X, Y)
        pilot.save(os.path.join(OUTD, "readout.npz"))
        h, m = int(rig.game.hits.sum()), int(rig.game.misses.sum())
        report.append(dict(round=r, beta=beta, cv_r2=r2, hits=h, misses=m, wall_s=T()))
        log(f"round {r}: beta {beta}, hits {h} misses {m}, CV R2 {r2:.3f} ({T():.0f}s)")
    json.dump(report, open(os.path.join(OUTD, "train.json"), "w"), indent=1)


def steer(offset):
    """Paddle command from a decoded ball offset (m): the same proportional rule the coach uses."""
    return np.clip(offset / 0.03, -1, 1)


def play(mode, seconds=40, B=2, seed=5):
    rig = Rig(B, seed)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    for _ in range(40):
        rig.sense(mode.startswith("blind"))
    rig.game.hits[:] = 0; rig.game.misses[:] = 0
    for k in range(int(seconds / DT)):
        x = rig.sense(mode.startswith("blind"))
        u = np.zeros(B) if mode == "frozen paddle" else steer(pilot(x))
        rig.game.step(u)
    h, m = rig.game.hits.sum(), rig.game.misses.sum()
    return dict(mode=mode, seconds=seconds, hits=int(h), misses=int(m), hit_rate=float(h / max(h + m, 1)))


def test():
    res = [play(m) for m in ("trained fly", "blind fly", "frozen paddle")]
    for r in res:
        log(json.dumps(r))
    json.dump(res, open(os.path.join(OUTD, "test.json"), "w"), indent=1)


# --------------------------------------------------------------------------------------------- drawing
def court_img(ball, pads, W=300, H=420, two=False, trail=()):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (14, 16, 22)); dr = ImageDraw.Draw(img)
    s = min((W - 30) / (2 * HALF_W), (H - 30) / LEN)
    def P(x, y): return (W / 2 - y * s, H - 15 - x * s)
    dr.rectangle([P(LEN, HALF_W), P(0, -HALF_W)], outline=(230, 230, 235), width=2)
    for i in range(1, 12):
        dr.line([P(LEN / 2, -HALF_W + i * 2 * HALF_W / 12), P(LEN / 2, -HALF_W + (i + 0.5) * 2 * HALF_W / 12)], fill=(60, 60, 70), width=1)
    if len(trail) > 1:
        dr.line([P(*p) for p in trail], fill=(90, 90, 110), width=1)
    for x, y, col in pads:
        a, b = P(x, y + PAD_HALF), P(x, y - PAD_HALF)
        dr.rectangle([a[0], a[1] - 4, b[0], b[1] + 4], fill=col)
    x, y = P(*ball); r = BALL_R * s
    dr.ellipse([x - r, y - r, x + r, y + r], fill=(250, 250, 250))
    return np.asarray(img).copy()


def video(seconds=24, out="pong_fly.mp4"):
    rig = Rig(1, seed=9)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    panel = BrainPanel(rig.fb, 420, 300)
    vid = Video(os.path.join(OUTD, out))
    for _ in range(40):
        rig.sense()
    rig.game.hits[:] = 0; rig.game.misses[:] = 0
    us, trail = [], []
    stats = json.load(open(os.path.join(OUTD, "test.json"))) if os.path.exists(os.path.join(OUTD, "test.json")) else []
    for k in range(int(seconds / DT)):
        x = rig.sense(); u = steer(pilot(x)); rig.game.step(u)
        us.append(float(np.clip(u[0], -1, 1))); trail.append(rig.game.ball[0].copy()); trail = trail[-60:]
        if k % 3:
            continue
        can = np.zeros((720, 1280, 3), np.uint8); can[:] = (10, 11, 16)
        g = rig.game
        cam = rig.worlds[0].render_camera(np.array([-0.12, g.pad[0], 0.07]), np.array([1, 0, -0.28]), np.array([0, 0, 1.0]), 420, 236, 80)
        paste(can, to_rgb8(cam, exposure=1.4), 20, 70)
        paste(can, resize(rig.eye.picture(rig.rad[0].cpu().numpy()), 420, 194), 20, 360)
        paste(can, court_img(g.ball[0], [(0, g.pad[0], (255, 190, 60))], 300, 420, trail=trail), 470, 70)
        paste(can, panel.draw(0), 830, 70)
        paste(can, sparkline(us[-300:], 360, 70, lo=-1, hi=1), 830, 560)
        text(can, (20, 18), "A real fruit-fly brain playing Pong", 30, bold=True)
        text(can, (20, 50), "camera behind the paddle", 14, (150, 150, 165))
        text(can, (20, 340), "the fly's compound-eye view (it rides on the paddle)", 14, (150, 150, 165))
        text(can, (470, 50), f"hits {g.hits[0]}   misses {g.misses[0]}", 18, (255, 190, 60))
        text(can, (830, 50), "167,106 MaleCNS neurons (glow = firing)", 14, (150, 150, 165))
        text(can, (830, 390), "brain wiring: frozen, never edited", 18, (255, 190, 60))
        text(can, (830, 418), "learned: a linear paddle readout", 18, (255, 190, 60))
        text(can, (830, 446), f"from {len(rig.cells):,} visual neurons (incl. retinotopic optic lobe)", 16)
        for i, r in enumerate(stats):
            text(can, (830, 476 + 22 * i), f"{r['mode']:>13}: {100 * r['hit_rate']:4.0f}% returns ({r['hits']}/{r['hits'] + r['misses']})", 15)
        text(can, (830, 540), "paddle command from neurons, last 9 s", 14, (150, 150, 165))
        vid.add(can)
    vid.close(); log("wrote", vid.path)


def duel(seconds=30, out="pong_duel.mp4"):
    """Two flies, two independent copies of the connectome (batch rows), one court, paddles at both ends."""
    fb = brain(2, seed=21)
    eye = Eye(fb)
    feat = Features(fb)
    pilot = Pilot(np.load(os.path.join(OUTD, "readout.npz")))
    w = court_world(two=True)
    rng = np.random.default_rng(4)
    ball = np.array([LEN / 2, 0.0]); ang = rng.uniform(-0.5, 0.5)
    vel = BALL_V * np.array([-np.cos(ang), np.sin(ang)])
    pads = np.zeros(2); score = [0, 0]; rally, best_rally = 0, 0
    panel = BrainPanel(fb, 300, 214)
    vid = Video(os.path.join(OUTD, out))
    trail = []
    for k in range(int(seconds / DT) + 40):
        w.move_sphere(len(w.spheres) - 1, (ball[0], ball[1], BALL_R))
        # fly A at x=0 looking +x, fly B at x=LEN looking -x (its own left/right mirrored, as a real fly would be)
        eyes = np.array([[0, pads[0], EYE_H], [LEN, pads[1], EYE_H]])
        R = yaw_R(np.array([0.0, np.pi]))
        rad = eye.look(w, eyes, R)
        fb.vision(rad); fb.step(10.0)
        x = feat()
        u = steer(pilot(x))
        pads[0] = np.clip(pads[0] + np.clip(u[0], -1, 1) * PAD_V * DT, -HALF_W + PAD_HALF, HALF_W - PAD_HALF)
        pads[1] = np.clip(pads[1] - np.clip(u[1], -1, 1) * PAD_V * DT, -HALF_W + PAD_HALF, HALF_W - PAD_HALF)  # fly B's +u = its left = world -y
        if k < 40:
            continue
        ball += vel * DT
        if abs(ball[1]) > HALF_W - BALL_R:
            vel[1] *= -1; ball[1] = np.sign(ball[1]) * (HALF_W - BALL_R)
        for side, xline, sgn in ((0, BALL_R + 0.005, -1), (1, LEN - BALL_R - 0.005, 1)):
            if (ball[0] - xline) * sgn > 0 and vel[0] * sgn > 0:
                if abs(ball[1] - pads[side]) < PAD_HALF + BALL_R:
                    off = (ball[1] - pads[side]) / (PAD_HALF + BALL_R)
                    a = 0.7 * off + rng.uniform(-0.2, 0.2)
                    vel = BALL_V * np.array([-sgn * np.cos(a), np.sin(a)]); rally += 1
                    best_rally = max(best_rally, rally)
                else:
                    score[1 - side] += 1; rally = 0
                    ball = np.array([LEN / 2, rng.uniform(-0.1, 0.1)]); a = rng.uniform(-0.5, 0.5)
                    vel = BALL_V * np.array([sgn * np.cos(a), np.sin(a)])
        trail.append(ball.copy()); trail = trail[-60:]
        if k % 3:
            continue
        can = np.zeros((720, 1280, 3), np.uint8); can[:] = (10, 11, 16)
        cam = w.render_camera(np.array([LEN / 2, -0.62, 0.42]), np.array([0, 0.62, -0.42]), np.array([0, 0, 1.0]), 520, 300, 55)
        paste(can, to_rgb8(cam, exposure=1.4), 20, 70)
        paste(can, court_img(ball, [(0, pads[0], (255, 190, 60)), (LEN, pads[1], (90, 200, 255))], 300, 420, two=True, trail=trail), 560, 70)
        paste(can, panel.draw(0), 900, 70); paste(can, panel.draw(1), 900, 330)
        paste(can, resize(eye.picture(rad[0].cpu().numpy()), 250, 116), 20, 400)
        paste(can, resize(eye.picture(rad[1].cpu().numpy()), 250, 116), 290, 400)
        text(can, (20, 18), "Fly vs Fly Pong: two copies of a real fruit-fly brain", 28, bold=True)
        text(can, (20, 380), "orange fly's eyes", 14, (255, 190, 60)); text(can, (290, 380), "blue fly's eyes", 14, (90, 200, 255))
        text(can, (560, 500), f"orange {score[0]}  :  {score[1]} blue", 26, bold=True)
        text(can, (560, 540), f"rally {rally}   longest rally {best_rally}", 16)
        text(can, (900, 50), "orange fly's brain (167k neurons)", 14, (255, 190, 60))
        text(can, (900, 310), "blue fly's brain (independent copy)", 14, (90, 200, 255))
        text(can, (20, 560), "Each fly sees only through its own compound eyes.", 16)
        text(can, (20, 584), "Same connectome, same learned paddle readout, separate brain states.", 16)
        text(can, (20, 608), "Simulated on an Apple M2 laptop.", 16, (150, 150, 165))
        vid.add(can)
    vid.close()
    json.dump(dict(score=score, best_rally=best_rally, seconds=seconds), open(os.path.join(OUTD, "duel.json"), "w"))
    log("duel", score, "best rally", best_rally, "wrote", vid.path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"train": train, "test": test, "video": video, "duel": duel}[cmd]()
