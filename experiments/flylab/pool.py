"""Parallel MuJoCo flies: worker processes each own a few physfly.Body instances and step them in lockstep."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from .physfly import LEGS, JOINTS


def tilt_gravity(t, prof):
    """Platform roll (deg) at time t for a profile dict -> gravity vector in the platform frame."""
    amp, f, ph, ramp = prof["amp"], prof["freq"], prof["phase"], prof.get("ramp", 0.5)
    th = np.deg2rad(amp * min(1.0, t / ramp) * np.sin(2 * np.pi * f * t + ph))
    return np.array([0.0, 981.0 * np.sin(th), -981.0 * np.cos(th)])


def roll_deg(t, prof):
    return prof["amp"] * min(1.0, t / prof.get("ramp", 0.5)) * np.sin(2 * np.pi * prof["freq"] * t + prof["phase"])


def _worker(conn, genes_list, adhesion):
    from .physfly import Body
    bodies = [Body(g, adhesion=adhesion) for g in genes_list]
    t = 0.0
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        if cmd == "reset":
            for b in bodies:
                b.reset()
            t = 0.0
        elif cmd == "step":
            offsets, prof, t_set = arg
            if t_set is not None:
                t = t_set                     # the controller's clock is the truth (settle frames must not shift it)
            for b, off in zip(bodies, offsets):
                if off is not None:
                    b.set_targets({(li, j): off[li, ji] for li in range(6) for ji, j in enumerate(JOINTS)})
                b.step(10.0, lambda tt: tilt_gravity(tt, prof), t)
            t += 0.01
        elif cmd == "qpos":
            conn.send([b.d.qpos.copy() for b in bodies]); continue
        conn.send((np.array([b.leg_loads() for b in bodies]), np.array([b.flex() for b in bodies]),
                   np.array([b.tilt_vs_gravity() for b in bodies]), np.array([b.fallen for b in bodies]),
                   np.array([float(b.d.qvel[3]) for b in bodies]), np.array([b.up() for b in bodies])))


class FlyPool:
    def __init__(self, genes_list, workers=8, adhesion=1.0):
        ctx = mp.get_context("spawn")
        self.n = len(genes_list)
        chunks = np.array_split(np.arange(self.n), min(workers, self.n))
        self.chunks = [c for c in chunks if len(c)]
        self.conns, self.procs = [], []
        for ch in self.chunks:
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, [genes_list[i] for i in ch], adhesion), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)

    def _gather(self):
        parts = [c.recv() for c in self.conns]
        return tuple(np.concatenate([p[k] for p in parts]) for k in range(6))

    def reset(self):
        for c in self.conns:
            c.send(("reset", None))
        return self._gather()

    def step(self, offsets, prof, t=None):
        """offsets (n, 6, len(JOINTS)) radians or None (passive); t = episode time of this step (s)."""
        for c, ch in zip(self.conns, self.chunks):
            c.send(("step", ([None] * len(ch) if offsets is None else [offsets[i] for i in ch], prof, t)))
        return self._gather()

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
