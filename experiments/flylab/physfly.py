"""The physical fly: flybody's anatomically detailed MuJoCo body driven by the real MaleCNS ventral nerve cord.

Brain side (VNCBrain): every VNC neuron, every descending and ascending neuron and every leg proprioceptor of the
MaleCNS connectome (~19k neurons, real wiring, LIF as in flybrain). Per leg (T1/T2/T3 x L/R):
  * sensors  = the leg's proprioceptive afferents, chosen by their entry nerve (ProLN / MesoLN / MetaLN) and side
               (laterality of their outputs, flybrain's rule): campaniform sensilla <- load on that leg,
               chordotonal organ / hair plates <- that leg's femur-tibia angle (half the cells flexion-tuned, half
               extension-tuned: a modelling assumption, the connectome does not say which is which)
  * muscles  = the leg's motor neurons grouped by the muscle they innervate (MaleCNS types: Ti flexor MN, Tr
               extensor MN, Ta depressor MN ...). Each muscle pulls one flybody joint towards flexion or extension.
Body side (Body): flybody's fruitfly.xml (67 bodies, position servos on every leg joint, adhesive claws) with
optional morphology genes that rescale each leg pair and the abdomen.

A platform that tilts/shakes is simulated exactly by rotating/adding to gravity in the platform frame (equivalence
principle), so the arena itself never moves.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import torch

ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flybody", "flybody", "fruitfly", "assets"))
LEGS = [("T1", "left"), ("T1", "right"), ("T2", "left"), ("T2", "right"), ("T3", "left"), ("T3", "right")]
NERVE = {"T1": ("ProLN", "ProAN", "VProN", "DProN", "ProCN"), "T2": ("MesoLN",), "T3": ("MetaLN",)}
MN_LEG = {"T1": "fl", "T2": "ml", "T3": "hl"}
# muscle (MaleCNS MN type) -> (flybody joint stem, direction +1 extension / -1 flexion)
MUSCLES = {
    "Ti extensor MN": ("tibia", +1), "Ti flexor MN": ("tibia", -1), "Acc. ti flexor MN": ("tibia", -1),
    "ltm1-tibia MN": ("tibia", -1),
    "Tr extensor MN": ("femur", +1), "Tr flexor MN": ("femur", -1), "Acc. tr flexor MN": ("femur", -1),
    "Sternotrochanter MN": ("femur", +1), "Tergotr. MN": ("femur", +1), "ltm2-femur MN": ("femur", -1),
    "Fe reductor MN": ("femur_twist", +1),
    "Tergopleural/Pleural promotor MN": ("coxa", +1), "Pleural remotor/abductor MN": ("coxa", -1),
    "Sternal anterior rotator MN": ("coxa_twist", +1), "Sternal posterior rotator MN": ("coxa_twist", -1),
    "Sternal adductor MN": ("coxa_abduct", -1),
    "Ta depressor MN": ("tarsus", +1), "Ta levator MN": ("tarsus", -1), "ltm MN": ("tarsus", -1),
}
JOINTS = ["coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus"]


# ----------------------------------------------------------------------------------------------------- brain
def vnc_subset(c):
    n = c.neurons
    sc = n.superclass.fillna("").astype(str)
    from flybrain import regions
    keep = (sc.str.startswith("vnc_") | sc.str.contains("ascending") | sc.str.contains("descending")
            | (n["class"] == "mechanosensory_proprioceptive")).to_numpy().copy()
    keep |= regions.labels(c) == "mechanosensory"        # antennae: Johnston's organ (gravity/wind) + AMMC/WED
    return c.subset(np.flatnonzero(keep))


class VNCBrain:
    def __init__(self, c_full, batch=1, seed=0):
        from .core import brain
        from flybrain.senses import laterality
        self.c = vnc_subset(c_full)
        self.fb = brain(batch, c=self.c, seed=seed)
        c, n = self.fb.c, self.fb.c.neurons
        self.B, self.dev = batch, self.fb.device
        typ = n.type.fillna("").to_numpy().astype(str)
        sub = n.subclass.fillna("").to_numpy().astype(str)
        sup = n.superclass.fillna("").to_numpy().astype(str)
        nerve = n.entryNerve.fillna("").to_numpy().astype(str)
        soma = n.somaSide.fillna("").to_numpy().astype(str)
        pro = (n["class"] == "mechanosensory_proprioceptive").to_numpy()
        # afferent sides from output laterality on the full reference graph
        ref = c.reference
        rs = ref.neurons.somaSide.to_numpy()
        tL, tR = np.flatnonzero(rs == "L"), np.flatnonzero(rs == "R")
        aff = np.flatnonzero(pro)
        lat = laterality(ref, ref.index_of(n.bodyId.to_numpy()[aff]), tL, tR)
        aside = np.full(c.n, "", object); aside[aff] = np.where(lat > 0.2, "L", np.where(lat < -0.2, "R", ""))
        self.sens, self.mns, self.legs = [], [], LEGS
        for seg, sd in LEGS:
            S = "L" if sd == "left" else "R"
            leg_aff = pro & np.isin(nerve, NERVE[seg]) & (aside == S)
            load = np.flatnonzero(leg_aff & (sub == "campaniform sensilla"))
            angle = np.flatnonzero(leg_aff & np.isin(sub, ["chordotonal organ", "leg", "hair plate"]))
            bid = n.bodyId.to_numpy()
            self.sens.append(dict(load=load, flex=angle[bid[angle] % 2 == 0], ext=angle[bid[angle] % 2 == 1]))
            groups = {}
            for m in MUSCLES:
                idx = np.flatnonzero((sup == "vnc_motor") & (sub == MN_LEG[seg]) & (soma == S) & (typ == m))
                if len(idx):
                    groups[m] = idx
            self.mns.append(groups)
        self.muscles = list(MUSCLES)
        # flat gather tensor for MN means: (leg, muscle) slots
        flat, seg_id, self.slot_n = [], [], np.zeros((6, len(self.muscles)), np.float32)
        for li, g in enumerate(self.mns):
            for mi, m in enumerate(self.muscles):
                if m in g:
                    flat.append(g[m]); seg_id.append(np.full(len(g[m]), li * len(self.muscles) + mi))
                    self.slot_n[li, mi] = len(g[m])
        self.mn_idx = torch.as_tensor(np.concatenate(flat), device=self.dev)
        self.mn_seg = torch.as_tensor(np.concatenate(seg_id), device=self.dev)
        self.sens_idx = np.concatenate([np.concatenate([s["load"], s["flex"], s["ext"]]) for s in self.sens])
        self.sens_layout = [(len(s["load"]), len(s["flex"]), len(s["ext"])) for s in self.sens]
        self.dn = c.select(superclass="descending_neuron")

    def summary(self):
        lines = [f"VNC brain: {self.c.n:,} neurons ({len(self.dn)} descending)"]
        for (seg, sd), s, g in zip(LEGS, self.sens, self.mns):
            lines.append(f"  {seg} {sd:5s}: load {len(s['load']):3d}, angle {len(s['flex']) + len(s['ext']):3d} afferents; "
                         f"{sum(len(v) for v in g.values()):3d} MNs in {len(g)} muscles")
        return "\n".join(lines)

    def sense(self, load, flex, tonic_dn=0.0):
        """load (B,6) in body weights, flex (B,6) femur-tibia flexion relative to rest (rad, + = more flexed)."""
        rates = []
        for li, (nl, nf, ne) in enumerate(self.sens_layout):
            l = np.clip(10 + 120 * load[:, li], 0, 200)
            f = np.clip(10 + 150 * np.maximum(flex[:, li], 0), 0, 200)
            e = np.clip(10 + 150 * np.maximum(-flex[:, li], 0), 0, 200)
            rates += [np.repeat(l[:, None], nl, 1), np.repeat(f[:, None], nf, 1), np.repeat(e[:, None], ne, 1)]
        hz = np.concatenate(rates, 1).astype(np.float32)
        self.fb._input("legsense", self.sens_idx, hz)
        if tonic_dn:
            self.fb._input("dn_tone", self.dn, np.full((self.B, len(self.dn)), tonic_dn, np.float32))

    def step(self, ms=10.0):
        self.fb.step(ms)

    def muscle_rates(self):
        """(B, 6, n_muscles) mean MN rate (Hz) per leg and muscle (0 where the leg has no such MN)."""
        r = self.fb.brain.rate[:, self.mn_idx].float()
        out = torch.zeros(r.shape[0], 6 * len(self.muscles), device=r.device).index_add_(1, self.mn_seg, r)
        out = out.cpu().numpy().reshape(-1, 6, len(self.muscles))
        return out / np.maximum(self.slot_n, 1)[None]


# ------------------------------------------------------------------------------------------------------ body
DEFAULT_GENES = dict(front=1.0, mid=1.0, hind=1.0, abdomen=1.0, stance=0.0)
GENE_RANGE = dict(front=(0.6, 1.8), mid=(0.6, 1.8), hind=(0.6, 1.8), abdomen=(0.4, 2.5), stance=(-0.35, 0.35))


def _scale_attr(el, key, s):
    if key in el.attrib:
        el.set(key, " ".join(f"{float(v) * s:.6g}" for v in el.get(key).split()))


def build_xml(genes: dict | None = None) -> str:
    g = dict(DEFAULT_GENES, **(genes or {}))
    tree = ET.parse(os.path.join(ASSETS, "fruitfly.xml"))
    root = tree.getroot()
    root.find("compiler").set("meshdir", ASSETS)
    meshes = {m.get("name"): m for m in root.iter("mesh") if m.get("name")}
    pair = {"T1": g["front"], "T2": g["mid"], "T3": g["hind"]}
    for body in root.iter("body"):
        mm = re.match(r"coxa_(T[123])_(left|right)$", body.get("name", ""))
        if not mm:
            continue
        s = pair[mm.group(1)]
        for el in body.iter():
            if el is not body and el.tag == "body":
                _scale_attr(el, "pos", s)
            if el.tag in ("geom", "site"):
                _scale_attr(el, "pos", s); _scale_attr(el, "fromto", s); _scale_attr(el, "size", s)
                if el.get("mesh") in meshes:
                    mesh = meshes[el.get("mesh")]
                    if "scale" not in mesh.attrib:
                        mesh.set("scale", "0.1 0.1 0.1")             # the <default><mesh scale> of fruitfly.xml
                    _scale_attr(mesh, "scale", s)
    for el in root.iter("geom"):
        if el.get("name", "").startswith("abdomen") and "collision" not in el.get("name", ""):
            el.set("density", f"{0.478 * g['abdomen']:.5g}")
    body_xml = ET.tostring(root, encoding="unicode")
    # wrap: floor + light, the fly included inline
    inner = body_xml.split(">", 1)[1].rsplit("</mujoco>", 1)[0]
    return f"""<mujoco model="fly_on_platform">{inner}
  <asset>
    <texture name="plat" type="2d" builtin="checker" rgb1=".22 .24 .30" rgb2=".32 .34 .42" width="200" height="200"/>
    <material name="plat" texture="plat" texrepeat="6 6" reflectance=".1"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="platform" type="box" size="0.9 0.9 0.02" pos="0 0 -0.152" material="plat" solref="0.0002 1"/>
    <camera name="cam_back" pos="-1.1 0 0.05" xyaxes="0 -1 0 0.1 0 1" fovy="22"/>
    <camera name="cam_side" pos="0 -1.3 0.12" xyaxes="1 0 0 0 0.1 1" fovy="22"/>
  </worldbody>
</mujoco>"""


class Body:
    """One flybody in MuJoCo with a gravity-tilt platform. Leg joint targets come from the brain."""

    def __init__(self, genes=None, timestep=0.0002, adhesion=1.0):
        import mujoco
        self.mj = mujoco
        self.genes = dict(DEFAULT_GENES, **(genes or {}))
        self.m = mujoco.MjModel.from_xml_string(build_xml(self.genes))
        self.m.opt.timestep = timestep
        self.m.opt.noslip_iterations = 0
        self.d = mujoco.MjData(self.m)
        name = lambda t, i: mujoco.mj_id2name(self.m, t, i)
        A = mujoco.mjtObj.mjOBJ_ACTUATOR
        self.claws = [i for i in range(self.m.nu) if "claw" in name(A, i)]
        self.act = {}                          # (leg index, joint stem) -> actuator id
        for li, (seg, sd) in enumerate(LEGS):
            for j in JOINTS:
                i = mujoco.mj_name2id(self.m, A, f"{j}_{seg}_{sd}")
                if i >= 0:
                    self.act[(li, j)] = i
        self.claw_geoms = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"tarsal_claw_{seg}_{sd}_collision") for seg, sd in LEGS]
        self.leg_bodies = []
        for seg, sd in LEGS:
            ids = set()
            for b in range(self.m.nbody):
                nm = name(mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if nm.endswith(f"{seg}_{sd}"):
                    ids.add(b)
            self.leg_bodies.append(ids)
        self.thorax = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "thorax")
        self.weight = float(self.m.body_subtreemass[1] * 981)
        self.adhesion = adhesion
        self.reset()

    def reset(self):
        mj = self.mj
        mj.mj_resetDataKeyframe(self.m, self.d, 0)
        # stance gene: spread / narrow the legs at the coxa
        for li in range(6):
            a = self.act.get((li, "coxa_abduct"))
            if a is not None:
                j = self.m.actuator_trnid[a, 0]
                self.d.qpos[self.m.jnt_qposadr[j]] += self.genes["stance"]
        mj.mj_forward(self.m, self.d)
        # drop the body so that its lowest claw sits on the platform top (legs of any length start on the ground)
        plat = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "platform")
        top = self.m.geom_pos[plat][2] + self.m.geom_size[plat][2]
        low = min(self.d.geom_xpos[g][2] - self.m.geom_rbound[g] * 0.5 for g in self.claw_geoms)
        self.d.qpos[2] += top - low + 0.001
        mj.mj_forward(self.m, self.d)
        self.q0 = np.zeros(self.m.nu)
        for i in range(self.m.nu):
            if self.m.actuator_trntype[i] == 0:
                self.q0[i] = self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[i, 0]]]
        self.d.ctrl[:] = self.q0
        self.d.ctrl[self.claws] = self.adhesion
        self.tib0 = np.array([self.tibia(li) for li in range(6)])
        # settle
        for _ in range(int(0.15 / self.m.opt.timestep)):
            mj.mj_step(self.m, self.d)
        self.fallen = False
        self.load0 = np.maximum(self.raw_loads(), 1e-3)

    def tibia(self, li):
        a = self.act[(li, "tibia")]
        return float(self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]]])

    def set_targets(self, offsets: dict):
        """offsets: {(leg, joint stem): radians added to the rest pose}. Clipped to the actuator ranges."""
        ctrl = self.q0.copy()
        for k, v in offsets.items():
            a = self.act.get(k)
            if a is not None:
                ctrl[a] += v
        lo, hi = self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1]
        self.d.ctrl[:] = np.clip(ctrl, lo, hi)
        self.d.ctrl[self.claws] = self.adhesion

    def step(self, ms, gravity_fn, t0):
        n = int(round(ms / 1000 / self.m.opt.timestep))
        for k in range(n):
            self.m.opt.gravity[:] = gravity_fn(t0 + k * self.m.opt.timestep)
            self.mj.mj_step(self.m, self.d)
        if not np.isfinite(self.d.qpos).all():
            self.fallen = True

    def leg_loads(self):
        """Load on each leg relative to its resting load (1 = as when standing level; adhesion pull cancels out)."""
        return self.raw_loads() / self.load0

    def raw_loads(self):
        """Normal contact force on each leg (in body weights, adhesion included)."""
        mj, m, d = self.mj, self.m, self.d
        out = np.zeros(6); f = np.zeros(6)
        for i in range(d.ncon):
            con = d.contact[i]
            b1, b2 = m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]
            for li, ids in enumerate(self.leg_bodies):
                if b1 in ids or b2 in ids:
                    mj.mj_contactForce(m, d, i, f)
                    out[li] += abs(f[0])
                    break
        return out / max(self.weight, 1e-9)

    def flex(self):
        return np.array([self.tibia(li) for li in range(6)]) - self.tib0

    def up(self):
        R = self.d.xmat[self.thorax].reshape(3, 3)
        return R[:, 2]

    def tilt_vs_gravity(self):
        g = -self.m.opt.gravity / np.linalg.norm(self.m.opt.gravity)
        return float(np.degrees(np.arccos(np.clip(self.up() @ g, -1, 1))))
