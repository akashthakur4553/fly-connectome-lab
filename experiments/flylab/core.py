"""Shared plumbing for the fly-connectome-lab experiments.

Everything here sits *around* the real fly brain (FlyBrain = all 167k MaleCNS neurons as leaky integrate-and-fire
units on the real wiring). Nothing in this file edits a synapse: the connectome is frozen. What learns in the
experiments is always a small readout from named neuron populations to an output (a steering wheel, a paddle,
joint targets), exactly like flybrain's own "Train a fly" dino game.

Pieces:
  * brain(...)          build a FlyBrain on the Apple GPU (fast mode: LIF 1 ms, optic lobe 2 ms)
  * Eye                 render any flybrain World through the fly's real compound eye (per-ommatidium rays)
  * Populations         mean firing rate per (cell type, side) for named sets of neurons
  * ES                  evolution strategy whose candidates run as *batch rows of one brain* in parallel
  * Panel helpers       numpy/PIL drawing for the LinkedIn videos (brain map, fly-eye view, plots, text)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

FLYBRAIN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "flybrain"))
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "out"))
os.environ.setdefault("FLYBRAIN_DATA", os.path.join(FLYBRAIN, "data"))
if FLYBRAIN not in sys.path:
    sys.path.insert(0, FLYBRAIN)

from flybrain import FlyBrain, connectome, optic  # noqa: E402
from flybrain.brain import LIFParams  # noqa: E402
from flybrain.world import World, Sphere, Box, Plane, MATERIALS, Material, to_rgb8  # noqa: E402
from flybrain.brainmap import BrainMap  # noqa: E402

FRAME_MS = 10.0
_C = None


def conn():
    """The full MaleCNS connectome (cached after the first call)."""
    global _C
    if _C is None:
        _C = connectome.load(verbose=False)
    return _C


def brain(batch: int = 1, modules=None, seed: int = 0, c=None) -> FlyBrain:
    return FlyBrain(c if c is not None else conn(), modules=modules, batch=batch, seed=seed,
                    lif_params=LIFParams(dt=1.0), optic_params=optic.OpticParams(dt_ms=2.0))


def add_material(name: str, refl, emit=(0, 0, 0, 0), pattern=0, refl2=(0, 0, 0, 0), scale=0.1) -> str:
    MATERIALS[name] = Material(name, tuple(refl), tuple(emit), pattern, tuple(refl2), scale)
    return name


# ----------------------------------------------------------------------------------------------------- the eye
class Eye:
    """The fly's two compound eyes as the real MaleCNS retina defines them (~1,500 columns, 7 rays each)."""

    def __init__(self, fb: FlyBrain):
        dirs, w = fb.retina.ray_directions()
        self.dirs = dirs.astype(np.float32)                         # (n_col, k, 3) body frame: x fwd, y left, z up
        self.w = torch.tensor(w, dtype=torch.float32, device=fb.device)
        self.n_col = dirs.shape[0]
        self.az_el = fb.retina.col_az_el
        self.side = fb.retina.col_side

    def look(self, world: World, eye: np.ndarray, R: np.ndarray) -> torch.Tensor:
        """eye (B,3) world positions, R (B,3,3) columns = body x/y/z axes in world. Returns (B, n_col, 4) radiance."""
        B = len(eye)
        d = np.einsum("ij,bkj->bik", self.dirs.reshape(-1, 3), R).reshape(B, -1, 3)
        o = np.broadcast_to(eye[:, None, :], d.shape)
        rad = world.trace(torch.from_numpy(np.ascontiguousarray(o, np.float32).reshape(-1, 3)),
                          torch.from_numpy(np.ascontiguousarray(d, np.float32).reshape(-1, 3)))
        return (rad.reshape(B, self.n_col, -1, 4) * self.w[None, None, :, None]).sum(2)

    def picture(self, rad_row: np.ndarray, w: int = 260, h: int = 120, exposure: float = 1.6) -> np.ndarray:
        """Unwrap one fly's column radiance into a panoramic image (azimuth across, elevation up), fly false colour
        (UV -> blue, blue -> green, green -> red: what the fly's photoreceptors weight)."""
        az, el = self.az_el[:, 0], self.az_el[:, 1]
        img = np.zeros((h, w, 3), np.float32)
        cnt = np.zeros((h, w, 1), np.float32)
        x = ((-az + 120) / 240 * (w - 1)).astype(int).clip(0, w - 1)      # left eye on the left
        y = ((-el + 80) / 160 * (h - 1)).astype(int).clip(0, h - 1)
        col = np.stack([rad_row[:, 3] * 0.5 + rad_row[:, 2] * 0.6, rad_row[:, 2] * 0.5 + rad_row[:, 1] * 0.6,
                        rad_row[:, 0] * 0.9 + rad_row[:, 1] * 0.3], -1)
        r = 3
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                xx, yy = (x + dx).clip(0, w - 1), (y + dy).clip(0, h - 1)
                np.add.at(img, (yy, xx), col)
                np.add.at(cnt, (yy, xx), 1)
        img = img / np.maximum(cnt, 1)
        img = (np.clip(img * exposure, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
        img[cnt[..., 0] == 0] = (12, 14, 20)
        return img


def yaw_R(heading: np.ndarray) -> np.ndarray:
    """(B,) headings -> (B,3,3) body->world rotation for a level fly."""
    c, s = np.cos(heading), np.sin(heading)
    z, o = np.zeros_like(c), np.ones_like(c)
    return np.stack([np.stack([c, s, z], -1), np.stack([-s, c, z], -1), np.stack([z, z, o], -1)], -1)


# ------------------------------------------------------------------------------------------------ populations
class Populations:
    """Mean firing rate (Hz/20) per (cell type, side) group, read straight off the GPU for every batch row."""

    def __init__(self, fb: FlyBrain, types: list[str], sides="LR"):
        c = fb.c
        found = c.neurons.type.dropna()
        names = []
        for q in types:
            if q.startswith("~"):
                names += sorted(set(found[found.str.contains(q[1:], regex=True)]))
            else:
                names.append(q)
        groups, labels = [], []
        for t in dict.fromkeys(names):
            for s in sides:
                idx = c.select(type=t, somaSide=s)
                if len(idx):
                    groups.append(idx); labels.append(f"{t} {s}")
        if not groups:
            raise ValueError(f"no neurons for {types}")
        self.labels = labels
        self.sizes = np.array([len(g) for g in groups], np.float32)
        flat = np.concatenate(groups)
        seg = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
        self.idx = torch.as_tensor(flat, device=fb.device)
        self.seg = torch.as_tensor(seg, device=fb.device)
        self.n = len(groups)
        self.fb = fb
        self.cells = int(len(flat))

    def read(self, source: str = "rate") -> np.ndarray:
        r = getattr(self.fb.brain, source)[:, self.idx].float()          # (B, cells)
        out = torch.zeros(r.shape[0], self.n, device=r.device).index_add_(1, self.seg, r)
        return (out.cpu().numpy() / self.sizes / 20.0).astype(np.float32)


# ------------------------------------------------------------------------------------------- evolution strategy
class ES:
    """Separable-CMA-lite (antithetic Gaussian ES with rank weights). Each candidate is one batch row of a single
    FlyBrain, so one sparse matmul simulates the whole population: training = many flies living in parallel."""

    def __init__(self, dim: int, pop: int, sigma: float = 0.5, lr: float = 1.0, seed: int = 0, init=None):
        assert pop % 2 == 0
        self.dim, self.pop = dim, pop
        self.mean = np.zeros(dim, np.float32) if init is None else np.asarray(init, np.float32).copy()
        self.sigma = np.full(dim, sigma, np.float32)
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.best, self.best_fit = self.mean.copy(), -np.inf
        self.history = []

    def ask(self, include_mean: bool = True) -> np.ndarray:
        half = self.rng.standard_normal((self.pop // 2, self.dim)).astype(np.float32)
        self._eps = np.concatenate([half, -half])
        cand = self.mean + self._eps * self.sigma
        if include_mean:
            cand[0] = self.mean; self._eps[0] = 0
        return cand

    def tell(self, cand: np.ndarray, fit: np.ndarray) -> None:
        fit = np.asarray(fit, np.float64)
        i = int(np.argmax(fit))
        if fit[i] > self.best_fit:
            self.best_fit, self.best = float(fit[i]), cand[i].copy()
        order = np.argsort(-fit)
        mu = self.pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1)); w /= w.sum()
        elite_eps = self._eps[order[:mu]]
        step = (w[:, None] * elite_eps).sum(0)
        self.mean = (self.mean + self.lr * self.sigma * step).astype(np.float32)
        spread = np.sqrt((w[:, None] * elite_eps ** 2).sum(0))
        self.sigma = np.clip(self.sigma * (0.8 + 0.2 * spread), 0.02, 2.0).astype(np.float32)
        self.history.append((float(np.mean(fit)), float(np.max(fit)), float(fit[0])))


# ------------------------------------------------------------------------------------------------- drawing
_FONT = {}


def font(size=16, bold=False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _FONT:
        cands = ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"]
        for p in ([cands[2]] + cands) if bold else (cands[2:] + cands[:2]):
            try:
                _FONT[key] = ImageFont.truetype(p, size); break
            except Exception:
                continue
        else:
            _FONT[key] = ImageFont.load_default()
    return _FONT[key]


def text(img: np.ndarray, xy, s: str, size=16, color=(235, 235, 240), bold=False, anchor="la") -> np.ndarray:
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text(xy, s, fill=color, font=font(size, bold), anchor=anchor)
    img[:] = np.asarray(im)
    return img


def paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img[..., :3]


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


class BrainPanel:
    """Every located neuron of the CNS drawn dorsal-view; firing neurons glow. Uses flybrain's BrainMap binning."""

    def __init__(self, fb: FlyBrain, width=420, height=300):
        self.bm = BrainMap(fb.c, width=width, dorsal_h=height, lateral_h=10)
        self.fb, self.W, self.H = fb, width, height
        self.idx_t = torch.as_tensor(self.bm.idx, device=fb.device)
        self.glow = np.zeros(width * height, np.float32)
        self.optic_idx = None
        if fb.optic is not None:
            self.optic_idx = torch.as_tensor(fb.optic.rate_idx, device=fb.device)

    def draw(self, row: int = 0, highlight: dict | None = None) -> np.ndarray:
        rate = self.fb.brain.rate[row].float()
        a = (rate / 40.0).clamp(0, 1)
        if self.optic_idx is not None:
            # optic units are graded rates: show deviation from rest
            o = self.fb.optic
            try:
                dr = (o.r[row] - o.r0).abs() if hasattr(o, "r0") else None
            except Exception:
                dr = None
            if dr is not None:
                a[self.optic_idx] = (dr * 2).clamp(0, 1)
        act = a[self.idx_t].cpu().numpy()
        s = np.bincount(self.bm.bin_d, weights=act, minlength=self.W * self.H) / self.bm.norm_d * 3
        self.glow = np.maximum(self.glow * 0.6, s.astype(np.float32))
        g = np.clip(self.glow, 0, 1).reshape(self.H, self.W)
        bg = self.bm.bg_d.astype(np.float32)
        img = np.stack([bg * 0.55 + g * 255, bg * 0.65 + g * 200, bg * 0.9 + g * 90], -1)
        if highlight:
            for (idx, color) in highlight.values():
                m = np.isin(self.bm.idx, idx)
                if m.any():
                    bins = self.bm.bin_d[m]
                    hot = act[m] > 0.05
                    yy, xx = np.divmod(bins[hot], self.W)
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            img[(yy + dy).clip(0, self.H - 1), (xx + dx).clip(0, self.W - 1)] = color
        return np.clip(img, 0, 255).astype(np.uint8)


def sparkline(values, w=300, h=60, color=(90, 200, 255), lo=None, hi=None, bg=(20, 22, 30)) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8); img[:] = bg
    v = np.asarray(values, float)
    if len(v) < 2:
        return img
    lo = v.min() if lo is None else lo; hi = v.max() if hi is None else hi
    if hi <= lo:
        hi = lo + 1
    xs = np.linspace(0, w - 1, len(v)).astype(int)
    ys = ((1 - (v - lo) / (hi - lo)) * (h - 4) + 2).clip(0, h - 1).astype(int)
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).line(list(zip(xs.tolist(), ys.tolist())), fill=tuple(color), width=2)
    return np.asarray(im).copy()


class Video:
    def __init__(self, path: str, fps: int = 30):
        import imageio.v2 as iio
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.w = iio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])

    def add(self, frame: np.ndarray):
        self.w.append_data(frame)

    def close(self):
        self.w.close()


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
