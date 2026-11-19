"""Two 'people' the fly can see, built from geometry (the ray tracer has no image textures), and the fly's room.

The fly stands at the origin facing +x. A face hangs 4.5 cm in front of it against a dark backdrop, about 50 deg
wide: at the fly's ~4.6 deg ommatidial spacing that is ~11 columns across, so eyes / glasses / mouth are each a few
ommatidia. Face A ("Alex"): round face, two big eyes, a smile. Face B ("Sam"): long face, dark hair, sunglasses
bar, a flat mouth.
"""
from __future__ import annotations

import numpy as np

from .core import World, Sphere, Plane, Box, add_material

D = 0.045          # face distance (m)
Z = 0.002          # fly eye height above the floor (m); faces are centred a little above it


def _mats():
    add_material("skin", (0.45, 0.85, 0.95, 1.0))
    add_material("ink", (0.01, 0.015, 0.015, 0.02))
    add_material("backdrop", (0.02, 0.04, 0.05, 0.05))
    add_material("table2", (0.05, 0.10, 0.12, 0.14), pattern=1, refl2=(0.03, 0.06, 0.08, 0.09), scale=0.01)


def face_parts(kind: str, cx=D, cz=0.012):
    """List of (center, radii, material). y is left(+)/right(-) from the fly's view, z up."""
    P = []
    if kind == "A":
        P.append(((cx, 0, cz), (0.002, 0.020, 0.020), "skin"))                    # round face
        for s in (+1, -1):
            P.append(((cx - 0.0025, s * 0.0075, cz + 0.006), (0.001, 0.0042, 0.0042), "ink"))   # big eyes
        for a in np.linspace(-0.9, 0.9, 7):                                        # smile: arc of dots
            P.append(((cx - 0.0025, 0.009 * np.sin(a), cz - 0.006 - 0.004 * np.cos(a)), (0.001, 0.0022, 0.0022), "ink"))
    elif kind == "B":
        P.append(((cx, 0, cz - 0.002), (0.002, 0.015, 0.024), "skin"))            # long face
        P.append(((cx - 0.0025, 0, cz + 0.017), (0.001, 0.016, 0.008), "ink"))     # hair cap
        P.append(((cx - 0.0025, 0, cz + 0.003), (0.001, 0.014, 0.0035), "ink"))    # sunglasses bar
        P.append(((cx - 0.0025, 0, cz - 0.013), (0.001, 0.007, 0.0016), "ink"))    # flat mouth
    return P


def room(kind: str | None):
    """World with an optional face (None = blank backdrop)."""
    _mats()
    w = World(detail=0.3, light_pos=(0.0, 0.0, 0.5), light_color=(1.0, 2.0, 2.0, 2.0), ambient=(0.35, 0.4, 0.4, 0.4))
    w.planes.append(Plane((0, 0, -0.0005), (0, 0, 1), "table2"))
    w.planes.append(Plane((D + 0.004, 0, 0), (-1, 0, 0), "backdrop"))
    w.planes.append(Plane((0, 0, 0.3), (0, 0, -1), "ceiling"))
    for a, n in [((-0.3, 0, 0), (1, 0, 0)), ((0, 0.3, 0), (0, -1, 0)), ((0, -0.3, 0), (0, 1, 0))]:
        w.planes.append(Plane(a, n, "wall"))
    if kind:
        for c, r, m in face_parts(kind):
            w.spheres.append(Sphere(c, r, m))
    return w
