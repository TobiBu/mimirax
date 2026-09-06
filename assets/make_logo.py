"""The mimirax logo, drawn as SVG primitives: an eye whose iris is a set of inference contours.

Chosen by TB on 2026-09-06 from three drafts (the well, root into water, the eye) and three eye
variants; this file keeps only the chosen form. ``python assets/make_logo.py .`` regenerates the
two SVGs at the repo root; the PNG is a 1024x1024 render of ``mimirax.svg``.

Geometry is written out as SVG primitives (ellipses, paths, circles) by this
script; nothing is traced or rasterised. Deterministic starfield from a fixed
seed so the files are reproducible.
"""

import math
import pathlib
import random
import sys

OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

BG = "#050103"
DEEP, BLUE, MID, BRIGHT, CYAN, WHITE = (
    "#101050",
    "#103070",
    "#1050b0",
    "#1090f0",
    "#d0f0f0",
    "#f0f0f0",
)

DEFS = f"""
  <defs>
    <filter id="bloom" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="22"/></filter>
    <filter id="halo" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="7"/></filter>
    <filter id="soft" x="-40%" y="-40%" width="180%" height="180%"><feGaussianBlur stdDeviation="2.5"/></filter>
    <radialGradient id="flare">
      <stop offset="0" stop-color="{WHITE}"/>
      <stop offset="0.25" stop-color="{CYAN}"/>
      <stop offset="0.55" stop-color="{BRIGHT}" stop-opacity="0.7"/>
      <stop offset="1" stop-color="{MID}" stop-opacity="0"/>
    </radialGradient>
    <radialGradient id="pool">
      <stop offset="0" stop-color="{MID}" stop-opacity="0.55"/>
      <stop offset="0.7" stop-color="{DEEP}" stop-opacity="0.35"/>
      <stop offset="1" stop-color="{DEEP}" stop-opacity="0"/>
    </radialGradient>
  </defs>"""


def glow_path(
    d,
    core=3.5,
    halo=14,
    bloom=34,
    core_color=CYAN,
    halo_color=BRIGHT,
    bloom_color=MID,
    core_op=1.0,
    halo_op=0.65,
    bloom_op=0.35,
    extra="",
):
    """A crisp core stroke over a soft halo over a wide bloom: the raster siblings' glow, as vector."""
    return (
        f'  <path d="{d}" fill="none" stroke="{bloom_color}" stroke-width="{bloom}" stroke-opacity="{bloom_op}" filter="url(#bloom)" {extra}/>\n'
        f'  <path d="{d}" fill="none" stroke="{halo_color}" stroke-width="{halo}" stroke-opacity="{halo_op}" filter="url(#halo)" {extra}/>\n'
        f'  <path d="{d}" fill="none" stroke="{core_color}" stroke-width="{core}" stroke-opacity="{core_op}" {extra}/>\n'
    )


def ellipse_path(cx, cy, rx, ry, start_deg=0.0, sweep_deg=360.0):
    """Ellipse (or arc of one) as a path, so it can take the glow stack."""
    if sweep_deg >= 360.0:
        return (
            f"M{cx - rx:.1f} {cy:.1f} A{rx:.1f} {ry:.1f} 0 1 0 {cx + rx:.1f} {cy:.1f} "
            f"A{rx:.1f} {ry:.1f} 0 1 0 {cx - rx:.1f} {cy:.1f} Z"
        )
    a0, a1 = math.radians(start_deg), math.radians(start_deg + sweep_deg)
    x0, y0 = cx + rx * math.cos(a0), cy + ry * math.sin(a0)
    x1, y1 = cx + rx * math.cos(a1), cy + ry * math.sin(a1)
    large = 1 if sweep_deg > 180 else 0
    return f"M{x0:.1f} {y0:.1f} A{rx:.1f} {ry:.1f} 0 {large} 1 {x1:.1f} {y1:.1f}"


def flare(cx, cy, r=70, arm=150):
    """The family's central four-point star: a radial disc plus two thin lozenges."""
    return (
        f'  <circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#flare)"/>\n'
        f'  <path d="M{cx - arm} {cy} L{cx} {cy - 6} L{cx + arm} {cy} L{cx} {cy + 6} Z" fill="{WHITE}" opacity="0.9" filter="url(#soft)"/>\n'
        f'  <path d="M{cx} {cy - arm * 0.75} L{cx + 5} {cy} L{cx} {cy + arm * 0.75} L{cx - 5} {cy} Z" fill="{WHITE}" opacity="0.9" filter="url(#soft)"/>\n'
    )


def starfield(cx, cy, radius, n=34, seed=7, avoid_y=None):
    """Specks around the motif, denser near it, none over the wordmark."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        ang = rng.uniform(0, 2 * math.pi)
        rad = radius * (0.35 + 0.95 * rng.random() ** 0.6)
        x, y = cx + rad * math.cos(ang), cy + 0.7 * rad * math.sin(ang)
        if not (40 < x < 984 and 40 < y < 984):
            continue
        if avoid_y is not None and y > avoid_y:
            continue
        r = rng.choice([1.5, 1.5, 2, 2, 2.5, 3, 4])
        col = rng.choice([CYAN, CYAN, BRIGHT, WHITE])
        op = round(rng.uniform(0.35, 1.0), 2)
        out.append(
            f'  <circle cx="{x:.0f}" cy="{y:.0f}" r="{r}" fill="{col}" opacity="{op}"/>'
        )
    return "\n".join(out) + "\n"


# --- wordmark: MIMIRAX from strokes, geometric sans, wide tracking -------------------------
def wordmark(cx, baseline, cap=58, sw=9, gap=30):
    """Letterforms as stroked paths, so the file has no font dependency."""
    top = baseline - cap

    def M(x):
        return f"M{x} {baseline} L{x} {top} L{x + 31} {top + 44} L{x + 62} {top} L{x + 62} {baseline}"

    def I(x):
        return f"M{x} {top} L{x} {baseline}"

    def R(x):
        return (
            f"M{x} {baseline} L{x} {top} L{x + 26} {top} A{cap * 0.26:.1f} {cap * 0.26:.1f} 0 0 1 {x + 26} {top + cap * 0.52:.1f} L{x} {top + cap * 0.52:.1f} "
            f"M{x + 22} {top + cap * 0.52:.1f} L{x + 48} {baseline}"
        )

    def A(x):
        return (
            f"M{x} {baseline} L{x + 29} {top} L{x + 58} {baseline} "
            f"M{x + 9.7:.1f} {top + cap * 2 / 3:.1f} L{x + 48.3:.1f} {top + cap * 2 / 3:.1f}"
        )

    def X(x):
        return f"M{x} {top} L{x + 52} {baseline} M{x + 52} {top} L{x} {baseline}"

    letters = [(M, 62), (I, 0), (M, 62), (I, 0), (R, 48), (A, 58), (X, 52)]
    total = sum(w for _, w in letters) + gap * (len(letters) - 1)
    x = cx - total / 2
    d = []
    for fn, w in letters:
        d.append(fn(x))
        x += w + gap
    path = " ".join(d)
    return (
        f'  <path d="{path}" fill="none" stroke="{WHITE}" stroke-width="{sw}" stroke-linecap="butt" stroke-linejoin="bevel"/>\n'
        f'  <path d="{path}" fill="none" stroke="{BRIGHT}" stroke-width="{sw + 8}" stroke-opacity="0.25" filter="url(#halo)"/>\n'
    )


def svg(body, view="0 0 1024 1024", size=1024):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" width="{size}" height="{size}" role="img" aria-label="mimirax">\n'
        f'  <rect x="{view.split()[0]}" y="{view.split()[1]}" width="{view.split()[2]}" height="{view.split()[3]}" fill="{BG}"/>\n'
        + DEFS
        + "\n"
        + body
        + "</svg>\n"
    )


CX, CY = 512, 420

# --- The eye: lid, an iris of contours tightening toward the pupil, the flare as the pupil -------
LENS_HW, LENS_BULGE = 300, 165


def lens_path(cx=CX, cy=CY, hw=LENS_HW, bulge=LENS_BULGE):
    return (
        f"M{cx - hw} {cy} Q{cx} {cy - bulge * 2} {cx + hw} {cy} "
        f"Q{cx} {cy + bulge * 2} {cx - hw} {cy} Z"
    )


def eye_c1(favicon=False):
    """The iris rings tighten geometrically toward the pupil: a posterior concentrating on its mode.

    ``favicon=True`` is the small-size form: lid, two rings and a dot, heavier strokes, no bloom.
    """
    b = "" if favicon else starfield(CX, CY, 360, seed=23, avoid_y=760)
    b += f'  <circle cx="{CX}" cy="{CY}" r="150" fill="url(#pool)"/>\n'
    if favicon:
        b += glow_path(
            lens_path(), core=9, halo=22, bloom=0, core_color=WHITE, bloom_op=0
        )
        for r in (150, 78):
            b += glow_path(
                ellipse_path(CX, CY, r, r), core=8, halo=18, bloom=0, bloom_op=0
            )
        b += f'  <circle cx="{CX}" cy="{CY}" r="70" fill="url(#flare)"/>\n'
        b += f'  <circle cx="{CX}" cy="{CY}" r="16" fill="{WHITE}"/>\n'
        return b
    b += glow_path(lens_path(), core=4.5, halo=18, bloom=44, core_color=WHITE)
    # radii shrink by a constant factor: equal steps in log-radius, the look of a Gaussian's
    # nested credible regions
    for r, op in [(150, 0.9), (104, 0.85), (72, 0.8), (50, 0.75), (35, 0.7)]:
        b += glow_path(
            ellipse_path(CX, CY, r, r),
            core=3,
            halo=11,
            bloom=26,
            core_op=op,
            halo_op=0.5 * op,
            bloom_op=0.3 * op,
        )
    b += flare(CX, CY, r=40, arm=84)
    b += wordmark(512, 862)
    return b


(OUT / "mimirax.svg").write_text(svg(eye_c1()))
(OUT / "mimirax-favicon.svg").write_text(
    svg(eye_c1(favicon=True), view="192 100 640 640", size=640)
)
print("wrote mimirax.svg and mimirax-favicon.svg")
