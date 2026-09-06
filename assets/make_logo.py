"""Hand-authored SVG logo drafts for mimirax, to the A2.3 style spec.

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


# --- Draft A: the well --------------------------------------------------------------------
def draft_a(favicon=False):
    b = ""
    rx, ry = 300, 150
    if favicon:
        # The small-size form: the rim, two contours and the centre, drawn heavier and
        # without the wide bloom, so that 32 px keeps three rings and 16 px keeps a
        # ring and a dot instead of a smear.
        b += f'  <ellipse cx="{CX}" cy="{CY}" rx="{rx}" ry="{ry}" fill="url(#pool)"/>\n'
        b += glow_path(
            ellipse_path(CX, CY, rx, ry),
            core=9,
            halo=22,
            bloom=0,
            core_color=WHITE,
            bloom_op=0,
        )
        for f in (0.58, 0.27):
            b += glow_path(
                ellipse_path(CX, CY, rx * f, ry * f),
                core=8,
                halo=18,
                bloom=0,
                bloom_op=0,
            )
        b += f'  <circle cx="{CX}" cy="{CY}" r="80" fill="url(#flare)"/>\n'
        b += f'  <circle cx="{CX}" cy="{CY}" r="16" fill="{WHITE}"/>\n'
        return b
    b += starfield(CX, CY, 360, avoid_y=760)
    # pool
    b += f'  <ellipse cx="{CX}" cy="{CY}" rx="{rx}" ry="{ry}" fill="url(#pool)"/>\n'
    # well wall: lower rim offset down, joined by short verticals (a cylinder mouth seen from above-front)
    depth = 52
    b += glow_path(
        ellipse_path(CX, CY + depth, rx, ry, 0, 180),
        core=3,
        halo=12,
        bloom=28,
        core_op=0.55,
        halo_op=0.4,
        bloom_op=0.25,
    )
    b += glow_path(
        f"M{CX - rx} {CY} L{CX - rx} {CY + depth} M{CX + rx} {CY} L{CX + rx} {CY + depth}",
        core=3,
        halo=12,
        bloom=28,
        core_op=0.55,
        halo_op=0.4,
        bloom_op=0.25,
    )
    # rim
    b += glow_path(
        ellipse_path(CX, CY, rx, ry), core=4.5, halo=18, bloom=44, core_color=WHITE
    )
    # ripples: contour levels converging on the centre -- spacing shrinks inward
    for f, op in [(0.78, 0.9), (0.58, 0.85), (0.41, 0.8), (0.27, 0.75), (0.15, 0.7)]:
        b += glow_path(
            ellipse_path(CX, CY, rx * f, ry * f),
            core=3,
            halo=11,
            bloom=26,
            core_op=op,
            halo_op=0.5 * op,
            bloom_op=0.3 * op,
        )
    b += flare(CX, CY, r=64, arm=130)
    b += wordmark(512, 862)
    return b


# --- Draft B: root into water --------------------------------------------------------------
def draft_b():
    b = starfield(CX, 380, 380, seed=11, avoid_y=760)
    surf_y = 470
    # root: main taper with two rootlets, descending from the top edge of the motif band
    root = f"M512 120 C 496 190, 534 250, 512 {surf_y}"
    rootlet_l = f"M512 250 C 470 280, 440 330, 430 {surf_y}"
    rootlet_r = f"M512 300 C 560 330, 585 380, 590 {surf_y}"
    b += glow_path(root, core=5, halo=16, bloom=36, core_color=WHITE)
    b += glow_path(rootlet_l, core=3, halo=11, bloom=26, core_op=0.8)
    b += glow_path(rootlet_r, core=3, halo=11, bloom=26, core_op=0.8)
    # water surface
    b += f'  <ellipse cx="{CX}" cy="{surf_y + 60}" rx="330" ry="120" fill="url(#pool)"/>\n'
    b += glow_path(
        f"M{CX - 320} {surf_y} L{CX + 320} {surf_y}",
        core=4,
        halo=16,
        bloom=40,
        core_color=WHITE,
    )
    # ripples below the surface: lower half-ellipses, the reflection of the contours
    for rx_, f in [(60, 0.95), (125, 0.85), (200, 0.7), (285, 0.5)]:
        b += glow_path(
            ellipse_path(CX, surf_y, rx_, rx_ * 0.42, 0, 180),
            core=3,
            halo=11,
            bloom=26,
            core_op=f,
            halo_op=0.5 * f,
            bloom_op=0.3 * f,
        )
    b += flare(CX, surf_y, r=56, arm=120)
    b += wordmark(512, 862)
    return b


# --- Draft C: the eye -------------------------------------------------------------------------
def draft_c():
    b = starfield(CX, CY, 360, seed=23, avoid_y=760)
    hw, bulge = 300, 165
    lens = (
        f"M{CX - hw} {CY} Q{CX} {CY - bulge * 2} {CX + hw} {CY} "
        f"Q{CX} {CY + bulge * 2} {CX - hw} {CY} Z"
    )
    b += f'  <circle cx="{CX}" cy="{CY}" r="150" fill="url(#pool)"/>\n'
    b += glow_path(lens, core=4.5, halo=18, bloom=44, core_color=WHITE)
    # iris: concentric circles = contour levels; a rune-like tick at the four cardinal points
    for r, op in [(150, 0.9), (112, 0.85), (78, 0.8), (48, 0.75)]:
        b += glow_path(
            ellipse_path(CX, CY, r, r),
            core=3,
            halo=11,
            bloom=26,
            core_op=op,
            halo_op=0.5 * op,
            bloom_op=0.3 * op,
        )
    ticks = " ".join(
        f"M{CX + 150 * math.cos(a):.1f} {CY + 150 * math.sin(a):.1f} L{CX + 172 * math.cos(a):.1f} {CY + 172 * math.sin(a):.1f}"
        for a in [math.radians(d) for d in (90, 270)]
    )
    b += glow_path(ticks, core=3, halo=10, bloom=22, core_op=0.8)
    b += flare(CX, CY, r=44, arm=90)
    b += wordmark(512, 862)
    return b


(OUT / "draft-a-well.svg").write_text(svg(draft_a()))
(OUT / "draft-b-root.svg").write_text(svg(draft_b()))
(OUT / "draft-c-eye.svg").write_text(svg(draft_c()))
# favicon: motif-only square crop of draft A, outer bloom dropped by the smaller canvas
(OUT / "favicon-a-well.svg").write_text(
    svg(draft_a(favicon=True), view="192 100 640 640", size=640)
)
print("wrote", sorted(p.name for p in OUT.iterdir()))
