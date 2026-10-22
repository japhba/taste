"""Compose new handwritten words from the strokes of the handwritten deck (slides/ink/*.svg, filled outline paths in PDF points).

  python scripts/ink_compose.py debug INK_DIR OUT_DIR             numbered stroke maps of every ink file (to pick glyphs)
  python scripts/ink_compose.py compose INK_DIR SPEC.json OUT.svg  a word sequence from glyphs: SPEC = {"lines": [[glyph, ...], ...]}
      glyph = {"src": "02-title", "paths": [3, 4]}  or  {"space": 1.0}  (in units of the median letter gap)
Glyphs from a source are placed on the target baseline (each source's baseline = median bottom of its x-height letters, given per
source in SPEC["baselines"] as the y of the baseline in that file's coordinates) and scaled to the x-height of the first source.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def load(f):
    s = Path(f).read_text()
    vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    return vb, re.findall(r'<path d="([^"]+)"', s)


def to_mpl(d):
    verts, codes = [], []
    for cmd, args in re.findall(r"([MLCZ])([^MLCZ]*)", d):
        a = [float(v) for v in NUM.findall(args)]
        if cmd == "M": verts.append(a[:2]); codes.append(MPath.MOVETO)
        elif cmd == "L":
            for i in range(0, len(a), 2): verts.append(a[i:i + 2]); codes.append(MPath.LINETO)
        elif cmd == "C":
            for i in range(0, len(a), 6):
                for j in range(3): verts.append(a[i + 2 * j:i + 2 * j + 2]); codes.append(MPath.CURVE4)
        elif cmd == "Z": verts.append(verts[-1] if verts else [0, 0]); codes.append(MPath.CLOSEPOLY)
    return MPath(np.array(verts), codes)


def bbox(d):
    v = np.array([float(x) for x in NUM.findall(d)]).reshape(-1, 2)
    return v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()


def debug(ink, out):
    for f in sorted(Path(ink).glob("*.svg")):
        vb, paths = load(f)
        fig, ax = plt.subplots(figsize=(max(6, vb[2] / 12), max(2, vb[3] / 12)))
        for i, d in enumerate(paths):
            ax.add_patch(PathPatch(to_mpl(d), facecolor="black", edgecolor="none"))
            x0, y0, x1, y1 = bbox(d)
            ax.text((x0 + x1) / 2, y0 - 0.6, str(i), color="red", ha="center", va="bottom")
        ax.set_xlim(vb[0], vb[0] + vb[2]); ax.set_ylim(vb[1] + vb[3], vb[1] - 4); ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(Path(out) / f"{f.stem}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
        print(f.stem, len(paths), "strokes")


def compose(ink, spec, out):
    """Each glyph's strokes keep their own shape; the glyph is scaled to the target x-height, its baseline (bottom of its strokes unless
    "base" gives the source y of the baseline, e.g. for p) put on the line, and set after the previous glyph with the writer's letter gap."""
    S = json.loads(Path(spec).read_text()); XH = S["xheights"]; gap = S.get("gap", 1.2); word = S.get("word", 22.0)
    target = S.get("xheight", XH[S["lines"][0][0]["src"]])
    parts, pts, y_line = [], [], 0.0
    for line in S["lines"]:
        x = 0.0
        for g in line:
            if "space" in g: x += g["space"] * word - gap; continue
            vb, paths = load(Path(ink) / f"{g['src']}.svg")
            sel = [paths[i] for i in g["paths"]]; k = target / XH[g["src"]] * g.get("scale", 1.0)
            bx = [bbox(d) for d in sel]; x0, x1 = min(b[0] for b in bx), max(b[2] for b in bx)
            base = g.get("base", max(b[3] for b in bx))
            dx, dy = x - x0 * k + g.get("dx", 0.0), y_line - base * k + g.get("dy", 0.0)
            for d in sel:
                vals = [float(v) for v in NUM.findall(d)]
                new = iter(f"{v * k + (dx if i % 2 == 0 else dy):.2f}" for i, v in enumerate(vals))
                parts.append(NUM.sub(lambda m: next(new), d)); pts.append(np.array(vals).reshape(-1, 2) * k + [dx, dy])
            x += (x1 - x0) * k + gap + g.get("after", 0.0)
        y_line += S.get("line_height", 30.0)
    P = np.vstack(pts); pad = 3.0; (x0, y0), (x1, y1) = P.min(0) - pad, P.max(0) + pad
    scale = S.get("px_per_pt", 2.1502)
    Path(out).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f}" width="{(x1 - x0) * scale:.2f}" '
                         f'height="{(y1 - y0) * scale:.2f}">\n<g fill="#000000">' + "".join(f'<path d="{d}"/>' for d in parts) + "</g>\n</svg>\n")
    print(out, f"{(x1 - x0) * scale:.1f} x {(y1 - y0) * scale:.1f} px")


def preview(svg, png):
    vb, paths = load(svg)
    fig, ax = plt.subplots(figsize=(vb[2] / 15, vb[3] / 15))
    for d in paths: ax.add_patch(PathPatch(to_mpl(d), facecolor="black", edgecolor="none"))
    ax.set_xlim(vb[0], vb[0] + vb[2]); ax.set_ylim(vb[1] + vb[3], vb[1]); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(png, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    {"debug": debug, "compose": compose, "preview": preview}[sys.argv[1]](*sys.argv[2:])
