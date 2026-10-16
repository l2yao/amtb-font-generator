"""Detect cutoff artifacts using the flat-cut signature.

A truncated stroke leaves a long, ruler-straight axis-aligned ink edge at
the glyph's bounding-box extreme. Natural brushwork tapers and wobbles, so
a straight run longer than ~35% of the bbox side is a cutoff. Also flags
small disconnected components lurking in the outer band (neighbor slivers
fully inside the crop).

Usage: python src/preprocess/cutoff_scan.py [--apply]
Without --apply: report only. With --apply: quarantine flagged files to
data/processed/cutoff_review/.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def straight_run_ratio(edge_positions, span):
    """Longest run of near-constant positions, as fraction of span."""
    if len(edge_positions) < 3:
        return 0.0
    best = cur = 1
    for a, b in zip(edge_positions[:-1], edge_positions[1:]):
        if abs(a - b) <= 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / max(span, 1)


def scan_crop(path):
    """Returns (flags: dict). Flags: flat_left/right/top/bottom, sliver."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"unreadable": True}
    h, w = img.shape
    ink = img < 200
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    if not rows.any():
        return {"blank": True}
    top = int(np.argmax(rows))
    bottom = h - 1 - int(np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = w - 1 - int(np.argmax(cols[::-1]))
    bh, bw = bottom - top + 1, right - left + 1
    flags = {}
    # Left/right extremes: leftmost/rightmost ink column per row.
    for y in range(top, bottom + 1):
        xs = np.where(ink[y, :])[0]
        if len(xs) == 0:
            continue
    lefts, rights, tops, bottoms = [], [], [], []
    for y in range(top, bottom + 1):
        xs = np.where(ink[y, :])[0]
        if len(xs):
            lefts.append(int(xs[0]))
            rights.append(int(xs[-1]))
    for x in range(left, right + 1):
        ys = np.where(ink[:, x])[0]
        if len(ys):
            tops.append(int(ys[0]))
            bottoms.append(int(ys[-1]))
    if straight_run_ratio(lefts, bh) > 0.35:
        flags["flat_left"] = True
    if straight_run_ratio(rights, bh) > 0.35:
        flags["flat_right"] = True
    if straight_run_ratio(tops, bw) > 0.35:
        flags["flat_top"] = True
    if straight_run_ratio(bottoms, bw) > 0.35:
        flags["flat_bottom"] = True
    # Interior sliver: small disconnected component in the outer band.
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if n > 2:
        areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
        main_area = areas[0][0]
        band = int(min(h, w) * 0.15)
        for area, i in areas[1:]:
            if area > 0.08 * main_area:
                continue
            x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
            y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
            if x0 < band or y0 < band or x1 > w - band or y1 > h - band:
                flags["sliver"] = True
                break
    return flags


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Scan for cutoff artifacts.")
    parser.add_argument("--glyph_dir", default="data/glyphs")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    glyphs = Path(args.glyph_dir)
    review = Path("data/processed/cutoff_review")
    if args.apply:
        review.mkdir(parents=True, exist_ok=True)
    files = sorted(glyphs.glob("*.png"))
    print(f"Scanning {len(files)} glyphs...", flush=True)
    flagged = {}
    for n, png in enumerate(files):
        flags = scan_crop(png)
        if flags:
            flagged[png.stem] = flags
            if args.apply:
                shutil.move(str(png), str(review / png.name))
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(files)} flagged={len(flagged)}", flush=True)
    print(f"Flagged: {len(flagged)}")
    kinds = {}
    for stem, flags in flagged.items():
        for k in flags:
            kinds[k] = kinds.get(k, 0) + 1
    print("By kind:", kinds)
    for stem in sorted(flagged)[:60]:
        print(f"  {stem}: {sorted(flagged[stem])}")
    if len(flagged) > 60:
        print(f"  ... and {len(flagged) - 60} more")


if __name__ == "__main__":
    main()
