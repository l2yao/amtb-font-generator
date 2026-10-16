"""Find and fix edge-cutoff glyphs (neighbor slivers / truncated strokes).

A clean crop always has white margin on all sides (extraction pads >=8px).
Ink within EDGE_PX of the border means either a truncated character or a
sliver of a neighboring character leaked into the box.

For each flagged glyph in data/glyphs, the best passing sample from
data/processed/candidates is promoted; if none passes, the file moves to
data/processed/edge_review/ (font falls back for that char).

Usage:
    python src/preprocess/edge_cleanup.py [--apply]
    Without --apply, only reports.
"""
import argparse
import shutil
import sys
from pathlib import Path

EDGE_PX = 5


def repair_crop(path):
    """Erase border-touching sliver components (disconnected from the main
    glyph). Returns True if the repaired image passes the edge check.
    Operates in place on the given path."""
    import cv2
    import numpy as np

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape
    ink = img < 200
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    if n <= 2:
        return False
    areas = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
    main = areas[0][1]
    repaired = False
    for _area, i in areas[1:]:
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
        if x0 <= 3 or y0 <= 3 or x1 >= w - 3 or y1 >= h - 3:
            img[_labels == i] = 255
            repaired = True
    if not repaired:
        return False
    ok, _m, _s = edge_report_array(img)
    if not ok:
        return False
    # cv2.imwrite cannot handle non-ASCII paths on Windows; encode in
    # memory and write with Python instead.
    success, buf = cv2.imencode(".png", img)
    if not success:
        return False
    with open(path, "wb") as fh:
        fh.write(bytes(buf))
    return True


def edge_report_array(img):
    import cv2
    import numpy as np

    h, w = img.shape
    ink = img < 200
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    if not rows.any():
        return False, 0, False
    top = int(np.argmax(rows))
    bottom = h - 1 - int(np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = w - 1 - int(np.argmax(cols[::-1]))
    margin = min(top, bottom, left, right)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    sliver = False
    if n > 2:
        areas = sorted((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n))
        for area, i in areas[:-1]:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y0 = stats[i, cv2.CC_STAT_TOP]
            x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
            y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
            if area > h * w * 0.002 and (x0 <= 3 or y0 <= 3 or x1 >= w - 3 or y1 >= h - 3):
                sliver = True
                break
    return margin >= EDGE_PX and not sliver, margin, sliver


def edge_report(path):
    """Returns (ok, min_margin_px, has_border_sliver)."""
    import cv2
    import numpy as np

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0, False
    return edge_report_array(img)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Quarantine edge-cut glyphs.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--repair-review", action="store_true",
                        help="surgically repair quarantined glyphs (erase border slivers) "
                             "and restore fixed ones to data/glyphs")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from preprocess.pick_best import quality_score

    if args.repair_review:
        review = Path("data/processed/edge_review")
        glyphs = Path("data/glyphs")
        cand_root = Path("data/processed/candidates")
        repaired = replaced = 0
        still_bad = []
        for png in sorted(review.glob("*.png")):
            dest = glyphs / png.name
            if repair_crop(png):
                shutil.move(str(png), str(dest))
                repaired += 1
                continue
            best, best_score = None, -1.0
            cdir = cand_root / png.stem
            if cdir.is_dir():
                for cand in cdir.glob("*.png"):
                    ok, _, _ = edge_report(cand)
                    if not ok:
                        continue
                    s = quality_score(cand)
                    if s > best_score:
                        best, best_score = cand, s
            if best is not None:
                shutil.copy(best, dest)
                png.unlink()
                replaced += 1
            else:
                still_bad.append(png.stem)
        print(f"Repaired: {repaired}, replaced from candidates: {replaced}, "
              f"still quarantined: {len(still_bad)}")
        for stem in still_bad[:30]:
            print(f"  {stem}")
        return

    glyphs = Path("data/glyphs")
    cand_root = Path("data/processed/candidates")
    review = Path("data/processed/edge_review")
    if args.apply:
        review.mkdir(parents=True, exist_ok=True)

    flagged, fixed, dropped = [], [], []
    files = sorted(glyphs.glob("*.png"))
    print(f"Scanning {len(files)} glyphs...", flush=True)
    for n, png in enumerate(files):
        ok, margin, sliver = edge_report(png)
        if ok:
            continue
        flagged.append((png.stem, margin, sliver))
        if not args.apply:
            continue
        # 1) Try surgical repair in place (erase disconnected slivers).
        if repair_crop(png):
            fixed.append(png.stem)
            continue
        # 2) Fall back to the best clean candidate sample.
        best, best_score = None, -1.0
        cdir = cand_root / png.stem
        if cdir.is_dir():
            for cand in cdir.glob("*.png"):
                ok2, _, _ = edge_report(cand)
                if not ok2:
                    continue
                s = quality_score(cand)
                if s > best_score:
                    best, best_score = cand, s
        if best is not None:
            shutil.copy(best, png)
            fixed.append(png.stem)
        else:
            shutil.move(str(png), str(review / png.name))
            dropped.append(png.stem)
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(files)} flagged={len(flagged)}", flush=True)

    print(f"Flagged: {len(flagged)}")
    for stem, margin, sliver in flagged[:40]:
        print(f"  {stem} margin={margin} sliver={sliver}")
    if len(flagged) > 40:
        print(f"  ... and {len(flagged) - 40} more")
    if args.apply:
        print(f"Fixed (repaired or clean sample): {len(fixed)}; quarantined: {len(dropped)}")


if __name__ == "__main__":
    main()
