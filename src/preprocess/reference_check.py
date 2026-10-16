"""Check each calligraphy glyph against the Noto Sans SC reference rendering.

Idea: the reference shows where ink *should* be. A cutoff glyph is missing
an entire region present in the reference. Both images are normalized to
their ink bounding boxes (position/size agnostic), then we measure how much
of the reference ink is covered by (slightly dilated) calligraphy ink --
overall recall plus per-side recall (top/bottom/left/right thirds) to say
WHERE the cutoff is.

Usage:
    python src/preprocess/reference_check.py [--limit N] [--threshold 0.62]
    Writes data/processed/refcheck.json (char -> scores).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NORM = 96
DILATE = 3


def render_ref(ch, font, size=256):
    img = Image.new("L", (size, size), 255)
    d = ImageDraw.Draw(img)
    d.text((10, 10), ch, font=font, fill=0)
    a = np.array(img)
    ink = a < 200
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    return a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def normalize(gray):
    ink = gray < 200
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    crop = gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return cv2.resize(crop, (NORM, NORM), interpolation=cv2.INTER_AREA)


def component_coverage(call_img, ref_img, dilate_px=5, min_comp_frac=0.10):
    """For each significant reference component, how much of its bounding-box
    region is covered by (dilated) calligraphy ink. Returns list of
    (fraction, bbox) for components >= min_comp_frac of reference ink.
    A cutoff removes a whole component -> fraction near 0."""
    call = normalize(call_img)
    ref = normalize(ref_img)
    if call is None or ref is None:
        return []
    ref_ink = (ref < 200).astype(np.uint8)
    total = int(ref_ink.sum())
    if total == 0:
        return []
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(ref_ink, 8)
    call_ink = (call < 200).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1,) * 2)
    call_d = cv2.dilate(call_ink, kernel) > 0
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < total * min_comp_frac:
            continue
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        x1 = x0 + stats[i, cv2.CC_STAT_WIDTH]
        y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
        region = np.zeros_like(ref_ink, dtype=bool)
        region[y0:y1, x0:x1] = True
        comp_mask = (_labels == i)
        # coverage of the component's own pixels (not whole bbox: style shifts)
        cov = float((comp_mask & call_d).sum()) / max(int(comp_mask.sum()), 1)
        out.append((round(cov, 3), (int(x0), int(y0), int(x1), int(y1))))
    return out


def recall_score(call_img, ref_img):
    """Fraction of reference ink covered by dilated calligraphy ink."""
    call = normalize(call_img)
    ref = normalize(ref_img)
    if call is None or ref is None:
        return 0.0, {}
    ref_ink = ref < 200
    call_ink = (call < 200).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE * 2 + 1,) * 2)
    call_d = cv2.dilate(call_ink, kernel) > 0
    total = int(ref_ink.sum())
    if total == 0:
        return 1.0, {}
    overall = float((ref_ink & call_d).sum()) / total
    regions = {}
    h, w = ref_ink.shape
    thirds = [(0, h // 3, "top"), (h // 3, 2 * h // 3, "mid"), (2 * h // 3, h, "bottom")]
    for y0, y1, name in thirds:
        seg = ref_ink[y0:y1, :]
        t = int(seg.sum())
        regions[name] = float((seg & call_d[y0:y1, :]).sum()) / t if t else 1.0
    thirds_v = [(0, w // 3, "left"), (2 * w // 3, w, "right")]
    for x0, x1, name in thirds_v:
        seg = ref_ink[:, x0:x1]
        t = int(seg.sum())
        regions[name] = float((seg & call_d[:, x0:x1]).sum()) / t if t else 1.0
    return overall, regions


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Reference-based cutoff check.")
    parser.add_argument("--glyph_dir", default="data/glyphs")
    parser.add_argument("--ref_ttf", default="data/fonts/NotoSansSC-Regular.ttf")
    parser.add_argument("--out", default="data/processed/refcheck.json")
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    font = ImageFont.truetype(args.ref_ttf, 200)
    files = sorted(Path(args.glyph_dir).glob("*.png"))
    if args.limit:
        files = files[: args.limit]
    print(f"Checking {len(files)} glyphs against reference...", flush=True)
    results = {}
    for n, png in enumerate(files):
        raw = np.fromfile(str(png), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None:
            results[png.stem] = {"error": "unreadable"}
            continue
        ref = render_ref(png.stem, font)
        if ref is None:
            results[png.stem] = {"error": "no-reference"}
            continue
        overall, regions = recall_score(img, ref)
        comps = component_coverage(img, ref)
        entry = {"recall": round(overall, 3), **{k: round(v, 3) for k, v in regions.items()}}
        worst = min([c for c, _b in comps], default=1.0)
        entry["min_comp_cov"] = worst
        entry["n_big_comp"] = len(comps)
        if worst < 0.35:
            entry["missing"] = [[c, list(b)] for c, b in comps if c < 0.35]
        results[png.stem] = entry
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(files)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False)
    bad = sorted(((v["recall"], k) for k, v in results.items() if "recall" in v))[:40]
    print(f"Lowest recall (threshold {args.threshold}):")
    for score, ch in bad:
        if score < args.threshold:
            r = results[ch]
            print(f"  {ch}: {score} " + " ".join(f"{k}={r[k]}" for k in ("top", "mid", "bottom", "left", "right")))
    missing = sorted(
        ((min([c for c, _ in v.get("missing", [])] or [1.0]), k) for k, v in results.items() if "missing" in v)
    )
    print(f"Glyphs with nearly-uncovered reference components: {len(missing)}")
    for score, ch in missing[:40]:
        print(f"  {ch}: min_comp_cov={score} n_big={results[ch]['n_big_comp']}")


if __name__ == "__main__":
    main()
