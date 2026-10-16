"""Pick the best glyph sample per character.

Sources: manifest candidates (trusted transcriptions) + existing
data/glyphs labels (OCR-labeled, re-verified here). Quality score combines
sharpness (Laplacian variance), ink-coverage band, and edge margin.
Existing labels that OCR confidently contradicts are dropped in favor of
manifest candidates. Winners are written to data/glyphs/{char}.png.

Usage:
    python src/preprocess/pick_best.py
"""
import os
import shutil
import sys
from pathlib import Path

VERIFY_CONF = 0.8
VETO_CONF = 0.85

SIMP2TRAD = {
    "爱": "愛", "门": "門", "义": "義", "圣": "聖", "觉": "覺", "观": "觀",
    "说": "說", "读": "讀", "讲": "講", "经": "經", "习": "習", "时": "時",
    "现": "現", "实": "實", "让": "讓", "过": "過", "还": "還", "进": "進",
    "远": "遠", "运": "運", "边": "邊", "达": "達", "极": "極", "乐": "樂",
    "无": "無", "为": "為", "发": "發", "个": "個", "们": "們", "国": "國",
    "学": "學", "宝": "寶", "净": "淨", "汇": "匯", "汉": "漢", "华": "華",
    "尽": "盡", "应": "應", "广": "廣", "庄": "莊", "庆": "慶", "寿": "壽",
    "将": "將", "尔": "爾", "尘": "塵", "对": "對", "严": "嚴", "护": "護",
    "报": "報", "担": "擔", "拟": "擬", "证": "證", "识": "識", "误": "誤",
    "调": "調", "请": "請", "诸": "諸", "谢": "謝", "历": "歷", "钟": "鐘",
    "药": "藥", "苏": "蘇", "虑": "慮", "虽": "雖", "诗": "詩", "话": "話",
    "语": "語", "车": "車", "轮": "輪", "辩": "辯", "农": "農", "连": "連",
    "迟": "遲", "释": "釋", "弥": "彌", "愿": "願",
}


def norm(ch):
    return SIMP2TRAD.get(ch, ch)


def quality_score(path):
    """Higher is better. Combines sharpness, ink band, edge margin."""
    import cv2
    import numpy as np

    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return -1.0
    if img is None:
        return -1.0
    h, w = img.shape
    sharp = float(np.log1p(max(0.0, cv2.Laplacian(img, cv2.CV_64F).var())))
    ink = img < 200
    ratio = float(ink.mean())
    if ratio < 0.02 or ratio > 0.9:
        ink_f = 0.0
    elif 0.08 <= ratio <= 0.6:
        ink_f = 1.0
    elif ratio < 0.08:
        ink_f = (ratio - 0.02) / 0.06
    else:
        ink_f = (0.9 - ratio) / 0.3
    rows = np.any(ink, axis=1)
    cols = np.any(ink, axis=0)
    if not rows.any() or not cols.any():
        return -1.0
    top = int(np.argmax(rows))
    bottom = h - 1 - int(np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = w - 1 - int(np.argmax(cols[::-1]))
    if min(top, bottom, left, right) <= 2:
        return -1.0  # edge-cut: truncated char or neighbor sliver
    margin = min(top, bottom, left, right) / max(h, w)
    margin_f = min(1.0, margin / 0.03) if margin > 0 else 0.0
    return sharp * (0.2 + 0.8 * ink_f) * (0.3 + 0.7 * margin_f)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse

    parser = argparse.ArgumentParser(description="Pick best glyph per character.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    shard_args = parser.parse_args()

    from rapidocr_onnxruntime import RapidOCR

    glyphs_dir = Path("data/glyphs")
    cand_root = Path("data/processed/candidates")
    ocr = RapidOCR()

    def ocr_one(path):
        try:
            res, _ = ocr(str(path), use_det=False, use_cls=False)
            if res and res[0]:
                return norm(res[0][0]), float(res[0][1])
        except Exception:
            pass
        return "", 0.0

    # Gather: (char, path, trusted, score placeholder)
    pool = {}
    for char_dir in cand_root.iterdir():
        if not char_dir.is_dir() or len(char_dir.name) != 1:
            continue
        for png in char_dir.glob("*.png"):
            pool.setdefault(char_dir.name, []).append((png, True))
    existing = {}
    for png in glyphs_dir.glob("*.png"):
        if len(png.stem) == 1:
            existing.setdefault(png.stem, png)
    for ch, png in existing.items():
        pool.setdefault(ch, []).append((png, False))

    print(f"{len(pool)} distinct characters, "
          f"{sum(len(v) for v in pool.values())} samples", flush=True)

    kept = upgraded = dropped_bad = 0
    chars = sorted(pool.keys())
    if shard_args.nshards > 1:
        chars = [c for i, c in enumerate(chars) if i % shard_args.nshards == shard_args.shard]
    print(f"shard {shard_args.shard}/{shard_args.nshards}: {len(chars)} chars", flush=True)

    import cv2 as _cv2
    import numpy as _np

    def _ink_ratio(p):
        try:
            raw = _np.fromfile(str(p), dtype=_np.uint8)
            img = _cv2.imdecode(raw, _cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            return float((img < 200).mean())
        except Exception:
            return 0.0

    for n, ch in enumerate(chars):
        samples = pool[ch]
        # Fragment guard: drop samples with far less ink than the char's
        # fullest sample (segmentation fragments score well on sharpness
        # and margins but are broken glyphs).
        ratios = [(s, _ink_ratio(s[0])) for s in samples]
        fullest = max((r for _, r in ratios), default=0.0)
        if fullest > 0:
            samples = [s for s, r in ratios if r >= 0.45 * fullest]
        if not samples:
            dropped_bad += 1
            if ch in existing:
                existing[ch].unlink(missing_ok=True)
            continue
        best_path, best_score, best_trusted = None, -2.0, False
        for png, trusted in samples:
            if not trusted:
                pred, conf = ocr_one(png)
                if pred and conf >= VERIFY_CONF and norm(pred) != norm(ch):
                    continue  # OCR confidently contradicts old label
            if trusted:
                pred, conf = ocr_one(png)
                if pred and conf >= VETO_CONF and norm(pred) != norm(ch):
                    continue  # alignment likely wrong at this position
            score = quality_score(png)
            if score > best_score:
                best_score, best_path, best_trusted = score, png, trusted
        if best_path is None:
            dropped_bad += 1
            if ch in existing:
                existing[ch].unlink(missing_ok=True)
            continue
        dest = glyphs_dir / f"{ch}.png"
        current = existing.get(ch)
        if current is not None and current.resolve() == best_path.resolve():
            kept += 1
        else:
            shutil.copy(best_path, dest)
            upgraded += 1
        existing[ch] = dest
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(chars)} kept={kept} upgraded={upgraded} dropped={dropped_bad}",
                  flush=True)

    print(f"kept={kept} upgraded_or_new={upgraded} dropped_all_bad={dropped_bad}")
    print(f"data/glyphs now: {len(list(glyphs_dir.glob('*.png')))} files")


if __name__ == "__main__":
    main()
