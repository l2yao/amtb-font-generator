"""Extract + label amtb.tw images using manifest transcriptions.

For each manifest entry (except JS seals), the `content` field holds the
exact text shown in the image. Layout handling:
- wide images: single pass, rows top-to-bottom, chars left-to-right
- portrait images: split into vertical bands at wide ink gaps, extract
  each band with band-local thresholds, then try candidate reading orders
  (columns right-to-left; rows top-to-bottom for grids) and keep the order
  that best matches OCR anchor predictions (>=60% agreement + exact count)

Accepted matches go to data/processed/candidates/{char}/ for quality
picking. Everything else goes to the unlabeled pool for the OCR pass.

Usage:
    python src/preprocess/manifest_extract.py --manifest data/amtb_raw/manifest.jsonl
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # amtb originals reach 200MP; trusted source

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ANCHOR_THRESHOLD = 0.6


def _is_han(cp):
    return ((0x3400 <= cp <= 0x4DBF) or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF) or (0x20000 <= cp <= 0x3134F))


# OCR often outputs simplified variants; normalize for anchor matching.
SIMP2TRAD = {
    "爱": "愛", "门": "門", "义": "義", "圣": "聖", "觉": "覺", "观": "觀",
    "说": "說", "读": "讀", "讲": "講", "经": "經", "习": "習", "时": "時",
    "现": "現", "实": "實", "让": "讓", "过": "過", "还": "還", "进": "進",
    "远": "遠", "运": "運", "边": "邊", "达": "達", "极": "極", "乐": "樂",
    "无": "無", "为": "為", "发": "發", "个": "個", "们": "們", "国": "國",
    "学": "學", "宝": "寶", "净": "淨", "亿": "億", "汇": "匯", "汉": "漢",
    "华": "華", "尽": "盡", "应": "應", "广": "廣", "庄": "莊", "庆": "慶",
    "寿": "壽", "将": "將", "尔": "爾", "尘": "塵", "对": "對", "严": "嚴",
    "护": "護", "报": "報", "担": "擔", "拟": "擬", "拣": "揀", "证": "證",
    "识": "識", "误": "誤", "调": "調", "请": "請", "诸": "諸", "谢": "謝",
    "历": "歷", "钟": "鐘", "药": "藥", "苏": "蘇", "虑": "慮", "虏": "虜",
    "虽": "雖", "诗": "詩", "话": "話", "语": "語", "车": "車", "轮": "輪",
    "辩": "辯", "农": "農", "运": "運", "连": "連", "迟": "遲", "释": "釋",
    "弥": "彌", "愿": "願", "專": "專",
}


def load_manifest(path):
    entries = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = item.get("file")
            if fp and fp not in entries:
                entries[fp] = item
    return entries


def clean_text(content):
    """Manifest content with whitespace removed and restricted to Han
    characters (punctuation in the text rarely segments as boxes)."""
    return "".join(ch for ch in (content or "") if not ch.isspace() and _is_han(ord(ch)))


def ocr_crops(ocr, paths):
    """Returns [(pred, conf)] with simplified forms normalized."""
    out = []
    for p in paths:
        try:
            res, _ = ocr(str(p), use_det=False, use_cls=False)
            if res and res[0]:
                text = SIMP2TRAD.get(res[0][0], res[0][0])
                out.append((text, float(res[0][1])))
            else:
                out.append(("", 0.0))
        except Exception:
            out.append(("", 0.0))
    return out


def anchor_score(preds, text):
    if not preds or len(preds) != len(text):
        return 0.0
    return sum(1 for (p, _c), t in zip(preds, text) if p == t) / len(text)


def align_nw(preds, text, match=3.0, mismatch=-1.0, empty=-0.5, gap=-2.0):
    """Needleman-Wunsch alignment of OCR predictions to text.

    Returns (pairs, match_ratio) where pairs is a list of
    (crop_idx or None, char_idx or None). match_ratio = exact matches
    over text length.
    """
    n, m = len(preds), len(text)

    def pair_score(p, t):
        pred = p[0] if isinstance(p, tuple) else p
        if not pred:
            return empty
        if pred == t:
            return match
        return mismatch

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = max(
                dp[i - 1][j - 1] + pair_score(preds[i - 1], text[j - 1]),
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap,
            )
    pairs, exact = [], 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + pair_score(preds[i - 1], text[j - 1]):
            pairs.append((i - 1, j - 1))
            pred = preds[i - 1][0] if isinstance(preds[i - 1], tuple) else preds[i - 1]
            if pred == text[j - 1]:
                exact += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs, (exact / m if m else 0.0)


def process_image(ocr, fp, text, tmp_root, stem,
                  min_match_ratio=0.5, max_gaps=4):
    """Returns list of (crop_path, char) assignments or None."""
    from PIL import Image, ImageOps
    from preprocess.extract_glyphs import (
        detect_glyph_boxes,
        group_columns,
        column_major_orders,
        row_major_order,
        save_ordered_crops,
        remove_red_seals,
    )

    rgb = Image.open(fp).convert("RGB")
    rgb = remove_red_seals(rgb)
    image = ImageOps.autocontrast(rgb.convert("L"))

    # Detect on a downscaled copy for speed and scale-stable thresholds;
    # crops are cut from the full-resolution image.
    k = max(image.width, image.height) / 2000.0 if max(image.width, image.height) > 2000 else 1.0
    det_image = image
    if k > 1.0:
        det_image = image.resize((int(image.width / k), int(image.height / k)),
                                 Image.Resampling.LANCZOS)

    items, h, w = detect_glyph_boxes(det_image)
    if k > 1.0:
        items = [(x * k, y * k, cw * k, ch * k, area * k * k, yc * k)
                 for x, y, cw, ch, area, yc in items]
        h, w = det_image.height * k, det_image.width * k
    if not items or not text:
        return None

    orders = []
    if w > h * 1.2:
        orders.append(row_major_order(items))
    else:
        cols = group_columns(items)
        orders.extend(column_major_orders(cols, column_order="rtl"))

    best = None
    best_key = None
    for oi, ordered in enumerate(orders):
        if not ordered or abs(len(ordered) - len(text)) > max(3, len(text) // 8):
            continue
        sub = Path(tmp_root) / stem / f"o{oi}"
        sub.mkdir(parents=True, exist_ok=True)
        saved = save_ordered_crops(image, ordered, 256, None, stem, str(sub))
        saved = sorted(saved)
        preds = ocr_crops(ocr, saved)
        pairs, ratio = align_nw(preds, list(text))
        gaps = sum(1 for a, b in pairs if a is None or b is None)
        key = (ratio, -gaps)
        if best_key is None or key > best_key:
            best_key = key
            best = (saved, pairs, ratio, gaps)
            if ratio >= min_match_ratio and gaps <= max_gaps:
                break  # good enough; skip remaining orders

    if best is not None:
        saved, pairs, ratio, gaps = best
        if ratio >= min_match_ratio and gaps <= max_gaps:
            assigned = [(saved[a], text[b]) for a, b in pairs
                        if a is not None and b is not None]
            return assigned
    return None



def _worker_init():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR

    _OCR = RapidOCR()


def _process_one(job):
    """Worker: extract + manifest-align one image. Returns summary tuple."""
    fp, text, stem, tmp_root, cand_dir, unl_dir = job
    import traceback

    stem_tmp = Path(tmp_root) / Path(fp).stem
    if stem_tmp.exists():
        shutil.rmtree(stem_tmp)
    try:
        assigned = process_image(_OCR, fp, text, tmp_root, stem)
    except Exception as exc:
        return (fp, "error", 0, str(exc) + " " + traceback.format_exc(limit=3))
    try:
        if assigned:
            for idx, (path, ch) in enumerate(assigned):
                char_dir = Path(cand_dir) / ch
                char_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, char_dir / f"{stem}_{idx:04d}.png")
            return (fp, "matched", len(assigned), "")
        src_dir = Path(tmp_root) / stem
        n = 0
        if src_dir.exists():
            for sub in sorted(src_dir.rglob("*.png")):
                shutil.copy(sub, Path(unl_dir) / f"{stem}_{n:04d}.png")
                n += 1
        return (fp, "mismatched", n, "")
    except Exception as exc:
        return (fp, "error", 0, str(exc))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Manifest-driven glyph extraction.")
    parser.add_argument("--manifest", default="data/amtb_raw/manifest.jsonl")
    parser.add_argument("--candidates_dir", default="data/processed/candidates")
    parser.add_argument("--unlabeled_dir", default="data/processed/glyphs_amtb_unlabeled")
    parser.add_argument("--done_list", default="data/processed/manifest_done.txt")
    parser.add_argument("--skip_categories", nargs="*", default=["JS"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    import multiprocessing as mp

    entries = load_manifest(args.manifest)
    done = set()
    if os.path.exists(args.done_list):
        with open(args.done_list, encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}

    todo = [(fp, it) for fp, it in sorted(entries.items())
            if fp not in done and it.get("menuid") not in args.skip_categories
            and os.path.exists(fp)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} images to process ({len(done)} already done)", flush=True)

    cand_dir = Path(args.candidates_dir)
    unl_dir = Path(args.unlabeled_dir)
    unl_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path("data/processed/manifest_tmp")

    jobs = [(fp, clean_text(item.get("content")), Path(fp).stem,
             str(tmp_root), str(cand_dir), str(unl_dir)) for fp, item in todo]

    matched = mismatched = errors = 0
    with open(args.done_list, "a", encoding="utf-8") as done_fh:
        try:
            with mp.Pool(args.workers, initializer=_worker_init) as pool:
                for n, (fp, status, count, extra) in enumerate(
                        pool.imap_unordered(_process_one, jobs, chunksize=2)):
                    if status == "matched":
                        matched += 1
                    elif status == "mismatched":
                        mismatched += 1
                    else:
                        errors += 1
                        print(f"  ERROR {Path(fp).stem}: {extra[:200]}", flush=True)
                    done_fh.write(fp + "\n")
                    if (n + 1) % 25 == 0:
                        done_fh.flush()
                        print(f"  {n + 1}/{len(jobs)} matched={matched} "
                              f"mismatched={mismatched} errors={errors}", flush=True)
        finally:
            done_fh.flush()

    print(f"Done. matched={matched} mismatched={mismatched} errors={errors}")


if __name__ == "__main__":
    main()
