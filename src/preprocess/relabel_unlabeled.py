"""Relabel unlabeled crops from manifest_tmp order grids.

Replays NW alignment for each processed page's kept order folder, and emits
crop->char labels where OCR failed (empty or wrong prediction). Those labels
are positional inferences, so they land in a separate pool for visual review:
    data/processed/candidates2/<char>/

Run: python src/preprocess/relabel_unlabeled.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess.manifest_extract import (
    load_manifest,
    clean_text,
    ocr_crops,
    align_nw,
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manifest = load_manifest("data/amtb_raw/manifest.jsonl")
    tmp_root = Path("data/processed/manifest_tmp")
    out_root = Path("data/processed/candidates2")
    out_root.mkdir(parents=True, exist_ok=True)

    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    pages = sorted(p for p in tmp_root.iterdir() if p.is_dir())
    print(f"{len(pages)} pages to relabel", flush=True)

    n_label = 0
    for pi, page in enumerate(pages):
        stem = page.name
        entry = None
        for fp, it in manifest.items():
            if Path(fp).stem == stem:
                entry = it
                break
        text = clean_text(entry.get("content")) if entry else ""
        if not text:
            continue
        order_dirs = sorted(page.glob("o*"))
        best = None
        for od in order_dirs:
            crops = sorted(od.glob("*.png"))
            if not crops:
                continue
            preds = ocr_crops(ocr, [str(c) for c in crops])
            pairs, ratio = align_nw(preds, list(text))
            gaps = sum(1 for a, b in pairs if a is None or b is None)
            key = (ratio, -gaps)
            if best is None or key > best[0]:
                best = (key, crops, pairs, preds)
        if best is None:
            continue
        _, crops, pairs, preds = best
        for a, b in pairs:
            if a is None or b is None:
                continue
            pred = preds[a][0]
            if pred == text[b]:
                continue  # already harvested as exact match earlier
            ch = text[b]
            ch_dir = out_root / ch
            ch_dir.mkdir(parents=True, exist_ok=True)
            name = f"{stem}_{od.name}_{a:04d}.png"
            dst = ch_dir / name
            if not dst.exists():
                import shutil
                shutil.copy(crops[a], dst)
            n_label += 1
        if (pi + 1) % 50 == 0:
            print(f"  {pi + 1}/{len(pages)} proposals so far: {n_label}", flush=True)

    counts = {d: len(os.listdir(os.path.join(out_root, d))) for d in os.listdir(out_root)}
    print(f"done. proposals: {n_label} into {len(counts)} char dirs")
    missing = json.load(open("data/processed/missing_no_cands.json", encoding="utf-8"))
    hit = [ch for ch in missing if ch in counts]
    print(f"missing targets covered by relabel pool: {len(hit)}")
    print("sample:", "".join(sorted(hit)[:80]))
    json.dump(hit, open("data/processed/relabel_hits.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)


if __name__ == "__main__":
    main()
