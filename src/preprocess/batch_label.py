"""Batch label unlabeled glyph crops with RapidOCR (CPU-friendly).

Strategy: single-character recognition per crop (no detection needed).
- High-confidence predictions are auto-accepted (with simplified->traditional fix-ups).
- Low-confidence / empty predictions go to a review queue.
- Run: python src/preprocess/batch_label.py --input_dir data/processed/glyphs_unlabeled
         --output_dir data/glyphs --review_list data/processed/review_queue.txt
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

CONF_THRESHOLD = 0.55

# Common simplified -> traditional corrections for this calligraphy set.
# Extend after reviewing the queue.
SIMP2TRAD = {
    "爱": "愛",
    "亿": "凡",  # RapidOCR misreads 凡; verify during review
}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Batch label glyph crops with RapidOCR.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--review_list", default="data/processed/review_queue.txt")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0, help="0 = all files")
    args = parser.parse_args()

    from rapidocr_onnxruntime import RapidOCR

    ocr = RapidOCR()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(Path(args.input_dir).glob("*.png"))
    if args.limit:
        files = files[: args.limit]
    print(f"Labeling {len(files)} crops...")

    accepted, queued = 0, 0
    with open(args.review_list, "w", encoding="utf-8") as review:
        for i, path in enumerate(files):
            try:
                res, _ = ocr(str(path))
            except Exception as exc:
                review.write(f"{path.name}\tERROR {exc}\n")
                queued += 1
                continue
            if not res or len(res[0][1]) != 1:
                review.write(f"{path.name}\tUNREADABLE {res}\n")
                queued += 1
                continue
            text, conf = res[0][1], float(res[0][2])
            text = SIMP2TRAD.get(text, text)
            if conf < args.conf:
                review.write(f"{path.name}\tLOWCONF {text} {conf:.2f}\n")
                queued += 1
                continue
            dest = Path(args.output_dir) / f"{text}.png"
            if dest.exists():
                # Keep first sample per character; extras stay in review for style picking.
                review.write(f"{path.name}\tDUPLICATE {text} {conf:.2f}\n")
                queued += 1
                continue
            shutil.copy(path, dest)
            accepted += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} accepted={accepted} queued={queued}")

    print(f"Done. accepted={accepted} queued_for_review={queued}")
    print(f"Review queue: {args.review_list}")


if __name__ == "__main__":
    main()
