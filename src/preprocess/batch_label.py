"""Batch label unlabeled glyph crops with RapidOCR recognition-only.

Each crop holds one centered character, so detection is skipped
(use_det=False): faster (~0.2s/crop) and more accurate than full OCR.
- Best-confidence sample per character is kept in output_dir.
- Everything else (low confidence, duplicates) goes to the review queue.
- Resumable via a done-list file; safe to relaunch.

Run: python src/preprocess/batch_label.py --input_dir data/processed/glyphs_unlabeled
         --output_dir data/glyphs --workers 3
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

CONF_THRESHOLD = 0.55

# Common corrections for this calligraphy set. Extend after review.
# Common corrections for this calligraphy set. Extend after review.
SIMP2TRAD = {
    "爱": "愛",
}


ILLEGAL_FILENAME_CHARS = set('<>:"/\\|?*')


def _is_safe_char(text):
    if len(text) != 1:
        return False
    if text in ILLEGAL_FILENAME_CHARS:
        return False
    if ord(text) < 32 or ord(text) == 127:
        return False
    return True


def _is_han_char(text):
    """Only Han ideographs belong in the font: no digits, Latin,
    punctuation, symbols, bopomofo, radicals, or compatibility forms."""
    cp = ord(text)
    return (
        (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
        or (0x4E00 <= cp <= 0x9FFF)  # CJK Unified Ideographs
        or (0xF900 <= cp <= 0xFAFF)  # CJK Compatibility Ideographs
        or (0x20000 <= cp <= 0x3134F)  # CJK Extensions B-G
    )


def _worker_init():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR

    _OCR = RapidOCR()


def _recognize(path_str):
    try:
        res, _ = _OCR(path_str, use_det=False, use_cls=False)
    except Exception as exc:  # noqa: BLE001
        return (path_str, "error", "", 0.0, str(exc))
    if not res or not res[0] or len(res[0]) < 2:
        return (path_str, "unreadable", "", 0.0, "")
    text, conf = res[0][0], float(res[0][1])
    if len(text) != 1:
        return (path_str, "multichar", text, conf, "")
    text = SIMP2TRAD.get(text, text)
    return (path_str, "ok", text, conf, "")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Batch label glyph crops.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--review_list", default="data/processed/review_queue.txt")
    parser.add_argument("--done_list", default="data/processed/label_done.txt")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    if os.path.exists(args.done_list):
        with open(args.done_list, encoding="utf-8") as fh:
            done = {line.strip() for line in fh if line.strip()}

    files = sorted(str(p) for p in Path(args.input_dir).glob("*.png"))
    files = [f for f in files if Path(f).name not in done]
    if args.limit:
        files = files[: args.limit]
    print(f"Labeling {len(files)} crops ({len(done)} already done)...", flush=True)

    # Current best confidence per accepted character.
    best = {}
    for existing in out_dir.glob("*.png"):
        best[existing.stem] = 1.01  # keep hand-verified labels untouched

    accepted = queued = 0
    review_fh = open(args.review_list, "a", encoding="utf-8")
    done_fh = open(args.done_list, "a", encoding="utf-8")

    def handle(result):
        nonlocal accepted, queued
        try:
            path_str, status, text, conf, extra = result
            name = Path(path_str).name
            if status != "ok":
                review_fh.write(f"{name}\t{status.upper()} {text} {extra}\n")
                queued += 1
            elif not _is_safe_char(text):
                review_fh.write(f"{name}\tILLEGAL {repr(text)} {conf:.2f}\n")
                queued += 1
            elif not _is_han_char(text):
                review_fh.write(f"{name}\tNONHAN {text} U+{ord(text):04X} {conf:.2f}\n")
                queued += 1
            elif conf < args.conf:
                review_fh.write(f"{name}\tLOWCONF {text} {conf:.2f}\n")
                queued += 1
            elif text in best:
                if conf > best[text]:
                    shutil.copy(path_str, out_dir / f"{text}.png")
                    best[text] = conf
                    review_fh.write(f"{name}\tREPLACE {text} {conf:.2f}\n")
                    accepted += 1
                else:
                    review_fh.write(f"{name}\tDUPLICATE {text} {conf:.2f}\n")
                    queued += 1
            else:
                shutil.copy(path_str, out_dir / f"{text}.png")
                best[text] = conf
                accepted += 1
        except Exception as exc:  # noqa: BLE001 - never let one file kill the run
            review_fh.write(f"{Path(result[0]).name}\tHANDLER_ERROR {exc}\n")
            queued += 1
        finally:
            try:
                done_fh.write(Path(result[0]).name + "\n")
            except Exception:
                pass

    try:
        with mp.Pool(args.workers, initializer=_worker_init) as pool:
            for i, result in enumerate(pool.imap_unordered(_recognize, files, chunksize=8)):
                handle(result)
                if (i + 1) % 500 == 0:
                    done_fh.flush()
                    review_fh.flush()
                    print(f"  {i + 1}/{len(files)} accepted={accepted} queued={queued}", flush=True)
    finally:
        review_fh.close()
        done_fh.close()

    print(f"Done. accepted={accepted} queued_for_review={queued}")
    print(f"Unique characters in {args.output_dir}: {len(best)}")


if __name__ == "__main__":
    main()
