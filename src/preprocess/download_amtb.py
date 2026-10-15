"""Download all calligraphy (墨寶) originals from https://www.amtb.tw/#/moIndex.

Discovers the 11 categories via GET /v2/mb, paginates
GET /v2/mb/{menuid}?page=&per_page=, and downloads each item's original
`jpg` file (NOT the _preview thumbnail).

Each item also carries `title` + `content` transcriptions, saved to a
manifest (manifest.jsonl) for future auto-labeling.

Usage:
    python src/preprocess/download_amtb.py --output_dir data/amtb_raw
    python src/preprocess/download_amtb.py --output_dir data/amtb_raw --categories MB-04 MB-03
    python src/preprocess/download_amtb.py --output_dir data/amtb_raw --overwrite
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://www.amtb.tw"
UA = {"User-Agent": "Mozilla/5.0 (amtb-font-generator downloader)"}


def api_get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def download_file(url, dest, retries=3):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(256 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            if dest.stat().st_size > 0:
                return True
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"    retry {attempt}/{retries} {url}: {exc}", flush=True)
            time.sleep(2 * attempt)
    return False


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Download amtb.tw 墨寶 originals.")
    parser.add_argument("--output_dir", default="data/amtb_raw")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="menuid subset, e.g. MB-04 MB-03 (default: all)")
    parser.add_argument("--per_page", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.2,
                        help="seconds between image downloads")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel download threads")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching category index...")
    index = api_get(BASE + "/v2/mb")
    categories = index["data"]["mbcategorys"]
    if args.categories:
        wanted = set(args.categories)
        categories = [c for c in categories if c["menuid"] in wanted]
    print(f"{len(categories)} categories: " +
          ", ".join(c["menuid"] + " " + c["category"] for c in categories))

    manifest_path = out_dir / "manifest.jsonl"
    manifest = open(manifest_path, "a", encoding="utf-8")
    manifest_lock = threading.Lock()

    def fetch_item(item, menuid, category):
        url = item.get("jpg", "")
        name = url.rsplit("/", 1)[-1] if url else f"{item.get('id')}.jpg"
        dest = out_dir / menuid / name
        if dest.exists() and dest.stat().st_size > 0 and not args.overwrite:
            return ("skip", None)
        if not url:
            return ("no-url", None)
        ok = download_file(url, dest)
        entry = {
            "id": item.get("id"),
            "menuid": menuid,
            "category": category,
            "title": item.get("title"),
            "content": item.get("content"),
            "jpg": url,
            "file": str(dest),
            "status": "ok" if ok else "fail",
        }
        with manifest_lock:
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
        time.sleep(args.delay)
        return ("ok" if ok else "fail", entry)

    total_new = total_skip = total_fail = 0
    try:
        for cat in categories:
            menuid = cat["menuid"]
            cat_dir = out_dir / menuid
            cat_dir.mkdir(parents=True, exist_ok=True)

            page, cat_total, cat_new = 1, None, 0
            while True:
                data = api_get(f"{BASE}/v2/mb/{menuid}?page={page}&per_page={args.per_page}")
                mblists = data["data"]["mblists"]
                items = mblists["data"]
                cat_total = mblists["total"]
                if not items:
                    break
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    results = list(pool.map(
                        lambda it: fetch_item(it, menuid, cat.get("category")),
                        items,
                    ))
                for status, _entry in results:
                    if status == "ok":
                        total_new += 1
                        cat_new += 1
                    elif status == "skip":
                        total_skip += 1
                    else:
                        total_fail += 1
                manifest.flush()
                print(f"  [{menuid}] page {page}: {len(items)} items "
                      f"(total {cat_total}, new {cat_new})", flush=True)
                if len(items) < args.per_page:
                    break
                page += 1
    finally:
        manifest.close()

    print(f"Done. new={total_new} skipped={total_skip} failed={total_fail}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
