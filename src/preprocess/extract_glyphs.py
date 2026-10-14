import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


OUTPUT_SIZE = 256
PADDING_RATIO = 0.15


def extract_glyphs_strip(image, output_size, text, stem, output_dir, image_path):
    img_array = np.array(image)
    h, w = img_array.shape
    img_area = h * w

    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = max(3, int(min(h, w) * 0.008))
    k = k | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = img_area * 0.0003
    max_area = img_area * 0.30
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 5 or ch < 5:
            continue
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 5:
            continue
        extent = area / (cw * ch) if cw * ch > 0 else 0
        if extent < 0.05:
            continue
        candidates.append([x, y, cw, ch, area, extent])

    # Merge nearby boxes (split components of one character). Applied only
    # for close-up strips where each character is large relative to the
    # image (e.g. data/train 6-char strips where 仁 splits into parts).
    # Dense multi-character pages skip merging: their inter-character gaps
    # are small and merging would fuse adjacent characters into columns.
    if len(candidates) > 1:
        areas_all = sorted([c[4] for c in candidates])
        median_area = areas_all[len(areas_all) // 2]
        wide_strip = w > h * 1.2
        close_up = median_area > img_area * 0.012
        do_merge = wide_strip or close_up
    else:
        do_merge = False
    if do_merge:
        dims = sorted([max(c[2], c[3]) for c in candidates])
        median_dim = dims[len(dims) // 2]
        merge_gap = max(5, int(median_dim * 0.15))
        changed = True
        while changed:
            changed = False
            used = [False] * len(candidates)
            merged = []
            for i in range(len(candidates)):
                if used[i]:
                    continue
                x0, y0, cw0, ch0 = candidates[i][:4]
                x1, y1 = x0 + cw0, y0 + ch0
                for j in range(i + 1, len(candidates)):
                    if used[j]:
                        continue
                    xa, ya, cwa, cha = candidates[j][:4]
                    xb, yb = xa + cwa, ya + cha
                    gap_x = max(0, max(x0, xa) - min(x1, xb))
                    gap_y = max(0, max(y0, ya) - min(y1, yb))
                    overlap_x = min(x1, xb) - max(x0, xa)
                    overlap_y = min(y1, yb) - max(y0, ya)
                    nearby = gap_x <= merge_gap and gap_y <= merge_gap
                    aligned = overlap_x > 0 or overlap_y > 0
                    small_gap = (gap_x + gap_y) <= max(cw0, ch0, cwa, cha) * 0.35
                    if nearby and (aligned or small_gap):
                        nx0, ny0 = min(x0, xa), min(y0, ya)
                        nx1, ny1 = max(x1, xb), max(y1, yb)
                        x0, y0, x1, y1 = nx0, ny0, nx1, ny1
                        cw0, ch0 = x1 - x0, y1 - y0
                        used[j] = True
                        changed = True
                merged.append([x0, y0, cw0, ch0, float(cw0 * ch0), 1.0])
                used[i] = True
            candidates = merged

    if not candidates:
        return extract_glyphs_single(image, output_size, text, stem, output_dir, image_path)

    areas = sorted([c[4] for c in candidates])
    median_a = areas[len(areas) // 2]
    filtered = [(x, y, cw, ch, area) for x, y, cw, ch, area, _ext in candidates if area >= median_a * 0.08]
    if not filtered:
        return []

    items = [(x, y, cw, ch, area, y + ch / 2) for x, y, cw, ch, area in filtered]
    is_tall = h > w * 1.5

    if is_tall:
        items.sort(key=lambda v: v[0])
        col_th = max([v[2] for v in items][len(items)//2], 10)
        cols = []
        cur_col = [items[0]]
        for item in items[1:]:
            if abs(item[0] - cur_col[-1][0]) <= col_th:
                cur_col.append(item)
            else:
                cur_col.sort(key=lambda v: v[5])
                cols.append(cur_col)
                cur_col = [item]
        if cur_col:
            cur_col.sort(key=lambda v: v[5])
            cols.append(cur_col)
        sorted_chars = []
        for col in cols:
            sorted_chars.extend(col)
    else:
        items.sort(key=lambda v: v[5])
        row_th = max([v[3] for v in items][len(items)//2], 10)
        rows = []
        cur_row = [items[0]]
        for item in items[1:]:
            if abs(item[5] - cur_row[-1][5]) <= row_th:
                cur_row.append(item)
            else:
                cur_row.sort(key=lambda v: v[0])
                rows.append(cur_row)
                cur_row = [item]
        if cur_row:
            cur_row.sort(key=lambda v: v[0])
            rows.append(cur_row)
        sorted_chars = []
        for row in rows:
            sorted_chars.extend(row)

    if text is not None and len(text) != len(sorted_chars):
        print(f"  Warning: text length ({len(text)}) != characters found ({len(sorted_chars)})")

    saved = []
    for idx, (x, y, cw, ch, area, yc) in enumerate(sorted_chars):
        pad = max(8, int(min(cw, ch) * PADDING_RATIO))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + cw + pad)
        y1 = min(h, y + ch + pad)
        char_img = image.crop((x0, y0, x1, y1))
        saved.append(_save_glyph(char_img, output_size, text, idx, stem, output_dir))

    return saved


def extract_glyphs_single(image, output_size, text, stem, output_dir, image_path):
    img_array = np.array(image)
    h, w = img_array.shape
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_rows = np.any(binary > 0, axis=1)
    ink_cols = np.any(binary > 0, axis=0)
    if np.any(ink_rows) and np.any(ink_cols):
        y1 = np.argmax(ink_rows)
        y2 = len(ink_rows) - np.argmax(ink_rows[::-1])
        x1 = np.argmax(ink_cols)
        x2 = len(ink_cols) - np.argmax(ink_cols[::-1])
        pad = max(8, int(min(x2 - x1, y2 - y1) * PADDING_RATIO))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        char_img = image.crop((x1, y1, x2, y2))
    else:
        char_img = image
    return [_save_glyph(char_img, output_size, text, 0, stem, output_dir)]


def extract_glyphs_page(image, output_size, text, stem, output_dir, image_path):
    img_array = np.array(image)
    h, w = img_array.shape
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink_rows = np.any(binary > 0, axis=1)
    ink_cols = np.any(binary > 0, axis=0)
    if not np.any(ink_rows) or not np.any(ink_cols):
        return extract_glyphs_strip(image, output_size, text, stem, output_dir, image_path)

    x_start = np.argmax(ink_cols)
    x_end = len(ink_cols) - np.argmax(ink_cols[::-1])
    content = binary[:, x_start:x_end]
    ch, cw = content.shape

    col_proj = np.sum(content > 0, axis=0)
    col_proj = np.convolve(col_proj, np.ones(max(3, cw//100))/max(3, cw//100), mode='same')
    gap_thresh = np.max(col_proj) * 0.03

    in_gap = False
    cuts = [0]
    for col in range(1, cw - 1):
        if not in_gap and col_proj[col] < gap_thresh:
            in_gap = True
            cuts.append(col)
        elif in_gap and col_proj[col] >= gap_thresh:
            in_gap = False
    cuts.append(cw)

    col_regions = [(cuts[i], cuts[i+1]) for i in range(0, len(cuts) - 1, 2) if cuts[i+1] - cuts[i] > 20]

    saved = []
    for col_start, col_end in col_regions:
        col_data = content[:, col_start:col_end]
        row_proj = np.sum(col_data > 0, axis=1)
        row_proj = np.convolve(row_proj, np.ones(max(3, ch//200))/max(3, ch//200), mode='same')
        row_gap = np.max(row_proj) * 0.03
        in_gap = False
        rcuts = [0]
        for row in range(1, ch - 1):
            if not in_gap and row_proj[row] < row_gap:
                in_gap = True
                rcuts.append(row)
            elif in_gap and row_proj[row] >= row_gap:
                in_gap = False
        rcuts.append(ch)

        for i in range(0, len(rcuts) - 1, 2):
            r1, r2 = rcuts[i], rcuts[i + 1]
            if r2 - r1 < 20:
                continue
            seg = col_data[r1:r2, :]
            ink_c = np.any(seg > 0, axis=0)
            ink_r = np.any(seg > 0, axis=1)
            if not np.any(ink_c) or not np.any(ink_r):
                continue
            cx1 = x_start + col_start + np.argmax(ink_c)
            cx2 = x_start + col_start + len(ink_c) - np.argmax(ink_c[::-1])
            cy1 = r1 + np.argmax(ink_r)
            cy2 = r1 + len(ink_r) - np.argmax(ink_r[::-1])
            if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                continue
            pad = max(8, int(min(cx2 - cx1, cy2 - cy1) * PADDING_RATIO))
            cx1 = max(0, cx1 - pad)
            cy1 = max(0, cy1 - pad)
            cx2 = min(w, cx2 + pad)
            cy2 = min(h, cy2 + pad)
            char_img = image.crop((cx1, cy1, cx2, cy2))
            idx = len(saved)
            saved.append(_save_glyph(char_img, output_size, text, idx, stem, output_dir))

    return saved


def _save_glyph(char_img, output_size, text, idx, stem, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    side = max(char_img.width, char_img.height)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(char_img, ((side - char_img.width) // 2, (side - char_img.height) // 2))
    # Extra white margin: edge-touching crops (e.g. last char in a strip)
    # would otherwise render clipped after vectorization.
    margin = max(2, int(side * 0.06))
    bordered = Image.new("L", (side + 2 * margin, side + 2 * margin), 255)
    bordered.paste(canvas, (margin, margin))
    canvas = bordered.resize((output_size, output_size), Image.Resampling.LANCZOS)

    if text is not None and idx < len(text):
        char = text[idx]
        filename = f"{char}.png" if char.strip() else f"{stem}_{idx:04d}.png"
    else:
        filename = f"{stem}_{idx:04d}.png"

    out_path = os.path.join(output_dir, filename)
    canvas.save(out_path)
    return out_path


def remove_red_seals(rgb_image):
    """Paint red seal pixels white so seals are not extracted as glyphs."""
    arr = np.array(rgb_image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return rgb_image
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    red_mask = (r > 110) & (r - g > 30) & (r - b > 30)
    if np.any(red_mask):
        arr[red_mask] = (255, 255, 255)
        return Image.fromarray(arr)
    return rgb_image


def extract_glyphs(image_path, output_dir, output_size=OUTPUT_SIZE, text=None, mode="strip"):
    rgb = Image.open(image_path).convert("RGB")
    rgb = remove_red_seals(rgb)
    image = ImageOps.autocontrast(rgb.convert("L"))
    stem = Path(image_path).stem

    if mode == "single":
        return extract_glyphs_single(image, output_size, text, stem, output_dir, image_path)
    elif mode == "page":
        return extract_glyphs_page(image, output_size, text, stem, output_dir, image_path)
    else:
        return extract_glyphs_strip(image, output_size, text, stem, output_dir, image_path)


def glyph_extract_main(args):
    if os.path.isdir(args.image_path):
        image_files = sorted(Path(args.image_path).rglob("*"))
        image_files = [p for p in image_files if p.suffix.lower() in
                       ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
    else:
        image_files = [Path(args.image_path)]

    if not image_files:
        print("No image files found.")
        return

    text = None
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8").strip()
        text = ''.join(text.split())
    elif args.text:
        text = ''.join(args.text.split())

    total = 0
    for img_idx, img_path in enumerate(image_files):
        print(f"Processing: {img_path}")
        saved = extract_glyphs(
            str(img_path), args.output_dir,
            output_size=args.output_size,
            text=text,
            mode=args.mode,
        )
        if text and len(saved) != len(text) and img_idx == 0:
            print(f"  Note: extracted {len(saved)} chars but text has {len(text)} chars")
        for fname in saved:
            print(f"  Saved: {Path(fname).name}")
        total += len(saved)

    print(f"\nTotal glyphs extracted: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract character glyphs from calligraphy images."
    )
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_size", type=int, default=OUTPUT_SIZE)
    parser.add_argument("--mode", choices=["strip", "single", "page"], default="strip",
                        help="strip: multi-char strips (default), single: each image is one char, page: full page scans")
    text_group = parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="Text string for naming glyphs in reading order")
    text_group.add_argument("--text_file", help="UTF-8 text file for naming glyphs")
    args = parser.parse_args()
    glyph_extract_main(args)
