import argparse
import glob
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageOps

from build_font.glyph_names import (
    SUPPORTED_EXTENSIONS,
    codepoint_from_filename,
    glyph_name_for_codepoint,
)


DEFAULT_UNITS_PER_EM = 1000
DEFAULT_ASCENT = 880
DEFAULT_DESCENT = -120


def load_ink_mask(image_path, threshold=210):
    image = Image.open(image_path).convert("L")
    image = ImageOps.autocontrast(image)
    gray = np.array(image)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def mask_content_bounds(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return x, y, x + width, y + height


def contour_area(contour):
    return abs(cv2.contourArea(contour))


def draw_contours_to_pen(mask, pen, units_per_em, ascent, descent, simplify):
    bounds = mask_content_bounds(mask)
    if bounds is None:
        return 0

    x0, y0, x1, y1 = bounds
    content_width = max(1, x1 - x0)
    content_height = max(1, y1 - y0)
    glyph_height = ascent - descent
    scale = min(units_per_em * 0.82 / content_width, glyph_height * 0.82 / content_height)
    x_offset = (units_per_em - content_width * scale) / 2
    y_offset = descent + (glyph_height - content_height * scale) / 2

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(4.0, mask.shape[0] * mask.shape[1] * 0.00008)
    drawn = 0

    for contour in contours:
        if contour_area(contour) < min_area:
            continue

        epsilon = max(0.5, simplify * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue

        start_x, start_y = points[0]
        pen.moveTo(
            (
                round((start_x - x0) * scale + x_offset),
                round(ascent - ((start_y - y0) * scale + (ascent - y_offset - content_height * scale))),
            )
        )

        for px, py in points[1:]:
            pen.lineTo(
                (
                    round((px - x0) * scale + x_offset),
                    round(ascent - ((py - y0) * scale + (ascent - y_offset - content_height * scale))),
                )
            )
        pen.closePath()
        drawn += 1

    return drawn


def iter_glyph_images(glyph_dir):
    paths = []
    for extension in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(glyph_dir, f"*{extension}")))
    return sorted(paths)


def images_to_ttf(
    glyph_dir,
    output_ttf_path,
    family_name="Generated Chinese Calligraphy",
    style_name="Regular",
    units_per_em=DEFAULT_UNITS_PER_EM,
    ascent=DEFAULT_ASCENT,
    descent=DEFAULT_DESCENT,
    threshold=210,
    simplify=0.002,
):
    glyph_paths = iter_glyph_images(glyph_dir)
    if not glyph_paths:
        raise FileNotFoundError(f"No glyph images found in {glyph_dir}")

    cmap = {}
    glyphs = {}
    metrics = {}
    glyph_order = [".notdef", "space"]

    empty_pen = TTGlyphPen(None)
    glyphs[".notdef"] = empty_pen.glyph()
    metrics[".notdef"] = (units_per_em, 0)
    glyphs["space"] = TTGlyphPen(None).glyph()
    metrics["space"] = (units_per_em, 0)
    cmap[0x20] = "space"

    skipped = []
    for glyph_path in glyph_paths:
        try:
            codepoint = codepoint_from_filename(glyph_path)
        except ValueError as exc:
            skipped.append(str(exc))
            continue

        glyph_name = glyph_name_for_codepoint(codepoint)
        mask = load_ink_mask(glyph_path, threshold=threshold)
        pen = TTGlyphPen(None)
        contour_count = draw_contours_to_pen(mask, pen, units_per_em, ascent, descent, simplify)
        if contour_count == 0:
            skipped.append(f"Skipped blank glyph image: {glyph_path}")
            continue

        cmap[codepoint] = glyph_name
        glyphs[glyph_name] = pen.glyph()
        metrics[glyph_name] = (units_per_em, 0)
        glyph_order.append(glyph_name)

    if len(glyph_order) <= 2:
        details = "\n".join(skipped[:10])
        raise ValueError(f"No usable glyphs were found. {details}")

    output_path = Path(output_ttf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fb = FontBuilder(units_per_em, isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ascent, descent=descent)
    fb.setupOS2(
        sTypoAscender=ascent,
        sTypoDescender=descent,
        usWinAscent=ascent,
        usWinDescent=abs(descent),
    )
    fb.setupNameTable(
        {
            "familyName": family_name,
            "styleName": style_name,
            "uniqueFontIdentifier": f"{family_name} {style_name}",
            "fullName": f"{family_name} {style_name}",
            "psName": re.sub(r"[^A-Za-z0-9-]", "", f"{family_name}-{style_name}"),
            "version": "Version 1.000",
        }
    )
    fb.setupPost()
    fb.setupMaxp()
    fb.save(str(output_path))

    return {
        "font_path": str(output_path),
        "glyph_count": len(glyph_order) - 2,
        "skipped": skipped,
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build a TrueType font from glyph images named by Chinese character or Unicode code point."
    )
    parser.add_argument("--glyph_dir", required=True, help="Directory containing glyph images.")
    parser.add_argument("--output_ttf", required=True, help="Output .ttf path.")
    parser.add_argument("--family_name", default="Generated Chinese Calligraphy")
    parser.add_argument("--style_name", default="Regular")
    parser.add_argument("--threshold", type=int, default=210, help="Ink threshold, 0-255.")
    parser.add_argument("--simplify", type=float, default=0.002, help="Contour simplification ratio.")
    args = parser.parse_args()

    result = images_to_ttf(
        glyph_dir=args.glyph_dir,
        output_ttf_path=args.output_ttf,
        family_name=args.family_name,
        style_name=args.style_name,
        threshold=args.threshold,
        simplify=args.simplify,
    )
    print(f"Saved {result['font_path']} with {result['glyph_count']} glyphs.")
    for warning in result["skipped"][:20]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
