"""Graft calligraphy glyphs into a full CJK fallback font.

Instead of table-level merging, this overwrites the fallback's outlines
for the calligraphy codepoints with ours. Result: one font where covered
characters render in handwriting and everything else falls back cleanly.
Name table is set to the calligraphy family; OS/2 ranges stay valid.

Run: python src/build_font/merge_fallback.py \
        --calligraphy_ttf output/calligraphy-demo.ttf \
        --fallback_ttf data/fonts/NotoSansSC-Regular.ttf \
        --output_ttf output/calligraphy-full.ttf \
        --family_name "My Calligraphy"
"""
import argparse
import re
import sys


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Graft calligraphy glyphs into CJK fallback.")
    parser.add_argument("--calligraphy_ttf", required=True)
    parser.add_argument("--fallback_ttf", required=True)
    parser.add_argument("--output_ttf", required=True)
    parser.add_argument("--family_name", default="My Calligraphy")
    parser.add_argument("--style_name", default="Regular")
    args = parser.parse_args()

    from fontTools.ttLib import TTFont

    calli = TTFont(args.calligraphy_ttf)
    base = TTFont(args.fallback_ttf)

    calli_cmap = calli.getBestCmap()
    base_cmap = base.getBestCmap()
    codepoints = [cp for cp in calli_cmap if cp != 0x20 and cp in base_cmap]
    missing = [cp for cp in calli_cmap if cp != 0x20 and cp not in base_cmap]
    if missing:
        print("WARNING: fallback lacks: " + " ".join(f"U+{cp:04X}" for cp in missing))

    for cp in codepoints:
        src_name = calli_cmap[cp]
        dst_name = base_cmap[cp]
        glyph = calli["glyf"][src_name]
        glyph.recalcBounds(base["glyf"])
        base["glyf"][dst_name] = glyph
        base["hmtx"][dst_name] = calli["hmtx"][src_name]

    base["name"].setName(args.family_name, 1, 3, 1, 0x409)
    base["name"].setName(args.style_name, 2, 3, 1, 0x409)
    base["name"].setName(f"{args.family_name} {args.style_name}", 4, 3, 1, 0x409)
    ps_name = re.sub(r"[^A-Za-z0-9-]", "", f"{args.family_name}-{args.style_name}")
    base["name"].setName(ps_name, 6, 3, 1, 0x409)
    base["name"].setName(args.family_name, 1, 1, 0, 0)
    base["name"].setName(args.style_name, 2, 1, 0, 0)
    base["name"].setName(f"{args.family_name} {args.style_name}", 4, 1, 0, 0)
    base["name"].setName(ps_name, 6, 1, 0, 0)
    base["OS/2"].fsSelection |= 1 << 6  # REGULAR

    base.save(args.output_ttf)
    print(f"Saved {args.output_ttf}: {len(base.getBestCmap())} mapped chars, "
          f"calligraphy glyphs grafted {len(codepoints)}.")


if __name__ == "__main__":
    main()
