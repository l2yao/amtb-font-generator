import argparse
import sys

from build_font.audit import audit_glyph_dir, print_audit_report


def configure_stdout():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def main():
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Build and audit Chinese calligraphy fonts from labeled glyph images."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Validate glyph filenames and optional target coverage.")
    audit_parser.add_argument("--glyph_dir", required=True, help="Directory containing glyph images.")
    audit_parser.add_argument("--target_text", default=None, help="Characters the font should cover.")
    audit_parser.add_argument("--target_file", default=None, help="UTF-8 text file containing target characters.")

    build_parser = subparsers.add_parser("build", help="Build a TTF from labeled glyph images.")
    build_parser.add_argument("--glyph_dir", required=True, help="Directory containing glyph images.")
    build_parser.add_argument("--output_ttf", required=True, help="Output .ttf path.")
    build_parser.add_argument("--family_name", default="Generated Chinese Calligraphy")
    build_parser.add_argument("--style_name", default="Regular")
    build_parser.add_argument("--threshold", type=int, default=210)
    build_parser.add_argument("--simplify", type=float, default=0.002)

    ocr_parser = subparsers.add_parser(
        "ocr-extract",
        help="Optional: crop and label characters from sample images with PaddleOCR.",
    )
    ocr_parser.add_argument("--image_path", required=True, help="Input JPG/PNG file or directory.")
    ocr_parser.add_argument("--output_dir", required=True, help="Directory for extracted character PNG files.")

    glyph_parser = subparsers.add_parser(
        "glyph-extract",
        help="Extract individual glyph images from calligraphy images.",
    )
    glyph_parser.add_argument("--image_path", required=True, help="Input image file or directory.")
    glyph_parser.add_argument("--output_dir", required=True, help="Output directory for glyph PNGs.")
    glyph_parser.add_argument("--output_size", type=int, default=256, help="Output glyph size in pixels (default 256).")
    glyph_parser.add_argument("--mode", choices=["strip", "single", "page"], default="strip",
                              help="strip: multi-char strips (default), single: each image is one char, page: full page scans")
    text_group = glyph_parser.add_mutually_exclusive_group()
    text_group.add_argument("--text", help="Text string to name extracted glyphs in reading order.")
    text_group.add_argument("--text_file", help="UTF-8 text file to name extracted glyphs.")

    args = parser.parse_args()

    if args.command == "audit":
        result = audit_glyph_dir(args.glyph_dir, args.target_text, args.target_file)
        print_audit_report(result)
        return

    if args.command == "ocr-extract":
        from preprocess.extract_grid import extract_characters

        extract_characters(args.image_path, args.output_dir)
        return

    if args.command == "glyph-extract":
        from preprocess.extract_glyphs import glyph_extract_main

        glyph_extract_main(args)
        return

    from build_font.vectorize import images_to_ttf

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
