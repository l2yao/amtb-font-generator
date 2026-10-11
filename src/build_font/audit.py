import argparse
import glob
import os
import sys
from pathlib import Path

from build_font.glyph_names import SUPPORTED_EXTENSIONS, codepoint_from_filename


def iter_glyph_files(glyph_dir):
    paths = []
    for extension in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(glyph_dir, f"*{extension}")))
    return sorted(paths)


def read_target_chars(text=None, text_file=None):
    if text_file:
        value = Path(text_file).read_text(encoding="utf-8")
    else:
        value = text or ""
    return sorted({char for char in value if not char.isspace()})


def audit_glyph_dir(glyph_dir, target_text=None, target_file=None):
    available = {}
    invalid = []
    duplicates = []

    for path in iter_glyph_files(glyph_dir):
        try:
            codepoint = codepoint_from_filename(path)
        except ValueError as exc:
            invalid.append(str(exc))
            continue

        if codepoint in available:
            duplicates.append(
                f"Duplicate U+{codepoint:04X}: {available[codepoint]} and {path}"
            )
        available[codepoint] = path

    target_chars = read_target_chars(target_text, target_file)
    missing = [char for char in target_chars if ord(char) not in available]

    return {
        "available": available,
        "duplicates": duplicates,
        "invalid": invalid,
        "missing": missing,
        "target_count": len(target_chars),
    }


def print_audit_report(result):
    print(f"Valid glyphs: {len(result['available'])}")
    if result["target_count"]:
        print(f"Target characters: {result['target_count']}")
        print(f"Missing target glyphs: {len(result['missing'])}")
        if result["missing"]:
            print("Missing characters:")
            print("".join(result["missing"]))

    if result["duplicates"]:
        print("\nDuplicate glyph mappings:")
        for item in result["duplicates"]:
            print(f"- {item}")

    if result["invalid"]:
        print("\nInvalid glyph filenames:")
        for item in result["invalid"]:
            print(f"- {item}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Audit labeled glyph images before building a font.")
    parser.add_argument("--glyph_dir", required=True)
    parser.add_argument("--target_text", default=None, help="Characters the font should cover.")
    parser.add_argument("--target_file", default=None, help="UTF-8 text file containing target characters.")
    args = parser.parse_args()

    result = audit_glyph_dir(args.glyph_dir, args.target_text, args.target_file)
    print_audit_report(result)


if __name__ == "__main__":
    main()
