import re
from pathlib import Path


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def codepoint_from_filename(path):
    stem = Path(path).stem
    if len(stem) == 1:
        return ord(stem)

    normalized = stem.upper()
    match = re.fullmatch(r"(?:UNI|U\+|0X)?([0-9A-F]{4,6})", normalized)
    if match:
        codepoint = int(match.group(1), 16)
        if 0 <= codepoint <= 0x10FFFF:
            return codepoint

    raise ValueError(
        f"Cannot infer Unicode code point from '{Path(path).name}'. "
        "Use a one-character filename, uni4F60.png, U+4F60.png, or 4F60.png."
    )


def glyph_name_for_codepoint(codepoint):
    if codepoint <= 0xFFFF:
        return f"uni{codepoint:04X}"
    return f"u{codepoint:06X}"
