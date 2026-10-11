# Chinese Font Generator

This project takes labeled Chinese calligraphy glyph images and builds a real TrueType font (`.ttf`).

The default workflow is intentionally small:

1. Put one glyph image per character in a folder.
2. Audit filenames and target coverage.
3. Build the font.

LLMs are best used as an assistant around this deterministic core: preparing a target character list, spotting missing glyphs, suggesting filename fixes, creating review checklists, and optionally driving a cloud image model for draft glyphs. The actual font compilation stays local and repeatable.

## Install

```bash
pip install -r requirements.txt
```

Default dependencies:

- `Pillow`
- `fonttools`
- `opencv-python`

Optional dependencies:

```bash
pip install -r requirements-ocr.txt
pip install -r requirements-ai.txt
```

OCR and AI/model dependencies are not required for the basic font builder.

## Glyph Filename Rules

Each image filename defines the Unicode character it represents. Supported forms:

- `你.png`
- `uni4F60.png`
- `U+4F60.png`
- `4F60.png`

Images with invalid names are skipped by the builder and reported by the audit command.

## Audit Glyphs

Check a glyph folder:

```bash
python src/generate_font.py audit --glyph_dir data/glyphs
```

Check coverage against target text:

```bash
python src/generate_font.py audit ^
  --glyph_dir data/glyphs ^
  --target_text "你好中文"
```

Or use a UTF-8 target file:

```bash
python src/generate_font.py audit ^
  --glyph_dir data/glyphs ^
  --target_file data/target_chars.txt
```

The missing-character output is suitable to paste into an LLM or image-generation workflow as the next glyph request list.

## Build a TTF

```bash
python src/generate_font.py build ^
  --glyph_dir data/glyphs ^
  --output_ttf output/calligraphy.ttf ^
  --family_name "My Calligraphy"
```

You can also call the builder directly:

```bash
python src/build_font/vectorize.py ^
  --glyph_dir data/glyphs ^
  --output_ttf output/calligraphy.ttf
```

## Optional OCR Extraction

If you want to crop and label characters from scanned pages, install OCR dependencies first:

```bash
pip install -r requirements-ocr.txt
```

Then run:

```bash
python src/generate_font.py ocr-extract ^
  --image_path data/samples ^
  --output_dir data/glyphs
```

Review OCR output carefully before building the font. A wrong filename creates a wrong Unicode mapping.

## Optional AI Generation

The repo keeps `src/engine/run_zi2zi.py` as an experimental adapter for local model-based glyph generation, but it is not part of the default path.

For a leaner LLM-centered product, prefer this loop:

1. Build or audit the current glyph set.
2. Use the missing-character report as the request list.
3. Ask an LLM/image model to create draft glyph images in the same style.
4. Save approved drafts using Unicode filenames.
5. Re-run `audit`, then `build`.

This keeps the font builder simple while allowing AI help where it is useful.
