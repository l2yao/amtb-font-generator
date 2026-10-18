# -*- coding: utf-8 -*-
import sys, os, json, cv2
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r'C:\Users\Long\Documents\chinese-font-generator'
GDIR = os.path.join(ROOT, 'data', 'glyphs')
CDIR = os.path.join(ROOT, 'data', 'processed', 'components')
IDS  = os.path.join(ROOT, 'data', 'ids', 'IDS-UCS-Basic.txt')
MISS = os.path.join(ROOT, 'data', 'processed', 'missing_no_cands.json')
TMP  = r'C:\Users\Long\AppData\Local\Temp\opencode'

os.makedirs(CDIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

def parse_ids(path):
    m = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        p = line.split('\t')
        if len(p) < 3 or not p[0].startswith('U+'):
            continue
        try:
            ch = chr(int(p[0][2:], 16))
        except Exception:
            continue
        seq = p[2].split()[0] if p[2].split() else ''
        seq = seq.split('@apparent=')[0].strip()
        if not seq or '&' in seq:
            continue
        m[ch] = seq
    return m

IDS_MAP = parse_ids(IDS)
GLYPHS = set(os.path.splitext(f)[0] for f in os.listdir(GDIR) if f.endswith('.png'))

def load_mask(ch):
    p = os.path.join(GDIR, ch + '.png')
    if not os.path.exists(p):
        return None
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img < 128).astype(np.uint8)

def ink_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def tok_list(seq):
    out = []
    i = 0
    while i < len(seq):
        o = ord(seq[i])
        if 0x2FF0 <= o <= 0x2FFB:
            out.append(('op', o)); i += 1; continue
        out.append(('c', seq[i])); i += 1
    return out

def parse_tree(seq):
    toks = tok_list(seq)
    def build(pos):
        t = toks[pos]
        if t[0] == 'op':
            n = 3 if t[1] in (0x2FF2, 0x2FF3) else 2
            pos += 1
            kids = []
            for _ in range(n):
                node, pos = build(pos)
                kids.append(node)
            return (t[1], kids), pos
        return t[1], pos + 1
    try:
        tree, end = build(0)
        return tree if end == len(toks) else None
    except Exception:
        return None

def leaves(tree):
    if isinstance(tree, tuple):
        out = []
        for k in tree[1]:
            out.extend(leaves(k))
        return out
    return [tree]

# ---------------- splitting ----------------
def find_cut(prof, lo, hi, min_frac=0.0):
    seg = prof[lo:hi]
    if len(seg) == 0:
        return None
    best = None
    for i, v in enumerate(seg):
        dist = abs((lo + i) - (lo + len(seg)) / 2) / max(1, len(seg))
        key = v + dist * 40
        if best is None or key < best[0]:
            best = (key, lo + i)
    return best[1]

def prof_rank(crop, axis):
    prof = crop.sum(axis=axis).astype(float)
    lo, hi = int(len(prof) * 0.25), int(len(prof) * 0.75)
    if hi <= lo:
        return 1.0, None
    mx = max(prof.max(), 1)
    # locate zero-ink corridors in the central band
    sub = prof[lo:hi]
    zeros = sub <= 0
    runs = []
    i = 0
    while i < len(zeros):
        if zeros[i]:
            j = i
            while j < len(zeros) and zeros[j]:
                j += 1
            runs.append((lo + i, lo + j))
            i = j
        else:
            i += 1
    if runs:
        # most central corridor, prefer wider
        center = len(prof) / 2
        best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - center) - 0.001 * (r[1] - r[0]))
        cut = (best[0] + best[1]) // 2
        return 0.0, cut
    best_col = None
    best_val = None
    for i in range(lo, hi):
        if best_val is None or prof[i] < best_val:
            best_val = prof[i]
            best_col = i
    return best_val / mx, best_col

def split_lr(mask):
    H, W = mask.shape
    prof = mask.sum(axis=0)
    q, c = prof_rank(mask, axis=0)[:1][0], None
    r = prof_rank(mask, 0)
    qr, c = r
    if c is None:
        return None
    cutq = qr
    return mask[:, :c], mask[:, c:], cutq

def split_tb(mask):
    H, W = mask.shape
    prof = mask.sum(axis=1)
    qr, r = prof_rank(mask, 1)
    if r is None:
        return None
    return mask[:r, :], mask[r:, :], qr

# ---------------- library build ----------------
def clean_piece(mask):
    b = ink_bbox(mask)
    if b is None:
        return None
    x0, y0, x1, y1 = b
    pw, ph = x1 - x0, y1 - y0
    if pw < 8 or ph < 8:
        return None
    return mask[y0:y1, x0:x1], pw, ph

lib = {}   # comp char -> list of (piece_mask, entry_meta)

def harvest():
    for ch in sorted(GLYPHS):
        m = load_mask(ch)
        if m is None:
            continue
        b = ink_bbox(m)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        crop = m[y0:y1, x0:x1]
        h, w = crop.shape
        seq = IDS_MAP.get(ch)
        if not seq:
            continue
        tree = parse_tree(seq)
        if tree is None or not isinstance(tree, tuple):
            continue
        op, kids = tree
        if op not in (0x2FF0, 0x2FF1) or len(kids) != 2:
            continue
        if any(not isinstance(k, str) for k in kids):
            continue
        if op == 0x2FF0:
            res = split_lr(crop)
        else:
            res = split_tb(crop)
        if res is None:
            continue
        a0, a1, cutq = res
        p0 = clean_piece(a0)
        p1 = clean_piece(a1)
        if p0 is None or p1 is None:
            continue
        clean = bool(cutq == 0.0)
        for comp, (piece, pw, ph) in zip((kids[0], kids[1]), (p0, p1)):
            entry = dict(src=ch, op=op, nw=round(pw / w, 3), nh=round(ph / h, 3),
                         q=round(cutq, 3), clean=clean)
            lib.setdefault(comp, []).append((piece, entry))
    # save
    meta = {}
    for comp, entries in lib.items():
        entries.sort(key=lambda x: (not x[1]['clean'], x[1]['q']))
        d = os.path.join(CDIR, comp.encode('unicode_escape').decode())
        os.makedirs(d, exist_ok=True)
        meta[comp] = []
        for i, (piece, e) in enumerate(entries):
            ok, buf = cv2.imencode('.png', (1 - piece) * 255)
            buf.tofile(os.path.join(d, f'p{i}.png'))
            meta[comp].append(e)
    json.dump(meta, open(os.path.join(CDIR, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    return meta

if __name__ == '__main__':
    meta = harvest()
    only1 = [c for c, e in meta.items() if len(e) == 1]
    many = sorted(((c, len(e)) for c, e in meta.items()), key=lambda x: -x[1])
    print('components harvested:', len(meta))
    print('top components:', [(c, n) for c, n in many[:20]])
    print('sample plentiful comps:', ''.join(c for c, e in list(lib.items())[:60]))
