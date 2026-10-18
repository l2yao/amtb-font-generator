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

# load library
LIB = json.load(open(os.path.join(CDIR, 'meta.json'), encoding='utf-8'))  # comp -> entries

_comp_cache = {}
FRESH = False

def load_comp_piece(comp, idx=0):
    key = (comp, idx)
    if key in _comp_cache:
        return _comp_cache[key]
    d = os.path.join(CDIR, comp.encode('unicode_escape').decode())
    p = os.path.join(d, f'p{idx}.png')
    if not os.path.exists(p):
        return None
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = (img < 128).astype(np.uint8)
    _comp_cache[key] = mask
    return mask

def load_mask(ch):
    p = os.path.join(GDIR, ch + '.png')
    if not os.path.exists(p):
        return None
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return (img < 128).astype(np.uint8)

def ink_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def crop_ink(mask):
    b = ink_bbox(mask)
    if b is None:
        return None
    return mask[b[1]:b[3], b[0]:b[2]]

def tok_list(seq):
    out = []
    for ch in seq:
        o = ord(ch)
        if 0x2FF0 <= o <= 0x2FFB:
            out.append(('op', o))
        else:
            out.append(('c', ch))
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

def flat_ops(tree, out):
    if isinstance(tree, tuple):
        out.append(tree[0])
        for k in tree[1]:
            flat_ops(k, out)
    return out

memo = {}
def comp_status(ch, depth=0):
    """'glyph' | 'lib' | 'build' | None"""
    if depth > 3:
        return None
    if ch in memo:
        return memo[ch]
    st = None
    if ch in GLYPHS:
        st = 'glyph'
    elif ch in LIB:
        st = 'lib'
    else:
        seq = IDS_MAP.get(ch)
        if seq:
            tree = parse_tree(seq)
            if tree is not None:
                ops = flat_ops(tree, [])
                if all(o in (0x2FF0, 0x2FF1) for o in ops):
                    if all(comp_status(k, depth + 1) for k in leaves(tree)):
                        st = 'build'
    memo[ch] = st
    return st

def leaves(tree):
    if isinstance(tree, tuple):
        out = []
        for k in tree[1]:
            out.extend(leaves(k))
        return out
    return [tree]

def scale_to(mask, target_h=None, target_w=None):
    h, w = mask.shape
    if target_h:
        s = target_h / h
        tw = max(1, int(round(w * s)))
        return cv2.resize(mask.astype(np.float32), (tw, target_h), interpolation=cv2.INTER_AREA) > 0.5
    s = target_w / w
    th = max(1, int(round(h * s)))
    return cv2.resize(mask.astype(np.float32), (target_w, th), interpolation=cv2.INTER_AREA) > 0.5

def canvas_placement(places, canvas=256, margin=14):
    # places: list of (mask, cx_center_frac unused) -> we place later
    pass

def assemble(ch, depth=0, fresh=False):
    mask = None if fresh else load_mask(ch)
    if mask is not None:
        return crop_ink(mask)
    seq = IDS_MAP.get(ch)
    if not seq:
        return None
    tree = parse_tree(seq)
    if tree is None or not isinstance(tree, tuple):
        return None
    op, kids = tree
    if op in (0x2FF0, 0x2FF1):
        return compose_lr_tb(op, kids, depth + 1)
    return None

def comp_medians(ch, op=None):
    ents = LIB.get(ch) or []
    def pick(o):
        es = [e for e in ents if e.get('op') == o] or ents
        return es
    es = pick(op) if op is not None else ents
    nhs = [e.get('nh') for e in es if e.get('nh')]
    nws = [e.get('nw') for e in es if e.get('nw')]
    mh = float(np.median(nhs)) if nhs else 1.0
    mw = float(np.median(nws)) if nws else 1.0
    return mh, mw

def part_mask_ex(ch, depth=0, top=None):
    """returns (mask, nh median height frac, nw median width frac)"""
    if isinstance(ch, tuple):
        ser = ser_node(ch)
        comp = INV_TREE.get(ser)
        m = compose_node(ch, depth)
        if m is None:
            return None, None, None
        if comp:
            mh, mw = comp_medians(comp, top or (LIB.get(comp) or [{}])[0].get('op'))
            return m, mh, mw
        return m, 1.0, 1.0
    m = load_mask(ch)
    if m is not None:
        cm = crop_ink(m)
        if cm is None:
            return None, None, None
        return cm, 1.0, 1.0
    # lib piece: pick best-quality source
    ents = LIB.get(ch)
    if ents:
        mh, mw = comp_medians(ch, top or (ents[0].get('op')))
        for i, e in enumerate(ents):
            cp = load_comp_piece(ch, i)
            if cp is not None:
                cm = crop_ink(cp)
                if cm is None:
                    return None, None, None
                return cm, mh, mw
    r = assemble(ch, depth)
    return (None, None, None) if r is None else (r, 1.0, 1.0)

def ser_node(node):
    if isinstance(node, tuple):
        return chr(node[0]) + ''.join(ser_node(k) for k in node[1])
    return node

INV_TREE = {}
for _c, _seq in IDS_MAP.items():
    try:
        _t = parse_tree(_seq)
        if _t and isinstance(_t, tuple):
            INV_TREE.setdefault(ser_node(_t), _c)
    except Exception:
        pass

INV_TREE = {}
for _c, _seq in IDS_MAP.items():
    try:
        _t = parse_tree(_seq)
        if _t and isinstance(_t, tuple):
            INV_TREE.setdefault(ser_node(_t), _c)
    except Exception:
        pass

def compose_node(node, depth=0):
    if not isinstance(node, tuple):
        m, _, _ = part_mask_ex(node, depth)
        return m
    op, kids = node
    return compose_lr_tb(op, kids, depth)

def compose_lr_tb(op, kids, depth=0):
    H = 200.0
    if op == 0x2FF0:
        pieces = []
        for k in kids:
            pm, nh, _nw = part_mask_ex(k, depth, top=0x2FF0)
            if pm is None:
                return None
            th = max(12, int(round(H * (nh if nh and 0.25 < nh <= 1.05 else 1.0))))
            pm = scale_to(pm, target_h=th)
            pieces.append(pm)
        gap = 6
        Ws = sum(p.shape[1] for p in pieces) + gap * (len(pieces) - 1)
        out = np.zeros((int(H), Ws), bool)
        x = 0
        for p in pieces:
            h, w = p.shape
            y = (out.shape[0] - h) // 2
            out[y:y + h, x:x + w] |= p
            x += w + gap
        return out
    if op == 0x2FF1:
        pieces = []
        for k in kids:
            pm, _nh, nw = part_mask_ex(k, depth, top=0x2FF1)
            if pm is None:
                return None
            tw = max(12, int(round(H * (nw if nw and 0.25 < nw <= 1.05 else 1.0))))
            pm = scale_to(pm, target_w=tw)
            pieces.append(pm)
        gap = 8
        Ws = max(p.shape[1] for p in pieces)
        Hs = sum(p.shape[0] for p in pieces) + gap * (len(pieces) - 1)
        out = np.zeros((Hs, Ws), bool)
        y = 0
        for p in pieces:
            h, w = p.shape
            out[y:y + h, (Ws - w)//2:(Ws - w)//2 + w] |= p
            y += h + gap
        return out
    return None

def finish(bm, size=256, margin=16):
    h, w = bm.shape
    s = min((size - 2 * margin) / h, (size - 2 * margin) / w, 1.0)
    bm = scale_to(bm, target_h=max(1, int(round(h * s))))
    # width follow
    return bm

def to_canvas(bm, size=256):
    h, w = bm.shape
    if h > size - 2 or w > size - 2:
        s = min((size - 2) / h, (size - 2) / w)
        bm = scale_to(bm, target_h=max(1, int(round(h * s))))
        h, w = bm.shape
    img = np.full((size, size), 255, np.uint8)
    y = (size - h) // 2
    x = (size - w) // 2
    img[y:y + h, x:x + w] = np.where(bm, 0, 255)
    return img

# ---- coverage over missing ----
if __name__ == '__main__':
    miss = json.load(open(MISS, encoding='utf-8'))
    miss = set(miss['missing'] if isinstance(miss, dict) and 'missing' in miss else miss)
    ok = [c for c in sorted(miss) if comp_status(c)]
    print('missing:', len(miss), '-> assemblable:', len(ok))
    from collections import Counter
    print(Counter(comp_status(c) for c in ok))

    demo = [c for c in ok if comp_status(c) in ('lib', 'build')][:120]
    sheet, cols = [], 15
    cell = 64
    rows = (len(demo) + cols - 1) // cols
    sheet = np.full((rows * (cell + 8) + 30, cols * (cell + 4) + 10), 255, np.uint8)
    for i, ch in enumerate(demo):
        bm = assemble(ch)
        if bm is None:
            continue
        img = to_canvas(bm, cell)
        r, c = i // cols, i % cols
        y, x = r * (cell + 8), c * (cell + 4)
        sheet[y:y + cell, x:x + cell] = img[:cell, :cell]
    ok2, buf = cv2.imencode('.png', sheet)
    buf.tofile(os.path.join(TMP, 'assemble0.png'))
    json.dump(demo, open(os.path.join(TMP, 'assemble0_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    print('sheet saved, demo n=', len(demo))
