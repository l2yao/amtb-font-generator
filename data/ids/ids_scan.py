# -*- coding: utf-8 -*-
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

def parse_ids(path):
    m = {}
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        cp = parts[0].strip()
        if not cp.startswith(('U+', 'U-')):
            continue
        try:
            if cp.startswith('U-'):
                ch = chr(int(cp[2:], 16))
            else:
                ch = chr(int(cp[2:], 16))
        except Exception:
            continue
        seq = parts[2].split()[0] if parts[2].split() else ''
        seq = seq.split('@apparent=')[0].strip()
        if not seq or '&' in seq:
            continue
        m[ch] = seq
    return m

ids = parse_ids(os.path.join(os.path.dirname(__file__), 'IDS-UCS-Basic.txt'))
print('IDS entries:', len(ids))

def seq_parts(seq):
    toks = []
    i = 0
    while i < len(seq):
        ch = seq[i]
        o = ord(ch)
        if 0x20000 <= o <= 0x3FFFF:  # ext-B.. surrogates handled naturally by python
            toks.append(ch); i += 1; continue
        if 0x2FF0 <= o <= 0x2FFB:
            toks.append(ch); i += 1; continue
        toks.append(ch); i += 1
    return toks

import json as J
glyphs = set()
gd = r'C:\Users\Long\Documents\chinese-font-generator\data\glyphs'
for f in os.listdir(gd):
    if f.endswith('.png'):
        glyphs.add(os.path.splitext(f)[0])

missing = J.load(open(r'C:\Users\Long\Documents\chinese-font-generator\data\processed\missing_no_cands.json', encoding='utf-8'))
missing = set(missing['missing'] if isinstance(missing, dict) and 'missing' in missing else missing)
print('missing:', len(missing))

memo_usable = {}
def usable(ch, depth=0):
    if ch in memo_usable:
        return memo_usable[ch]
    if depth > 4:
        return ch in glyphs
    if ch in glyphs:
        memo_usable[ch] = True
        return True
    seq = ids.get(ch)
    if not seq:
        memo_usable[ch] = False
        return False
    if not seq.startswith('\u2ff0'):
        memo_usable[ch] = False
        return False
    toks = seq_parts(seq)
    ok = all(t == '\u2ff0' or 0x2FF0 <= ord(t[0]) <= 0x2FFB or usable(t, depth+1) for t in toks)
    memo_usable[ch] = ok
    return ok

def topop(seq):
    # returns operator code point if top-level
    c0 = seq[0]
    if 0x2FF0 <= ord(c0) <= 0x2FFB:
        # count parts at this level (nesting)
        depth = 0; n = 0
        for ch in seq:
            o = ord(ch)
            if 0x2FF0 <= o <= 0x2FFB:
                if depth == 0 and 0x2FF0 <= ord(ch) <= 0x2FFB:
                    depth = 1; continue
                if depth == 1:
                    depth = 0
                else:
                    pass
        # simpler: trust parse of structure separately
    return None

import collections
simple_ops = {0x2FF0, 0x2FF1, 0x2FF2, 0x2FF3}  # l r t b l3 t3: LR, TB, 3LR, 3TB
hier_ops = {0x2FF4,0x2FF5,0x2FF6,0x2FF7,0x2FF8,0x2FF9,0x2FFA,0x2FFB}  # enclosures

def parse_tree(seq):
    toks = []
    i = 0
    while i < len(seq):
        o = ord(seq[i])
        if 0x20000 <= o <= 0x3FFFF and o not in (0x2FF0,0):
            toks.append(seq[i]); i += 1; continue
        if 0x2FF0 <= o <= 0x2FFB:
            toks.append(('op', seq[i])); i += 1; continue
        toks.append(('c', seq[i])); i += 1
    # build tree: operators consume 2 or 3 following items
    def build(pos):
        tok = toks[pos]
        if tok[0] == 'op':
            op = tok[1]
            n = 3 if ord(op) in (0x2FF2, 0x2FF3) else 2
            pos += 1
            kids = []
            for _ in range(n):
                node, pos = build(pos)
                kids.append(node)
            return (op, kids), pos
        return tok[1], pos + 1
    try:
        tree, end = build(0)
        if end != len(toks):
            return None
        return tree
    except Exception:
        return None

avail_targets = []
opstat = collections.Counter()
op_avail = collections.Counter()
unavail = collections.Counter()
for ch in sorted(missing):
    t = usable(ch)
    if not t:
        unavail['nousable'] += 1
        continue
    seq = ids.get(ch)
    if seq is None:
        continue
    try:
        if parse_tree(seq) is not None:
            avail_targets.append(ch)
            opstat['OK-parse'] += 1
    except Exception:
        pass

def flat_ops(tree, out):
    if isinstance(tree, tuple):
        op, kids = tree
        out.append(op)
        for k in kids:
            flat_ops(k, out)
    return out

def leaves_avail(tree):
    if isinstance(tree, tuple):
        return all(leaves_avail(k) for k in tree[1])
    return usable(tree) or tree in glyphs

# categorize
counts = collections.Counter()
examples = {}
def walk(tree):
    if isinstance(tree, tuple):
        walk(tree[1][0]) if False else [walk(k) for k in tree[1]]
        counts[tree[0]] += 1

cat = collections.Counter()
demo = collections.defaultdict(list)
def check(ch):
    seq = ids.get(ch)
    tree = parse_tree(seq) if seq else None
    if not seq:
        cat['no_ids'] += 1
        return False
    if tree is None:
        cat['parse_fail'] += 1
        return False
    ops = set(flat_ops(tree, []))
    hier = ops & hier_ops
    only_lr = ops == {chr(0x2FF0)}
    only_tb = ops == {chr(0x2FF1)}
    ok = leaves_avail(tree)
    if not ok:
        cat['comp_missing'] += 1
        return False
    if hier:
        cat['hier'] += 1
        if len(demo['hier']) < 30: demo['hier'].append(ch)
        return True
    if only_lr or only_tb:
        cat['simple'] += 1
        if len(demo['simple']) < 30: demo['simple'].append(ch)
        return True
    cat['mixed'] += 1
    if len(demo['mixed']) < 30: demo['mixed'].append(ch)
    return True

wins = [ch for ch in sorted(missing) if check(ch)]
for ch in wins:
    seq = ids.get(ch)
    demo.setdefault('list', []).append((ch, seq))

print('categories:', dict(cat))print('decomposable targets:', len(wins), '/', len(missing))
for op, tree in demo.get('list', []):
    pass
J.dump([{'ch': ch, 'seq': demo['list'] and dict(demo['list']).get(ch)} ], open('x', 'w')) if False else None
print('sample simple:', ''.join(demo['simple'][:30]))
print('sample hier:', ''.join(demo['hier'][:15]))
print('sample mixed:', ''.join(demo['mixed'][:15]))
J.dump([{'ch': ch, 'ids': seq} for ch, seq in demo['list']], open(r'C:\Users\Long\Documents\chinese-font-generator\data\processed\ids_decomposable.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
