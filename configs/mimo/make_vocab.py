#!/usr/bin/env python3
"""Rank tokens (Qwen3.8 / any GGUF vocab) by weighted corpus frequency and write draft-vocab subsets (int32 ids).
usage: make_vocab.py CORPUS_DIR GGUF OUTPREFIX SIZE...
CORPUS_DIR holds gen.ids / md.ids / code.ids / py.ids: Python lists of token ids (llama-tokenize --ids) of model-written
agent text, docs, code and Python. Writes OUTPREFIX-SIZE.bin (little-endian int32 ids) for --spec-draft-vocab."""
import sys, ast, collections, struct, os
sys.path.insert(0, os.environ.get("GGUF_PY", "gguf-py"))  # path to llama.cpp/gguf-py
import gguf
d, gg, outp = sys.argv[1], sys.argv[2], sys.argv[3]; sizes = [int(x) for x in sys.argv[4:]]
r = gguf.GGUFReader(gg)
f = r.fields["tokenizer.ggml.token_type"]; types = [int(f.parts[i][0]) for i in f.data]
n_vocab = len(types)
special = [i for i, t in enumerate(types) if t not in (1,)]  # control / user-defined / unknown / byte
W = {"gen": 0.4, "md": 0.25, "code": 0.2, "py": 0.15}
train, test = {}, {}
for s in W:
    ids = ast.literal_eval(open(f"{d}/{s}.ids").read())
    cut = int(len(ids) * 0.8)
    train[s] = collections.Counter(ids[:cut]); test[s] = collections.Counter(ids[cut:])
    print(f"{s}: {len(ids)} tokens, {len(train[s])} distinct (train)")
score = collections.Counter()
for s, w in W.items():
    tot = sum(train[s].values())
    for t, c in train[s].items(): score[t] += w * c / tot
rank = [t for t, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
print(f"n_vocab {n_vocab}, special/non-normal {len(special)}, ranked {len(rank)}")
for n in sizes:
    sub = set(special); i = 0
    while len(sub) < n and i < len(rank): sub.add(rank[i]); i += 1
    t = 0
    while len(sub) < n:  # fill by id order (BPE merge order) when the corpus runs out
        if t not in sub: sub.add(t)
        t += 1
    cov = {s: sum(c for k, c in test[s].items() if k in sub) / sum(test[s].values()) for s in W}
    print(f"size {n}: held-out coverage " + " ".join(f"{s} {100*v:.2f}%" for s, v in cov.items()))
    ids = sorted(sub)
    open(f"{outp}-{n}.bin", "wb").write(struct.pack(f"<{len(ids)}i", *ids))
