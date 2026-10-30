#!/usr/bin/env python3
"""rag_eval.py - measure a retrieval pipeline honestly.

Compares lexical (BM25), dense (embeddings) and hybrid (reciprocal rank fusion)
on a question set where each question has a known gold passage, and reports
every number with a confidence interval, plus a paired bootstrap on the
difference between systems (the number that tells you if "A beats B" is real).

Usage:
  python fetch_data.py                      # one-time: builds data/corpus.jsonl, data/questions.jsonl
  python rag_eval.py --data data            # runs all systems
  python rag_eval.py --data data --no-dense # lexical only (no model download)

Bring your own data: corpus.jsonl  -> {"id": "...", "text": "..."}
                     questions.jsonl -> {"q": "...", "gold": ["id", ...]}
A question counts as a hit@k if ANY gold id is in the top k.
"""
import argparse, json, math, random, re, sys, time
from collections import Counter, defaultdict

TOK = re.compile(r"[a-z0-9]+")
STOP = set("a an the of to in on for and or is are was were be by with as at from that this it its which who what when where how why did does do has have had".split())


def tokenize(s):
    return [t for t in TOK.findall(s.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / len(docs)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.inv = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.inv[t].append(i)

    def rank(self, q, k=100):
        sc = defaultdict(float)
        for t in set(tokenize(q)):
            if t not in self.idf:
                continue
            for i in self.inv[t]:
                f = self.tf[i][t]
                sc[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]


def rrf(rankings, k=60, top=100):
    sc = defaultdict(float)
    for r in rankings:
        for pos, i in enumerate(r):
            sc[i] += 1.0 / (k + pos + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:top]]


def wilson(h, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = h / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def per_question(ranked_ids, gold, k):
    return 1 if any(g in set(ranked_ids[:k]) for g in gold) else 0


def rr(ranked_ids, gold, k=10):
    for pos, i in enumerate(ranked_ids[:k]):
        if i in gold:
            return 1.0 / (pos + 1)
    return 0.0


def paired_bootstrap(a, b, iters=5000, seed=7):
    """95% CI of mean(a-b) over questions. If it contains 0, you have not shown a difference."""
    rnd = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rnd.randrange(n) for _ in range(n)]
        diffs.append(sum(a[i] - b[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    corpus = [json.loads(l) for l in open(f"{a.data}/corpus.jsonl", encoding="utf-8")]
    qs = [json.loads(l) for l in open(f"{a.data}/questions.jsonl", encoding="utf-8")]
    ids = [d["id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    print(f"corpus={len(corpus)} passages, questions={len(qs)}", file=sys.stderr)

    systems = {}
    bm = BM25(texts)
    systems["bm25"] = [bm.rank(q["q"]) for q in qs]

    if not a.no_dense:
        import numpy as np
        from fastembed import TextEmbedding
        m = TextEmbedding(a.model)
        t0 = time.time()
        D = np.array(list(m.embed(texts, batch_size=64)))
        Q = np.array(list(m.query_embed([q["q"] for q in qs])))
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        S = Q @ D.T
        systems["dense"] = [list(np.argsort(-S[i])[:100]) for i in range(len(qs))]
        systems["hybrid_rrf"] = [rrf([systems["bm25"][i], systems["dense"][i]]) for i in range(len(qs))]
        print(f"embedding+search {time.time()-t0:.0f}s ({a.model})", file=sys.stderr)

    res, pq = {}, {}
    for name, rk in systems.items():
        row = {}
        for k in (1, 3, 5, 10):
            hits = [per_question([ids[i] for i in r], q["gold"], k) for r, q in zip(rk, qs)]
            lo, hi = wilson(sum(hits), len(hits))
            row[f"hit@{k}"] = {"value": sum(hits) / len(hits), "ci95": [round(lo, 3), round(hi, 3)], "hits": sum(hits), "n": len(hits)}
            pq[(name, k)] = hits
        mrr = [rr([ids[i] for i in r], set(q["gold"])) for r, q in zip(rk, qs)]
        row["mrr@10"] = sum(mrr) / len(mrr)
        res[name] = row

    comp = {}
    names = list(systems)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in (1, 3):
                lo, hi = paired_bootstrap(pq[(names[i], k)], pq[(names[j], k)])
                comp[f"{names[i]} - {names[j]} @hit{k}"] = {"diff_ci95": [round(lo, 3), round(hi, 3)], "distinguishable_from_zero": not (lo <= 0 <= hi)}

    out = {"n_questions": len(qs), "n_passages": len(corpus), "systems": res, "paired_differences": comp}
    json.dump(out, open(a.out, "w"), indent=1)

    print(f"\n{'system':<12}" + "".join(f"{'hit@'+str(k):>22}" for k in (1, 3, 5, 10)) + "   mrr@10")
    for name, row in res.items():
        cells = "".join(f"  {row[f'hit@{k}']['value']*100:5.1f}% [{row[f'hit@{k}']['ci95'][0]*100:4.1f}-{row[f'hit@{k}']['ci95'][1]*100:4.1f}]" for k in (1, 3, 5, 10))
        print(f"{name:<12}{cells}   {row['mrr@10']:.3f}")
    print("\npaired differences (95% bootstrap CI; if it contains 0, the gap is not shown):")
    for k, v in comp.items():
        print(f"  {k:<32} {v['diff_ci95']}  {'REAL' if v['distinguishable_from_zero'] else 'not distinguishable'}")


if __name__ == "__main__":
    main()
