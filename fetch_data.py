#!/usr/bin/env python3
"""Build a small public test set from SQuAD 2.0 dev (CC BY-SA 4.0).

Corpus   = every distinct paragraph of the first N articles.
Question = answerable questions; gold = the paragraph holding the answer.

Caveat (be honest about it): SQuAD questions were written while looking at the
paragraph, so they share many words with it. That favours lexical search and
makes every system look better than on real employee questions. Use this to
learn the method; use YOUR questions to judge YOUR system.
"""
import json, os, sys, urllib.request

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
N_ARTICLES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
os.makedirs("data", exist_ok=True)
raw = json.load(urllib.request.urlopen(URL, timeout=60))["data"][:N_ARTICLES]
corpus, qs = [], []
for ai, art in enumerate(raw):
    for pi, p in enumerate(art["paragraphs"]):
        pid = f"{ai}-{pi}"
        corpus.append({"id": pid, "text": f"{art['title'].replace('_', ' ')}. {p['context']}"})
        for q in p["qas"]:
            if not q.get("is_impossible"):
                qs.append({"q": q["question"], "gold": [pid]})
with open("data/corpus.jsonl", "w", encoding="utf-8") as f:
    for d in corpus: f.write(json.dumps(d) + "\n")
with open("data/questions.jsonl", "w", encoding="utf-8") as f:
    for q in qs: f.write(json.dumps(q) + "\n")
print(f"{len(corpus)} passages, {len(qs)} questions from {len(raw)} articles")
