# rag-eval-kit

Measure a retrieval pipeline honestly: lexical vs dense vs hybrid, every number with a confidence interval, and a paired test for "is B really better than A?".

~150 lines of Python, no framework. Lexical search is a pure-Python BM25; dense uses `fastembed` (local ONNX model, no API key, no cost).

## Run it (5 minutes)

```bash
pip install fastembed numpy
python fetch_data.py 20          # builds data/ from SQuAD 2.0 dev (CC BY-SA 4.0)
python rag_eval.py               # all systems  (add --no-dense for lexical only)
```

Output: hit@1/3/5/10 with 95% Wilson intervals, MRR@10, and paired bootstrap intervals for the difference between systems. Full numbers in `results.json`.

## Use it on your own documents

Two files in `data/`:

- `corpus.jsonl`: `{"id": "doc-12", "text": "..."}` one passage per line
- `questions.jsonl`: `{"q": "How many days of notice...?", "gold": ["doc-12"]}`

Write the questions the way your colleagues ask them, not by copying phrases from the passage. 60-100 real questions is enough to see big gaps, and not enough to see small ones. The tool will tell you which.

## Read the output like this

- **hit@k**: share of questions whose right passage is in the top k. Not "accuracy": a right passage in the top 3 does not mean the final answer is right.
- **[low-high]**: where the true value plausibly lies. With 65 questions the interval is about ±10 points wide. Two systems with overlapping intervals are not shown to differ.
- **paired difference**: the correct way to compare two systems on the same questions. If its interval contains 0, you have not shown a difference.

`is-my-rag-better.html` does the same two calculations for numbers you already have (works offline).

## Known limits

- The bundled SQuAD set is easy for lexical search: its questions were written looking at the passage. Treat the bundled run as a demonstration of the method, not a verdict on any product.
- One embedding model (`BAAI/bge-small-en-v1.5`), one fusion rule (reciprocal rank fusion, k=60), no reranker, no chunking study. Change one thing at a time.
- Gold labels are one passage per question. Multi-passage questions need `gold` with several ids; a hit means any of them.

MIT licence for the code. Data: SQuAD 2.0, CC BY-SA 4.0 (Rajpurkar et al.), downloaded by the script, not redistributed here.
