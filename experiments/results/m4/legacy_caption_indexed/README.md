# Original Caption-Indexed BM25 — Diagnostic / Leakage-Affected Protocol

This directory archives the uncorrected Milestone 4 benchmark results where BM25 documents were formed by concatenating all 5 human captions ($D_i = C_{i1} \mathbin{\Vert} \dots \mathbin{\Vert} C_{i5}$), allowing the exact test query text to be present in the target image document during retrieval.

These numbers are preserved for historical audit and diagnostic contrast:
- Method A1 (BM25 Leaky): R@1 = 85.65%, R@5 = 96.38%, R@10 = 98.57%, MRR = 0.9028
- Method A2 (CLIP ViT-B/32 Zero-Shot): R@1 = 5.45%, R@5 = 17.71%, R@10 = 27.89%, MRR = 0.1307
- Method A3 (CLIP + Prompt Ensemble): R@1 = 5.14%, R@5 = 17.00%, R@10 = 28.01%, MRR = 0.1268

The corrected, leakage-controlled protocol (Leave-One-Caption-Out BM25 and Category Metadata BM25) is documented in `experiments/results/m4/` and `docs/m4_semantic_retrieval.md`.
