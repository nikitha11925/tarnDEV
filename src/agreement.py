"""Pairwise CER between models' outputs per page (no ground truth needed).
Usage: python src/agreement.py   (reads results/outputs/<model>/<page>.txt)
Lower = the two models read the page more alike. A model that disagrees with all others is suspect.
"""
import itertools, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from metrics import pair_agreement, looks_degenerate

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = f"{root}/results/outputs"
models = sorted(d for d in os.listdir(out) if os.path.isdir(f"{out}/{d}"))
pages = sorted({f[:-4] for m in models for f in os.listdir(f"{out}/{m}") if f.endswith(".txt")})
read = lambda m, p: open(f"{out}/{m}/{p}.txt", encoding="utf-8").read() if os.path.exists(f"{out}/{m}/{p}.txt") else ""

print("Degenerate/empty outputs:")
for m in models:
    bad = [p for p in pages if looks_degenerate(read(m, p))]
    print(f"  {m}: {len(bad)}/{len(pages)} {bad}")

print("\nPairwise CER (mean over pages; each pair's order averaged both ways):")
mean = {}
for a, b in itertools.combinations(models, 2):
    v = [(pair_agreement(read(a, p), read(b, p)) + pair_agreement(read(b, p), read(a, p))) / 2 for p in pages]
    mean[(a, b)] = sum(v) / len(v)
    print(f"  {a:12s} vs {b:12s}: {mean[(a, b)]:.3f}  per page: {[round(x, 3) for x in v]}")

print("\nMean disagreement with the other models:")
for m in models:
    v = [x for (a, b), x in mean.items() if m in (a, b)]
    print(f"  {m}: {sum(v) / len(v):.3f}")
