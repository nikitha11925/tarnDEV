"""Build a manifest from data/raw page images (student booklets only).
Usage: python src/make_manifest.py <out.csv> [max_pages_per_booklet (0 = all)]
"""
import csv, os, re, sys, glob
out, cap = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
rows = []
for d in sorted(glob.glob("data/raw/*/")):
    name = os.path.basename(os.path.dirname(d))
    if "answer booklet" not in name.lower() and "student" not in name.lower():
        continue
    pages = sorted(glob.glob(os.path.join(d, "p*.png")))
    for p in (pages[:cap] if cap else pages):
        rows.append(dict(page_id=re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + "_" + os.path.basename(p)[:-4],
                         image=p.replace("\\", "/"), truth_file=""))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["page_id", "image", "truth_file"]); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} pages -> {out}")
