"""Run readers on a manifest and write per-page results.

manifest CSV columns: page_id,image,truth_file   (truth_file may be empty: then only latency/agreement are measured)
Usage: python src/run_eval.py data/manifest.csv qwen3vl-2b glm-ocr ...
Outputs: results/outputs/<model>/<page_id>.txt and results/eval_pages_{"_".join(names)}.csv
"""
import csv, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(__file__))
import torch
from metrics import text_scores, looks_degenerate
from readers import READERS

manifest, names = sys.argv[1], sys.argv[2:]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pages = list(csv.DictReader(open(manifest, encoding="utf-8")))
rows = []

for name in names:
    reader = READERS[name]()
    try:
        reader.load()
    except Exception as e:
        print(f"[{name}] LOAD FAILED: {type(e).__name__}: {str(e)[:300]}")
        rows.append(dict(model=name, page_id="*", failed="load:" + type(e).__name__))
        reader.unload(); continue
    os.makedirs(f"{root}/results/outputs/{name}", exist_ok=True)
    first = True
    for p in pages:
        if first:                              # warm-up run, not recorded
            try: reader.read(p["image"])
            except Exception: pass
            first = False
        try:
            text, sec, vram = reader.read(p["image"])
            err = ""
        except Exception as e:
            text, sec, vram, err = "", 0.0, 0.0, f"{type(e).__name__}: {str(e)[:120]}"
        open(f"{root}/results/outputs/{name}/{p['page_id']}.txt", "w", encoding="utf-8").write(text)
        row = dict(model=name, page_id=p["page_id"], seconds=round(sec, 2), vram_mb=round(vram),
                   chars=len(text), degenerate=looks_degenerate(text), failed=err)
        tf = p.get("truth_file")
        if tf and os.path.exists(tf):
            row.update({k: (round(v, 4) if v is not None else None)
                        for k, v in text_scores(open(tf, encoding="utf-8").read(), text).items()})
        rows.append(row)
        print(f"[{name}] {p['page_id']}: {sec:.1f}s, {vram:.0f} MB, {len(text)} chars {err}")
    reader.unload()

keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
stem = os.path.splitext(os.path.basename(manifest))[0]
with open(f"{root}/results/eval_{'_'.join(names)}__{stem}.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"wrote results/eval_{'_'.join(names)}__{stem}.csv")
