"""Summarise finished runs into results/summary.md (one table) and results/compare.md (text side by side).
Usage: python src/demo_summary.py <model> [<model> ...]
"""
import os, sys, pandas as pd
models = sys.argv[1:]

def _csv(m):   # run_eval names files eval_<model>__<manifest>.csv; older runs used eval_pages_<model>.csv
    for p in (f"results/eval_{m}__manifest_demo.csv", f"results/eval_pages_{m}.csv"):
        if os.path.exists(p): return p

df = pd.concat([pd.read_csv(_csv(m)) for m in models if _csv(m)], ignore_index=True)
ok = df[df.page_id != "*"]
g = ok.groupby("model")
t = pd.DataFrame({
    "pages": g.size(),
    "avg_sec_per_page": g.seconds.mean().round(1),
    "peak_vram_mb": g.vram_mb.max().round(),
    "failed_pages": g.failed.apply(lambda s: s.notna().sum()),
    "degenerate_outputs": g.degenerate.sum(),
    "avg_cer(if truth)": g.cer.mean().round(3) if "cer" in ok else None,
    "avg_wer(if truth)": g.wer.mean().round(3) if "wer" in ok else None,
})
open("results/summary.md", "w", encoding="utf-8").write("# Model comparison\n\n" + t.to_markdown() + "\n")
print(t.to_string())
with open("results/compare.md", "w", encoding="utf-8") as f:
    f.write("# Side-by-side OCR output\n")
    for p in sorted(ok.page_id.unique()):
        f.write(f"\n## {p}\n")
        for m in models:
            fp = f"results/outputs/{m}/{p}.txt"
            if os.path.exists(fp):
                f.write(f"\n**{m}**\n\n```\n{open(fp, encoding='utf-8').read().strip()}\n```\n")
print("wrote results/summary.md and results/compare.md")
