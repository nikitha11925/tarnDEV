"""Summarise per-page result CSVs that contain truth-based scores.
Usage: python src/summarize.py results/eval_pages_glm-ocr.csv results/eval_pages_qwen3vl-2b.csv ...
Prints per-model means, worst page, and a per-page CER table.
"""
import sys
import pandas as pd

df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True)
df = df[df.page_id != "*"]
for c in ("cer", "wer", "ins_rate", "digit_cer", "seconds", "vram_mb"):
    if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")

g = df.groupby("model")
summary = pd.DataFrame({
    "CER_mean": g.cer.mean(), "CER_median": g.cer.median(), "CER_worst": g.cer.max(),
    "WER_mean": g.wer.mean(), "insert_rate": g.ins_rate.mean(), "digit_CER": g.digit_cer.mean(),
    "sec_median": g.seconds.median(), "sec_p95": g.seconds.quantile(0.95), "vram_MB": g.vram_mb.max(),
}).round(3)
print(summary.sort_values("CER_mean").to_string())
print("\nPer-page CER:")
print(df.pivot(index="page_id", columns="model", values="cer").round(3).to_string())
print("\nPer-page digit CER:")
print(df.pivot(index="page_id", columns="model", values="digit_cer").round(3).to_string())
