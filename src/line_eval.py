"""Score line-by-line reading at PAGE level, so it can be compared fairly with reading the whole page.
For each page: join the model's line outputs in order, compare with the joined line truth.
Usage: python src/line_eval.py <model> <manifest.csv> [<manifest2.csv> ...]
"""
import csv, os, re, statistics as st, sys, collections
sys.path.insert(0, os.path.dirname(__file__))
from metrics import text_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRIP = "--strip-math" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
model = args[0]


def strip_math(t):
    """Drop $...$ blocks (e.g. \textcircled{1}, \frac...) that some readers wrap around numbers and symbols."""
    return re.sub(r"\s+", " ", re.sub(r"\$[^$]*\$", " ", t)).strip()

for mf in args[1:]:
    rows = list(csv.DictReader(open(f"{ROOT}/{mf}", encoding="utf-8")))
    pages = collections.defaultdict(lambda: ([], []))
    for r in rows:
        pid = r["page_id"]; page = re.sub(r"_[A-Z]\d+$", "", pid)
        op = f"{ROOT}/results/outputs/{model}/{pid}.txt"
        if not os.path.exists(op): continue
        pages[page][0].append(open(f"{ROOT}/{r['truth_file']}", encoding="utf-8").read())
        txt = open(op, encoding="utf-8").read(); pages[page][1].append(strip_math(txt) if STRIP else txt)
    cer = []
    for page, (t, h) in pages.items():
        s = text_scores(" ".join(t), " ".join(h))
        if s["cer"] is not None: cer.append(min(s["cer"], 1.0))
    if cer:
        print(f"{mf:42s} {model:12s} pages={len(cer):2d}  avg accuracy {100*(1-st.mean(cer)):5.1f}%  typical page {100*(1-st.median(cer)):5.1f}%  pages>=95%: {sum(c<=0.05 for c in cer)}/{len(cer)}  worst {100*(1-max(cer)):.0f}%")
