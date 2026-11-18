"""One-command demo: run OCR models on pages, then open an HTML comparison report.

  python src/demo.py                         run all 4 models on the full demo set (typed images + booklet pages)
  python src/demo.py --image some.png        run on ONE image of your choice (shows live text in the terminal)
  python src/demo.py --models glm-ocr        run only some models (glm-ocr qwen3vl-4b qwen3vl-2b paddleocr-vl)
  python src/demo.py --report-only           no model runs: rebuild + open the report from saved results (instant)

Each model runs in its own fresh process so GPU memory is released between models.
To score a page you must provide its typed text: put it in the manifest's truth_file column.
"""
import argparse, csv, os, subprocess, sys, webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
ALL = ["glm-ocr", "qwen3vl-4b", "qwen3vl-2b", "paddleocr-vl"]

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="data/manifest_demo.csv")
ap.add_argument("--image")
ap.add_argument("--truth", help="optional text file with the correct text for --image")
ap.add_argument("--models", nargs="+", default=ALL, choices=ALL)
ap.add_argument("--report-only", action="store_true")
a = ap.parse_args()

manifest = a.manifest
if a.image:
    manifest = "results/manifest_single.csv"
    pid = os.path.splitext(os.path.basename(a.image))[0].replace(" ", "_")
    with open(f"{ROOT}/{manifest}", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["page_id", "image", "truth_file"])
        w.writerow([pid, os.path.abspath(a.image), a.truth or ""])

if not a.report_only:
    py = sys.executable
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for m in a.models:
        print(f"\n=== {m} ===", flush=True)
        r = subprocess.run([py, "src/run_eval.py", manifest, m], cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        print("\n".join(l for l in r.stdout.splitlines() if l.startswith("[")), flush=True)
        if a.image:
            p = f"{ROOT}/results/outputs/{m}/{pid}.txt"
            if os.path.exists(p): print("--- text read ---\n" + open(p, encoding="utf-8").read() + "\n-----------------")

import report
out = report.build(manifest)
print("\nReport:", out)
webbrowser.open("file:///" + out.replace("\\", "/"))
