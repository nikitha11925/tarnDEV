"""Build one self-contained HTML report (results/report.html) from saved model outputs.

For every page in the manifest it shows the image, the typed answer key (if one exists),
and each model's text with errors highlighted. Pages with no answer key show model agreement instead.
Usage: python src/report.py [manifest]      (demo.py calls this for you)
"""
import base64, csv, difflib, html, io, itertools, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__))
from PIL import Image
from metrics import normalise, text_scores, pair_agreement, looks_degenerate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["glm-ocr", "qwen3vl-4b", "qwen3vl-2b", "paddleocr-vl", "trocr-base"]
E = html.escape


def thumb(path, width=760):
    im = Image.open(path).convert("RGB")
    im = im.resize((width, int(im.height * width / im.width)))
    b = io.BytesIO(); im.save(b, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def read_out(model, pid):
    p = f"{ROOT}/results/outputs/{model}/{pid}.txt"
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def timings(model):
    import glob
    out = {}
    for p in [f"{ROOT}/results/eval_pages_{model}.csv"] + sorted(glob.glob(f"{ROOT}/results/eval_{model}__*.csv")):
        if os.path.exists(p):
            out.update({r["page_id"]: r for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("page_id") != "*"})
    return out


def diff_html(ref, hyp):
    """Hypothesis words; red = wrong/extra, grey strike-through = word the model missed."""
    r, h = normalise(ref).split(), normalise(hyp).split()
    out = []
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
        if op == "equal": out += [E(w) for w in h[b1:b2]]
        else:
            out += [f'<span class="bad">{E(w)}</span>' for w in h[b1:b2]]
            out += [f'<span class="miss">{E(w)}</span>' for w in r[a1:a2] if op in ("delete", "replace")]
    return " ".join(out)


def build(manifest):
    pages = list(csv.DictReader(open(f"{ROOT}/{manifest}" if not os.path.isabs(manifest) else manifest, encoding="utf-8")))
    models = [m for m in MODELS if any(read_out(m, p["page_id"]) is not None for p in pages)]
    tm = {m: timings(m) for m in models}
    scores = {m: {} for m in models}
    for p in pages:
        tf = p.get("truth_file")
        if tf and os.path.exists(f"{ROOT}/{tf}"):
            ref = open(f"{ROOT}/{tf}", encoding="utf-8").read()
            for m in models:
                o = read_out(m, p["page_id"])
                if o is not None: scores[m][p["page_id"]] = text_scores(ref, o)

    # model-vs-model disagreement over ALL pages (needs no answer key)
    dis = {m: [] for m in models}
    for p in pages:
        outs = {m: read_out(m, p["page_id"]) for m in models}
        for a, b in itertools.combinations([m for m in models if outs[m] is not None], 2):
            d = (pair_agreement(outs[a], outs[b]) + pair_agreement(outs[b], outs[a])) / 2
            dis[a].append(min(d, 1.0)); dis[b].append(min(d, 1.0))

    pct = lambda v: "–" if v is None else f"{v*100:.1f}%"
    f = lambda v, nd=3: "–" if v is None else f"{v:.{nd}f}"
    mean = lambda xs: st.mean(xs) if xs else None
    rows = ""
    for m in models:
        s = list(scores[m].values())
        secs = [float(r["seconds"]) for r in tm[m].values() if r.get("seconds")]
        vram = [float(r["vram_mb"]) for r in tm[m].values() if r.get("vram_mb")]
        rows += (f"<tr><td><b>{m}</b></td><td>{pct(mean([min(x['cer'], 1.0) for x in s if x['cer'] is not None]))}</td>"
                 f"<td>{pct(mean([min(x['wer'], 1.0) for x in s if x['wer'] is not None]))}</td>"
                 f"<td>{pct(mean([min(x['digit_cer'], 1.0) for x in s if x['digit_cer'] is not None]))}</td>"
                 f"<td>{pct(mean([min(x['ins_rate'], 1.0) for x in s if x['ins_rate'] is not None]))}</td>"
                 f"<td>{len(s)}</td><td>{pct(mean(dis[m]))}</td>"
                 f"<td>{f(st.median(secs),1) if secs else '–'}</td><td>{f(max(vram)/1024,1) if vram else '–'}</td></tr>")


    # ---- accuracy section: accuracy = 1 - CER (never below 0), per sample set, against the 95% target ----
    SETS = [("Typed-style images (easy)", lambda q: q.startswith("Sample") or q == "samle4"),
            ("GNHK real handwriting (public dataset)", lambda q: q.startswith("gnhk_")),
            ("Answer booklets (scanned exam pages)", lambda q: q.startswith("booklet"))]
    acc = ""
    for title, test in SETS:
        trs = ""
        for m in models:
            v = [max(0.0, 1 - x["cer"]) for q, x in scores[m].items() if test(q) and x["cer"] is not None]
            if not v: continue
            mean_a, med_a = st.mean(v), st.median(v)
            hit = sum(1 for a in v if a >= 0.95)
            col = "#2e9e5b" if mean_a >= 0.95 else "#d9a400" if mean_a >= 0.85 else "#d64545"
            trs += (f"<tr><td><b>{m}</b></td><td>{len(v)}</td>"
                    f"<td><div class=bar><i style='width:{mean_a*100:.0f}%;background:{col}'></i></div>{mean_a*100:.1f}%</td>"
                    f"<td>{med_a*100:.1f}%</td><td>{hit}/{len(v)}</td><td>{min(v)*100:.0f}%</td></tr>")
        if trs:
            acc += (f"<h3>{title}</h3><table><tr><th>Model</th><th>Pages scored</th><th>Average accuracy</th>"
                    f"<th>Typical page (median)</th><th>Pages at 95%+</th><th>Worst page</th></tr>{trs}</table>")
    if acc:
        acc = ("<h2>Accuracy</h2><p class=note>Accuracy = 100% minus the character error rate (CER). The accuracy target used here is "
               "<b>95% or better</b>. A model that writes far more text than the page contains (runaway repetition) scores 0% on that page, "
               "so one failure pulls the average down: that is intended, because a grader cannot trust such output. "
               "Only pages with a typed answer key are scored. Booklet pages are scored once typed reference text is available for them.</p>" + acc)
    cards = ""
    for p in pages:
        pid, tf = p["page_id"], p.get("truth_file")
        has = pid in scores[models[0]] if models else False
        ref = open(f"{ROOT}/{tf}", encoding="utf-8").read() if has else None
        body = ""
        if ref is not None:
            body += f'<div class="m"><div class="h">Typed answer key</div><pre>{E(ref)}</pre></div>'
        for m in models:
            o = read_out(m, pid)
            if o is None: continue
            t = tm[m].get(pid, {})
            flag = " ⚠ empty/runaway" if looks_degenerate(o) else ""
            if ref is not None:
                sc = scores[m][pid]
                head = f"errors: {pct(sc['cer'])} of characters, {pct(sc['wer'])} of words · {t.get('seconds','?')}s{flag}"
                txt = diff_html(ref, o)
            else:
                head = f"{t.get('seconds','?')}s · {len(o)} chars{flag}"
                txt = E(o).replace("\n", "<br>")
            body += f'<div class="m"><div class="h">{m}<span>{head}</span></div><div class="t">{txt}</div></div>'
        tag = "scored against typed key" if has else "no answer key: compare by eye / agreement"
        cards += (f'<section><h3>{E(pid)} <small>{tag}</small></h3><div class="row">'
                  f'<img src="{thumb(f"{ROOT}/{p["image"]}")}"><div class="cols">{body}</div></div></section>')

    page = f"""<!doctype html><meta charset=utf-8><title>OCR model comparison</title><style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#1d2330}}
h1{{margin:0 0 4px}}table{{border-collapse:collapse;background:#fff}}td,th{{border:1px solid #d7dbe2;padding:6px 12px;text-align:left}}
th{{background:#eef1f6}}section{{background:#fff;border:1px solid #d7dbe2;border-radius:8px;padding:14px;margin:18px 0}}
.row{{display:flex;gap:16px;align-items:flex-start}}.row img{{width:34%;border:1px solid #ccc}}.cols{{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.m{{border:1px solid #e1e4ea;border-radius:6px;padding:8px;overflow:auto;max-height:420px}}.h{{font-weight:600;margin-bottom:6px}}.h span{{font-weight:400;color:#667;margin-left:8px;font-size:12px}}
.t,pre{{font:13px/1.5 ui-monospace,Consolas,monospace;white-space:pre-wrap;margin:0}}.bad{{background:#ffd5d5;border-radius:3px;padding:0 2px}}
.bar{{display:inline-block;width:120px;height:10px;background:#e6e9ef;border-radius:5px;margin-right:8px;vertical-align:middle}}.bar i{{display:block;height:100%;border-radius:5px}}h2{{margin-top:28px}}h3{{margin:18px 0 6px}}
.miss{{color:#888;text-decoration:line-through}}small{{color:#667;font-weight:400}}.note{{max-width:980px;color:#445}}
@media(max-width:900px){{.row{{flex-direction:column}}.row img{{width:100%}}.cols{{grid-template-columns:1fr}}}}</style>
<h1>OCR model comparison</h1>
<p class=note>All error figures are percentages; lower is better. Each page's error is capped at 100% so one runaway page cannot show a value above 100% (2% character errors = about 2 in every 100 characters wrong). Text is compared after lower-casing and whitespace
clean-up. <b>CER/WER/digit/insert are only computed on pages that have a typed answer key</b> (column "Scored pages").
"Disagreement" is the average CER between this model and the others on <i>all</i> pages: a model that disagrees with everyone is suspect,
but agreement is not proof of correctness. Red = wrong or extra word; grey strike-through = word the model missed.</p>
{acc}<h2>Detailed metrics</h2><table><tr><th>Model</th><th>Character errors</th><th>Word errors</th><th>Digit errors</th><th>Invented text</th><th>Scored pages</th><th>Disagreement</th><th>Median s/page</th><th>Peak VRAM GB</th></tr>{rows}</table>
{cards}"""
    out = f"{ROOT}/results/report.html"
    open(out, "w", encoding="utf-8").write(page)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "data/manifest_demo.csv"))
