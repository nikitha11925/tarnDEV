"""Web demo: upload a scanned page OR a whole booklet (PDF), pick models, see each model's text, speed and memory.

  .venv\\Scripts\\python.exe src\\app.py        then open http://127.0.0.1:7860

Accepts PDF, PNG, JPG. For a PDF choose which pages to read (e.g. "1-3,5"; blank = first 5 pages).
If you paste the correct (typed) text, CER / WER / digit errors are computed per model on the whole text read.
Models run one after another and are unloaded in between, so they fit on a 6 GB GPU.
"""
import os, re, sys, tempfile, time
# MUST be set before gradio/transformers are imported, or models get re-downloaded into the C: drive cache
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
import gradio as gr
import pymupdf
from readers import READERS
from metrics import text_scores, looks_degenerate

ORDER = ["glm-ocr", "qwen3vl-4b", "qwen3vl-2b", "trocr-base", "paddleocr-vl"]
NAMES = [n for n in ORDER if n in READERS] + [n for n in READERS if n not in ORDER]
SLOTS = len(NAMES)
DEFAULT_PAGES = 5
MAX_PAGES = 40


def parse_pages(spec, n):
    """'1-3,5' -> [0,1,2,4] (0-based). Blank -> first DEFAULT_PAGES. Capped at MAX_PAGES."""
    if not spec or not spec.strip():
        return list(range(min(n, DEFAULT_PAGES)))
    out = []
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part: continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m: raise ValueError(f"Can't read page range '{part}'. Use e.g. 1-3,5")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out += [p - 1 for p in range(a, b + 1) if 1 <= p <= n]
    out = sorted(set(out))
    if not out: raise ValueError(f"No valid pages in '{spec}' (file has {n} pages)")
    return out[:MAX_PAGES]


def to_images(files, spec, workdir):
    """Turn uploaded files into a list of (label, png_path)."""
    pages = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        base = os.path.splitext(os.path.basename(path))[0]
        if path.lower().endswith(".pdf"):
            doc = pymupdf.open(path)
            for i in parse_pages(spec, len(doc)):
                out = os.path.join(workdir, f"{base}_p{i + 1:02d}.png")
                doc[i].get_pixmap(dpi=150).save(out)
                pages.append((f"{base} p{i + 1}", out))
        else:
            pages.append((base, path))
    return pages[:MAX_PAGES]


def run(files, chosen, spec, truth):
    blanks = [""] * (2 * SLOTS)
    if not files:
        yield ("Upload a PDF or image first.", [], *blanks); return
    if not chosen:
        yield ("Pick at least one model.", [], *blanks); return
    work = tempfile.mkdtemp(prefix="ocr_demo_")
    try:
        pages = to_images(files, spec, work)
    except Exception as e:
        yield (f"Problem with the file or page range: {e}", [], *blanks); return
    gallery = [p for _, p in pages]
    outs = [""] * SLOTS; infos = [""] * SLOTS
    status = f"Reading {len(pages)} page(s)..."
    for i, name in enumerate(NAMES):
        if name not in chosen: continue
        infos[i] = f"**{name}**: loading model..."
        yield (status, gallery, *outs, *infos)
        r = READERS[name]()
        try:
            t0 = time.time(); r.load(); load_s = time.time() - t0
            texts, secs, peak = [], [], 0
            for k, (label, img) in enumerate(pages, 1):
                infos[i] = f"**{name}**: page {k}/{len(pages)}..."
                yield (status, gallery, *outs, *infos)
                text, sec, vram = r.read(img)
                texts.append(f"--- {label} ---\n{text}"); secs.append(sec); peak = max(peak, vram)
                outs[i] = "\n\n".join(texts)
                yield (status, gallery, *outs, *infos)
            full = "\n\n".join(texts)
            line = (f"**{name}**: {sum(secs):.0f}s total · {sum(secs) / len(secs):.1f}s/page · "
                    f"{peak / 1024:.1f} GB VRAM · {len(pages)} page(s) (load {load_s:.0f}s)")
            if any(looks_degenerate(t.split('---\n', 1)[-1]) for t in texts): line += " · ⚠ a page came out empty or repeated"
            if truth and truth.strip():
                body = "\n".join(t.split("---\n", 1)[-1] for t in texts)       # drop the page headers before scoring
                s = text_scores(truth, body)
                dig = "n/a" if s["digit_cer"] is None else f"{s['digit_cer']:.3f}"
                line += f"\n\nCER **{s['cer']:.3f}** · WER **{s['wer']:.3f}** · digit CER {dig} · invented-text rate {s['ins_rate']:.3f}"
            outs[i], infos[i] = full, line
        except Exception as e:
            infos[i] = f"**{name}**: FAILED ({type(e).__name__}: {str(e)[:150]})"
        finally:
            r.unload()
        yield (status, gallery, *outs, *infos)
    yield ("Done.", gallery, *outs, *infos)


with gr.Blocks(title="OCR model comparison") as demo:
    gr.Markdown("# OCR model comparison\nUpload a **PDF booklet** or page images. Optionally paste the correct text to get "
                "accuracy scores (CER = share of characters wrong; lower is better).")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Booklet PDF or page images", file_count="multiple",
                            file_types=[".pdf", ".png", ".jpg", ".jpeg"])
            spec = gr.Textbox(label="Pages to read (PDF only)", placeholder=f"e.g. 1-3,5   (blank = first {DEFAULT_PAGES})")
            chosen = gr.CheckboxGroup(NAMES, value=["glm-ocr"], label="Models to run")
            truth = gr.Textbox(lines=6, label="Correct text (optional, enables scoring)")
            go = gr.Button("Run models", variant="primary")
            status = gr.Markdown()
            gallery = gr.Gallery(label="Pages being read", columns=3, height=260)
        with gr.Column(scale=2):
            boxes, infos = [], []
            for n in NAMES:
                infos.append(gr.Markdown())
                boxes.append(gr.Textbox(label=n, lines=10, interactive=False))
    go.click(run, [files, chosen, spec, truth], [status, gallery, *boxes, *infos])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
