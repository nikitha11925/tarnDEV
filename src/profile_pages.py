"""Phase 1 step: extract every page image from the student booklet PDFs and measure basic quality.

No ground truth needed. Output: data/raw/<booklet>/pNN.png (local only) and results/page_profile.csv
Usage: python -I src/profile_pages.py <folder with PDFs> <project root>
"""
import csv, glob, math, os, sys
import cv2, numpy as np, pymupdf

src_dir, root = sys.argv[1], sys.argv[2]
out_raw = os.path.join(root, "data", "raw")
rows = []

def page_image(doc, page):
    """Largest embedded image on the page (these PDFs are scans: one image per page)."""
    imgs = page.get_images()
    if not imgs:
        pix = page.get_pixmap(dpi=200)
        return cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
    best = max(imgs, key=lambda i: i[2] * i[3])
    raw = doc.extract_image(best[0])["image"]
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

def metrics(img):
    h, w = img.shape[:2]
    s = 1200 / w
    g = cv2.cvtColor(cv2.resize(img, (1200, int(h * s))), cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())               # at fixed width, comparable across pages
    p5, p95 = np.percentile(g, [5, 95])
    contrast = float((p95 - p5) / 255)                              # ink vs paper spread
    bg = cv2.GaussianBlur(cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)), (0, 0), 25)
    small = cv2.resize(bg, (12, 16)).astype(float)
    illum = float(small.std() / max(small.mean(), 1))               # shadow / uneven light
    edges = cv2.Canny(g, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 120, minLineLength=300, maxLineGap=10)
    angs = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            a = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(a) < 20:
                angs.append(a)
    skew = float(np.median(angs)) if len(angs) >= 5 else float("nan")  # ruled lines drive this
    return blur, contrast, illum, skew, len(angs)

for pdf in sorted(glob.glob(os.path.join(src_dir, "*Student Answer Booklet*.pdf"))):
    name = os.path.splitext(os.path.basename(pdf))[0]
    d = os.path.join(out_raw, name)
    os.makedirs(d, exist_ok=True)
    doc = pymupdf.open(pdf)
    for i, page in enumerate(doc):
        img = page_image(doc, page)
        cv2.imwrite(os.path.join(d, f"p{i+1:02d}.png"), img)
        blur, con, ill, skew, nl = metrics(img)
        rows.append(dict(booklet=name, page=i + 1, width=img.shape[1], height=img.shape[0],
                         blur_lapvar=round(blur, 1), contrast=round(con, 3),
                         illum_nonuniformity=round(ill, 3), skew_deg=round(skew, 2), ruled_lines=nl))

os.makedirs(os.path.join(root, "results"), exist_ok=True)
with open(os.path.join(root, "results", "page_profile.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
print(len(rows), "pages profiled")
