"""Cut GNHK pages into straightened, masked line crops and make clean-up variants (oracle segmentation:
uses the dataset's line boxes, so it measures what segmentation + clean-up can give at best).
Variants:  tight  = straightened tight crop, paper shading flattened, neighbours masked out
           norm   = tight + contrast boost (CLAHE) + scaled to a standard height (zoom) + white margin
           clean  = norm + ruled-line removal + light denoise
           bin    = clean + adaptive black/white (Sauvola)
Usage: python src/prep_lines.py   -> data/gnhk/lines_<variant>/ and data/manifest_lines_<variant>.csv
"""
import collections, glob, json, os, re
import cv2, numpy as np
from skimage.filters import threshold_sauvola

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
VARS = ["tight", "norm", "clean", "bin"]
H = 96   # standard line height after zoom/normalise


def remove_rules(gray):
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, gray.shape[1] // 3), 1)))
    rules = cv2.dilate(rules, np.ones((3, 1), np.uint8))
    return cv2.inpaint(gray, rules, 3, cv2.INPAINT_TELEA)


def variants(tight):
    g = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
    s = H / g.shape[0]
    zoom = lambda a: cv2.copyMakeBorder(cv2.resize(a, (max(8, int(a.shape[1] * s)), H), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA),
                                        12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255)
    norm = zoom(cv2.createCLAHE(2.0, (4, 4)).apply(g))
    clean = cv2.fastNlMeansDenoising(zoom(remove_rules(cv2.createCLAHE(2.0, (4, 4)).apply(g))), None, 7, 7, 21)
    b = (clean > threshold_sauvola(clean, window_size=31, k=0.2)).astype(np.uint8) * 255
    return {"tight": tight, "norm": norm, "clean": clean, "bin": b}


rows = {v: ["page_id,image,truth_file"] for v in VARS}
for v in VARS: os.makedirs(f"data/gnhk/lines_{v}", exist_ok=True)
n_lines = 0
for jf in sorted(glob.glob("data/gnhk/images/*.json")):
    n = os.path.basename(jf)[:-5]
    if not os.path.exists(f"data/gnhk/truth/{n}.txt"): continue
    d = json.load(open(jf, encoding="utf-8")); im = cv2.imread(f"data/gnhk/images/{n}.jpg")
    lines = collections.defaultdict(list)
    for w in d: lines[w["line_idx"]].append(w)
    for k, ws in sorted(lines.items()):
        txt = re.sub(r"\s+", " ", re.sub(r"%[A-Za-z]+%", "", " ".join(w["text"] for w in sorted(ws, key=lambda w: w["polygon"]["x0"])))).strip()
        if len(txt) < 4: continue
        polys = [np.array([[w["polygon"][f"x{i}"], w["polygon"][f"y{i}"]] for i in range(4)], np.float32) for w in ws]
        pts = np.concatenate(polys)
        (cx, cy), (w_, h_), ang = cv2.minAreaRect(pts)
        if w_ < h_: w_, h_ = h_, w_; ang += 90
        if ang > 90: ang -= 180
        M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rot = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]), flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))
        mask = np.zeros(im.shape[:2], np.uint8)
        for p in polys:
            q = cv2.transform(p[None], M)[0].astype(np.int32); cv2.fillConvexPoly(mask, q, 255)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(h_ * 0.35) | 1, int(h_ * 0.35) | 1)))
        pad = int(h_ * 0.10) + 3
        x0, x1 = int(max(0, cx - w_ / 2 - pad)), int(min(im.shape[1], cx + w_ / 2 + pad))
        y0, y1 = int(max(0, cy - h_ / 2 - pad)), int(min(im.shape[0], cy + h_ / 2 + pad))
        crop = rot[y0:y1, x0:x1].copy(); m = mask[y0:y1, x0:x1]
        if crop.shape[1] < 30 or crop.shape[0] < 15: continue
        # flatten uneven paper shading/shadow (divide by a smooth estimate of the paper), then blank out neighbours
        flat = np.empty_like(crop)
        for c in range(3):
            paper = cv2.GaussianBlur(cv2.dilate(crop[:, :, c], np.ones((15, 15), np.uint8)), (0, 0), 21)
            flat[:, :, c] = cv2.divide(crop[:, :, c], paper, scale=255)
        crop = flat; crop[m == 0] = (255, 255, 255)
        n_lines += 1
        for v, img in variants(crop).items():
            pid = f"{v}_{n}_M{k:02d}"
            cv2.imwrite(f"data/gnhk/lines_{v}/{pid}.png", img)
            tf = f"data/gnhk/lines_{v}/{pid}.txt"; open(tf, "w", encoding="utf-8").write(txt)
            rows[v].append(f"{pid},data/gnhk/lines_{v}/{pid}.png,{tf}")
for v in VARS: open(f"data/manifest_lines_{v}.csv", "w").write("\n".join(rows[v]) + "\n")
print(n_lines, "lines x", len(VARS), "variants")
