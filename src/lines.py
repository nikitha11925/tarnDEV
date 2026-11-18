"""Improvement pipeline: clean page -> split into text lines -> normalise each line -> read line by line.

Why: a VLM given a whole page skips, merges and 'fixes' text. One short, clean line at a time gives it
less to guess from and makes the output verifiable.
Run `python src/lines.py <page.png> <out_dir>` to inspect the cleaning + segmentation without any model.
"""
import os, sys, time
import cv2, numpy as np
from PIL import Image

LINE_H = 96          # every line crop is scaled to this height (px) before reading
PAD = 14


def ink_mask(img_bgr):
    """Pen strokes only. Blue/black pen is separated from ruled lines, margins and stains."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(cv2.resize(gray, None, fx=0.1, fy=0.1), 31)           # paper brightness / shadows
    bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    flat = np.clip(gray.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    dark = flat < 150
    b, _, r = [c.astype(int) for c in cv2.split(img_bgr)]
    blue = dark & ((b - r) > 25)
    if blue.sum() > 0.5 * max(dark.sum(), 1):         # blue pen on dark ruled lines: colour does the work
        mask = blue
    else:                                             # black pen: remove long horizontal runs (ruled lines)
        d = dark.astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(gray.shape[1] // 12, 40), 1))
        mask = (d > 0) & (cv2.morphologyEx(d, cv2.MORPH_OPEN, k) == 0)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask, flat


def clean_page(img_bgr):
    """White paper, dark smooth ink: shadows, ruled lines and specks removed."""
    mask, flat = ink_mask(img_bgr)
    keep = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    out = np.where(keep, flat, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), 0.8)                                  # soften jagged edges
    lo = np.percentile(out[keep], 2) if keep.any() else 0
    return np.clip((out.astype(np.float32) - lo) * 255 / max(255 - lo, 1), 0, 255).astype(np.uint8), mask


def estimate_skew(mask):
    """Angle (deg) that makes text rows sharpest: maximise variance of the row profile."""
    small = cv2.resize(mask, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
    h, w = small.shape
    best, best_a = -1, 0.0
    for a in np.arange(-4, 4.01, 0.25):
        r = cv2.warpAffine(small, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h))
        v = r.sum(axis=1).var()
        if v > best: best, best_a = v, a
    return best_a


def line_pitch(prof):
    """Typical distance between text lines (px), from autocorrelation of the row profile."""
    p = prof - prof.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    lo, hi = 50, min(400, len(p) // 3)
    return int(lo + np.argmax(ac[lo:hi])) if hi > lo else 120


def find_lines(mask, min_h=30):
    """Row bands that contain ink; tall bands (several tight lines) are cut at the emptiest rows."""
    h, w = mask.shape
    raw = cv2.dilate(mask, np.ones((1, 25), np.uint8)).sum(axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(raw.reshape(-1, 1), (1, 0), 5).ravel()
    pitch = line_pitch(prof)
    thr = max(prof.max() * 0.03, 6)
    on, bands, start = prof > thr, [], None
    for y, v in enumerate(on):
        if v and start is None: start = y
        if not v and start is not None: bands.append([start, y]); start = None
    if start is not None: bands.append([start, h])
    out = []
    for a, b in bands:
        n = max(int(round((b - a) / pitch)), 1)
        cuts = [a]
        for k in range(1, n):                         # cut near the expected boundary, at the lowest ink row
            c = a + (b - a) * k / n
            win = slice(int(c - pitch * 0.3), int(c + pitch * 0.3))
            cuts.append(int(c - pitch * 0.3) + int(np.argmin(prof[win])))
        cuts.append(b)
        out += [(cuts[i], cuts[i + 1]) for i in range(n)]
    return [(a, b) for a, b in out if b - a >= min_h]


def normalise_line(clean, mask, y0, y1):
    """Crop one line, trim to ink, scale to a fixed height, add white margin."""
    rows = slice(max(y0 - 8, 0), min(y1 + 8, clean.shape[0]))
    cols = np.where(mask[rows].sum(axis=0) > 0)[0]
    if cols.size == 0: return None
    x0, x1 = max(cols[0] - PAD, 0), min(cols[-1] + PAD, clean.shape[1])
    sub = (mask[rows] > 0).astype(np.uint8)           # drop strokes that belong to the neighbouring lines
    n, lab, _, cen = cv2.connectedComponentsWithStats(sub, connectivity=8)
    foreign = np.zeros(n, bool)
    for i in range(1, n): foreign[i] = not (y0 <= cen[i][1] + rows.start <= y1)
    kill = cv2.dilate(foreign[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    crop = np.where(kill, 255, clean[rows])[:, x0:x1].astype(np.uint8)
    s = LINE_H / crop.shape[0]
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(crop).convert("RGB")


def split_page(path):
    img = cv2.imread(path)
    clean, mask = clean_page(img)
    ang = estimate_skew(mask)                         # straighten tilted photos before cutting lines
    if abs(ang) > 0.2:
        h, w = mask.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
        mask = cv2.warpAffine(mask, M, (w, h)); clean = cv2.warpAffine(clean, M, (w, h), borderValue=255)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    spans = find_lines(mask)
    crops = [c for c in (normalise_line(clean, mask, a, b) for a, b in spans) if c is not None]
    return img, clean, spans, crops


VERBATIM = ("This image is ONE line of handwriting. Copy it exactly as written, character by character. "
            "Keep every spelling mistake, wrong word, missing letter and abbreviation exactly as the writer wrote it. "
            "Do NOT correct spelling or grammar and do NOT replace a word with the word you think was meant. "
            "If a word is hard to read, write your best literal reading inside [? ?], like [?word?]. "
            "Output only the line, nothing else.")


class LineReader:
    """Wraps any Reader: segment page -> read each line -> join. Same interface as Reader."""
    def __init__(self, inner, name):
        self.inner, self.name = inner, name

    def load(self):
        self.inner.load()
        if hasattr(self.inner, "prompt"): self.inner.prompt = VERBATIM

    def unload(self): self.inner.unload()

    @staticmethod
    def _guard(t):
        """Runaway repetition ('0000...', same word looping) means the model gave up: flag, don't trust."""
        import re
        if re.search(r"(.{1,6}){7,}", t) or len(t) > 160:
            return "[?unreadable line?]"
        return t

    def read(self, image_path):
        import torch
        _, _, spans, crops = split_page(image_path)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
        lines = [self._guard(self.inner._run(c).strip()) for c in crops]
        torch.cuda.synchronize()
        return "\n".join(lines), time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 2**20


if __name__ == "__main__":                            # inspection mode: no model needed
    page, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    img, clean, spans, crops = split_page(page)
    ov = img.copy()
    for a, b in spans: cv2.rectangle(ov, (0, a), (img.shape[1] - 1, b), (0, 0, 255), 4)
    cv2.imwrite(f"{out}/overlay.jpg", cv2.resize(ov, None, fx=0.4, fy=0.4))
    cv2.imwrite(f"{out}/clean.jpg", cv2.resize(clean, None, fx=0.4, fy=0.4))
    for i, c in enumerate(crops): c.save(f"{out}/line{i + 1:02d}.png")
    print(f"{len(crops)} lines -> {out}")
