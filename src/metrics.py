"""Text metrics. All models are scored with the same normalisation (see docs/metrics.md)."""
import re
import jiwer


def normalise(s: str) -> str:
    """Case-insensitive, whitespace collapsed, markdown noise removed. Punctuation is KEPT."""
    s = re.sub(r"[*_#`>|]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def text_scores(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalise(reference), normalise(hypothesis)
    if not ref:
        return dict(cer=None, wer=None, ins_rate=None, digit_cer=None)
    c = jiwer.process_characters(ref, hyp)
    w = jiwer.process_words(ref, hyp)
    n = max(len(ref), 1)
    # digit-only CER: how well numbers (USN, marks) survive
    rd, hd = "".join(re.findall(r"\d", ref)), "".join(re.findall(r"\d", hyp))
    digit_cer = jiwer.cer(rd, hd) if rd else None
    return dict(cer=c.cer, wer=w.wer, ins_rate=c.insertions / n, digit_cer=digit_cer)


def pair_agreement(a: str, b: str) -> float:
    """CER between two readers: a free error signal when there is no ground truth."""
    a, b = normalise(a), normalise(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return jiwer.cer(a, b)


def looks_degenerate(text: str) -> bool:
    """Runaway repetition: same 12-char chunk repeated many times, or empty output."""
    t = normalise(text)
    if len(t) < 5:
        return True
    chunk = t[-12:]
    return t.count(chunk) > 8
