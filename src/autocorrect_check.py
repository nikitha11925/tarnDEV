"""How often does a model 'fix' what the student wrote, and do its [?..?] flags point at real errors?

Autocorrection proxy: take words in the answer key that are NOT in an English dictionary (student misspellings,
abbreviations, names). If the model outputs a DIFFERENT word that IS in the dictionary at that spot, it autocorrected.
This over-counts a little (names, slang) but is applied identically to every model, so it is fair for comparison.
Flag quality: words the model marked [?word?]: what share is truly wrong (precision), and what share of all wrong words got flagged (recall).
Usage: python src/autocorrect_check.py <model_dir_name> [<model_dir_name> ...]   (reads results/outputs/<name>/gnhk_*.txt)
"""
import difflib, glob, os, re, sys
from spellchecker import SpellChecker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = SpellChecker()
tok = lambda t: re.findall(r"[a-z']+", re.sub(r"\[\?|\?\]", " ", t.lower()))
FLAG = re.compile(r"\[\?(.+?)\?\]")

for name in sys.argv[1:]:
    mis_total = auto = wrong = flagged_wrong = flagged = words = 0
    for tf in sorted(glob.glob(f"{ROOT}/data/gnhk/truth/*.txt")):
        pid = "gnhk_" + os.path.basename(tf)[:-4]
        op = f"{ROOT}/results/outputs/{name}/{pid}.txt"
        if not os.path.exists(op): continue
        ref = tok(open(tf, encoding="utf-8").read()); raw = open(op, encoding="utf-8").read()
        hyp = tok(raw)
        flag_words = {w for f in FLAG.findall(raw) for w in tok(f)}
        words += len(ref)
        for op_, a1, a2, b1, b2 in difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes():
            if op_ == "equal":
                for w in ref[a1:a2]:
                    if len(w) >= 3 and w in sp.unknown([w]): mis_total += 1                # misspelling copied faithfully
                    if w in flag_words: flagged += 1
            elif op_ == "replace" and (a2 - a1) == (b2 - b1):
                for r, h in zip(ref[a1:a2], hyp[b1:b2]):
                    wrong += 1
                    if len(r) >= 3 and r in sp.unknown([r]):
                        mis_total += 1
                        if h not in sp.unknown([h]): auto += 1                                # fixed to a real word
                    if h in flag_words: flagged += 1; flagged_wrong += 1
            else:
                wrong += max(a2 - a1, 0)
    rate = 100 * auto / mis_total if mis_total else float("nan")
    prec = 100 * flagged_wrong / flagged if flagged else float("nan")
    rec = 100 * flagged_wrong / wrong if wrong else float("nan")
    print(f"{name:20s} misspelled words in key: {mis_total:3d} | autocorrected: {auto:3d} ({rate:4.1f}%) | "
          f"flagged words: {flagged:3d} (precision {prec:4.1f}%, caught {rec:4.1f}% of wrong words)")
