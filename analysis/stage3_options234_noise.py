"""Stage 3 Options 2 and 3 on the noise model, each done as the framework
describes. Option 4 (semantic similarity) is already computed in
stage3_semantic_summary.json and is only re-read here for the combined print.

Option 2  Dictionary-Based Lexical Analysis
          For each word substitution, look both words up in the CMU pronouncing
          dictionary. Identical pronunciation -> label Homophone automatically.

Option 3  Grammar-Based Context Analysis (spaCy)
          Detect the framework's three contextual error types between target and
          prediction: tense, agreement (number), and word order. Also flags an
          inserted or dropped function word (a/an/the/is/are/was), which is the
          article/auxiliary case of a contextual error.

Writes per-option tables to p2t_lora_checkpoints_noise/analysis/.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import jiwer
import spacy

import sys
sys.path.insert(0, "src")
from p2t_lora.evaluation.metrics import normalise
from p2t_lora.data import g2p

ROOT = Path("p2t_lora_checkpoints_noise/analysis")
FAIL = ROOT / "failing_rows.csv"
NLP = spacy.load("en_core_web_sm")
FUNCTION = {"a", "an", "the", "is", "are", "was", "were", "am", "be", "been"}


def cmu_pron(word):
    """Stress-stripped ARPAbet for a word, or None if out of dictionary."""
    ph = g2p.word_to_phonemes(word, stress=False)
    return tuple(ph) if ph else None


def option2_dictionary(rows):
    """Exact-pronunciation homophone detection over every word substitution."""
    homophones = []
    total_subs = 0
    for r in rows:
        ref, hyp = normalise(r["target"]).split(), normalise(r["prediction"]).split()
        out = jiwer.process_words([" ".join(ref)], [" ".join(hyp)])
        for c in out.alignments[0]:
            if c.type != "substitute":
                continue
            for rw, hw in zip(ref[c.ref_start_idx:c.ref_end_idx], hyp[c.hyp_start_idx:c.hyp_end_idx]):
                total_subs += 1
                pr, ph = cmu_pron(rw), cmu_pron(hw)
                if pr and ph and pr == ph and rw != hw:
                    homophones.append((rw, hw))
    return homophones, total_subs


def option3_grammar(rows):
    """Tense / agreement / word-order detection with spaCy."""
    flagged = []           # (target, prediction, subcategory)
    for r in rows:
        ref_n, hyp_n = normalise(r["target"]), normalise(r["prediction"])
        rw, hw = ref_n.split(), hyp_n.split()

        # Word order: same words, different sequence.
        if rw != hw and sorted(rw) == sorted(hw):
            flagged.append((r["target"], r["prediction"], "word-order"))
            continue

        dr, dh = NLP(ref_n), NLP(hyp_n)
        lem_r = {t.lemma_: t for t in dr}
        out = jiwer.process_words([ref_n], [hyp_n])
        sub = None
        for c in out.alignments[0]:
            if c.type == "substitute":
                for a, b in zip(rw[c.ref_start_idx:c.ref_end_idx], hw[c.hyp_start_idx:c.hyp_end_idx]):
                    ta = next((t for t in dr if t.text == a), None)
                    tb = next((t for t in dh if t.text == b), None)
                    if ta is not None and tb is not None and ta.lemma_ == tb.lemma_ and a != b:
                        # Same lemma, different surface form -> grammatical variant.
                        if ta.morph.get("Tense") != tb.morph.get("Tense"):
                            sub = "tense"
                        elif ta.morph.get("Number") != tb.morph.get("Number"):
                            sub = "agreement"
                        else:
                            sub = "form"
            elif c.type in ("insert", "delete"):
                span = (hw[c.hyp_start_idx:c.hyp_end_idx] if c.type == "insert"
                        else rw[c.ref_start_idx:c.ref_end_idx])
                if any(w in FUNCTION for w in span):
                    sub = sub or "article/auxiliary"
        if sub:
            flagged.append((r["target"], r["prediction"], sub))
    return flagged


def main():
    rows = list(csv.DictReader(open(FAIL, encoding="utf-8")))

    homs, total_subs = option2_dictionary(rows)
    with open(ROOT / "stage3_option2_homophones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target_word", "predicted_word"]); w.writerows(homs)
    print("OPTION 2 — Dictionary-based homophone analysis")
    print(f"  {len(homs)} of {total_subs} word substitutions are exact-pronunciation homophones "
          f"({len(homs)/total_subs*100:.1f}%)")
    print("  pairs:", ", ".join(f"{a}->{b}" for a, b in homs))

    flagged = option3_grammar(rows)
    subcat = Counter(s for _, _, s in flagged)
    with open(ROOT / "stage3_option3_grammar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["target", "prediction", "subcategory"]); w.writerows(flagged)
    print("\nOPTION 3 — Grammar-based context analysis (spaCy)")
    print(f"  {len(flagged)} of {len(rows)} failing sentences show a contextual (grammatical) error")
    for cat, n in subcat.most_common():
        print(f"    {cat:<18} {n}")

    sem = json.load(open(ROOT / "stage3_semantic_summary.json"))
    print("\nOPTION 4 — Semantic similarity (BERTScore, already computed)")
    print(f"  mean F1 {sem['mean']}  median {sem['median']}  "
          f">=0.90 {sem['ge_0.90']}/{sem['n']}  <0.50 {sem['lt_0.50']}/{sem['n']}")


if __name__ == "__main__":
    main()
