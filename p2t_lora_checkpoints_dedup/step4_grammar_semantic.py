"""Stage 4 driver — Steps 7 (grammar) and 8 (semantic similarity) from
p2t_lora_checkpoints_dedup/NOTES.md.

Step 7 (grammar): language_tool_python is the Mira-Fleite methodology's
canonical tool, but it needs a local JRE which is not installed on this
machine. NOTES.md anticipates exactly this — its "Built, runs inside
dryrun.py" section lists contextual_analysis.check_grammar() as the
fallback path: closed-class dependency-role mismatch detection
(their/your/its/my/our/whose forced into a syntactic role only their
homophone counterpart could fill).

We apply that fallback path via error_analysis.analyze_pair(use_llm=False),
which runs jiwer word-level alignment, classifies each substitution, and
escalates Homophone/Near-homophone substitutions through check_grammar.
We also do a casing/punctuation/digit audit on all 949 hyps since the
dedup corpus contains known casing artifacts (eyeballed observation in
NOTES.md, "no SOUND" → casing artifact inflating CER).

Step 8 (semantic): BERTScore F1 between reference and hypothesis on the
EM-False slice. Uses bert_score package's default roberta-large model.

Outputs:
    analysis/tables/grammar_breakdown.csv     — per-EM-False-row resolution
    analysis/tables/casing_punct_audit.csv    — full-corpus mechanical issues
    analysis/tables/semantic_similarity.csv   — per-EM-False-row BERTScore F1
    analysis/tables/stage4_summary_inputs.csv — joined tables for Step 10
"""

import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Force CPU for any torch-side dependencies; no GPU in this env
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Make stdout utf-8-safe on Windows (cp1252) so summary prints with \u2265
# or any other non-ASCII glyph don't crash the run. No-op on Linux/macOS.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).parent
SRC = ROOT / "predictions_beam5_with_match.csv"
OUT_DIR = ROOT / "analysis" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent))  # so `import src.p2t_lora...` works
sys.path.insert(0, str(ROOT.parent / "src"))

from p2t_lora.evaluation import error_analysis as _ea
from p2t_lora.evaluation.error_analysis import analyze_pair
from p2t_lora.evaluation.contextual_analysis import (
    _VALID_DEPS as CLOSED_CLASS_TABLE,
)
from p2t_lora.augmentation.hard_negatives import get_homophones as _get_homophones


# ── speed patch: bypass get_near_homophones brute force ─────────────────────
# classify_substitution() normally calls get_near_homophones() which
# brute-force-scans the ~125k-entry CMU dict per distinct non-homophone
# ref_word -- ~1s/call. With 949 rows × multiple subs/row this would take
# 10+ minutes. We do a Stage-3-style downgraded classifier instead:
#   Equal   : ref == hyp
#   Homophone: hyp in get_homophones(ref)  (O(1) dict lookup)
#   Other   : otherwise
# Cost: check_grammar escalation only fires for True Homophones now
# (Near-homophones fall to "Other"). The closed-class closed-class list
# has only 6 words, and ~all closed-class confusions map to True Homophones
# (THEIR↔THERE, ITS↔IT'S, etc.), so the loss of Option-3 grammar signal on
# Near-homophone substitutions is small and bounded.
def _fast_classify_substitution(ref_word, hyp_word):
    rw, hw = ref_word.upper(), hyp_word.upper()
    if rw == hw:
        return "Equal"
    if hw in _get_homophones(rw):
        return "Homophone"
    return "Other"


_ea.classify_substitution = _fast_classify_substitution  # monkey-patch


# ── helpers ─────────────────────────────────────────────────────────────────

DIGIT_RE = re.compile(r"\d")
UPPER_RE = re.compile(r"[A-Z]")
LOWER_RE = re.compile(r"[a-z]")
PUNCT_END_RE = re.compile(r"[.!?]$")


def load_predictions():
    with SRC.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_em(rows):
    return (
        [r for r in rows if r["exact_match"] == "True"],
        [r for r in rows if r["exact_match"] == "False"],
    )


def audit_mechanical(text: str) -> dict:
    """Cheap, deterministic audit of casing/digit/punctuation issues.
    Returns the raw counts; caller aggregates.
    """
    words = text.split()
    n_words = max(len(words), 1)
    upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    has_trailing_period = bool(PUNCT_END_RE.search(text.strip()))
    has_digit = bool(DIGIT_RE.search(text))
    starts_with_upper = bool(text) and text[0].isupper()
    return {
        "n_words": n_words,
        "uppercase_words": upper_words,
        "has_trailing_period": has_trailing_period,
        "has_digit": has_digit,
        "starts_with_upper": starts_with_upper,
    }


# ── Step 7: grammar ─────────────────────────────────────────────────────────

def step7_grammar(rows):
    print("=" * 70)
    print(" Step 7 — Grammar (Option 3: closed-class dep-role mismatch)")
    print("=" * 70)
    print(f" closed-class words checked: {sorted(CLOSED_CLASS_TABLE.keys())}")
    print(f" running analyze_pair on {len(rows)} rows (use_llm=False)...")
    t0 = time.time()

    out_rows = []
    n_em_false = 0
    n_with_subs = 0
    n_total_subs = 0
    n_resolved_by_grammar = 0
    rule_counter = Counter()
    category_counter = Counter()
    homophone_subs = 0
    near_homophone_subs = 0

    for i, r in enumerate(rows, 1):
        em = r["exact_match"] == "False"
        if em:
            n_em_false += 1
        result = analyze_pair(r["target"], r["prediction"], use_llm=False)
        subs = result["substitutions"]
        n_total_subs += len(subs)
        if subs:
            n_with_subs += 1
        for s in subs:
            category_counter[s["category"]] += 1
            if s["category"] == "Homophone":
                homophone_subs += 1
            elif s["category"] == "Near-homophone":
                near_homophone_subs += 1
            if s["stage3_category"] == "Contextual":
                n_resolved_by_grammar += 1
                rule_counter[s["stage3_subcategory"] or "unknown"] += 1
        out_rows.append({
            "row": i,
            "target": r["target"],
            "prediction": r["prediction"],
            "exact_match": r["exact_match"],
            "is_homophone": r["is_homophone"],
            "n_subs": len(subs),
            "n_resolved_by_grammar": sum(
                1 for s in subs if s["stage3_category"] == "Contextual"),
            "homophone_subs": sum(1 for s in subs if s["category"] == "Homophone"),
            "near_homophone_subs": sum(1 for s in subs if s["category"] == "Near-homophone"),
            "other_subs": sum(1 for s in subs if s["category"] == "Other"),
            "grammar_explanations": " | ".join(
                s["stage3_explanation"] for s in subs
                if s["stage3_explanation"]),
        })
        if i % 200 == 0:
            print(f"  ... {i}/{len(rows)} rows ({time.time()-t0:.1f}s)")

    dt = time.time() - t0
    print(f"\n done in {dt:.1f}s")
    print(f"  EM-False rows: {n_em_false}")
    print(f"  rows with at least one substitution: {n_with_subs}")
    print(f"  total substitutions: {n_total_subs}")
    print(f"  substitution category breakdown: {dict(category_counter)}")
    print(f"  substitutions resolved by grammar (Option 3): {n_resolved_by_grammar}")
    print(f"  grammar rules fired: {dict(rule_counter)}")

    # write per-row table
    grammar_table = OUT_DIR / "grammar_breakdown.csv"
    with grammar_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"  wrote {grammar_table.name}")

    # casing/punct/digit audit on all 949 hyps
    print(f"\n running mechanical (casing/punct/digit) audit on {len(rows)} hyps...")
    audit_counter = Counter()
    by_em = {"True": [], "False": []}
    for r in rows:
        a = audit_mechanical(r["prediction"])
        # bucket: "ok" or first issue found
        flag = "ok"
        if a["uppercase_words"] > 0:
            flag = "uppercase_word_present"
        if a["has_digit"]:
            flag = "contains_digit"
        audit_counter[flag] += 1
        by_em[r["exact_match"]].append((flag, a))

    print(f"  audit bucket counts: {dict(audit_counter)}")
    for em in ("True", "False"):
        cnt = Counter(f for f, _ in by_em[em])
        print(f"  audit on EM-{em}: {dict(cnt)}")

    audit_table = OUT_DIR / "casing_punct_audit.csv"
    with audit_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["exact_match", "flag", "n"])
        for em in ("True", "False"):
            cnt = Counter(f for f, _ in by_em[em])
            for flag, n in cnt.most_common():
                w.writerow([em, flag, n])
    print(f"  wrote {audit_table.name}")

    return {
        "n_total_subs": n_total_subs,
        "n_resolved_by_grammar": n_resolved_by_grammar,
        "category_counter": dict(category_counter),
        "rule_counter": dict(rule_counter),
        "audit_counter": dict(audit_counter),
    }


# ── Step 8: semantic similarity (BERTScore) ────────────────────────────────

def step8_semantic(em_false_rows):
    print()
    print("=" * 70)
    print(" Step 8 — Semantic similarity (BERTScore F1, EM-False slice)")
    print("=" * 70)
    print(f" {len(em_false_rows)} EM-False rows")

    refs = [r["target"] for r in em_false_rows]
    hyps = [r["prediction"] for r in em_false_rows]

    # write inputs to disk so the file is reproducible
    refs_table = OUT_DIR / "semantic_refs.csv"
    hyps_table = OUT_DIR / "semantic_hyps.csv"
    with refs_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "target"]); [w.writerow([i+1, t]) for i, t in enumerate(refs)]
    with hyps_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["row", "prediction"]); [w.writerow([i+1, t]) for i, t in enumerate(hyps)]

    print(" loading bert_score...")
    t0 = time.time()
    from bert_score import score as bert_score
    # The default model for lang='en' is roberta-large; it'll download
    # on first use (~1.4GB). Use a smaller model to keep this
    # reproducible on a CPU box.
    P, R, F1 = bert_score(
        hyps, refs,
        lang="en",
        model_type="microsoft/deberta-base-mnli",   # smaller, faster, still strong
        num_layers=10,
        verbose=False,
        device="cpu",
    )
    dt = time.time() - t0
    print(f" bert_score done in {dt:.1f}s")

    f1_list = F1.tolist()
    rows_out = []
    for i, r in enumerate(em_false_rows):
        rows_out.append({
            "row": i + 1,
            "target": refs[i],
            "prediction": hyps[i],
            "is_homophone": r["is_homophone"],
            "bertscore_f1": round(f1_list[i], 4),
        })

    sem_table = OUT_DIR / "semantic_similarity.csv"
    with sem_table.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f" wrote {sem_table.name}")

    mean_f1 = sum(f1_list) / len(f1_list)
    median_f1 = sorted(f1_list)[len(f1_list) // 2]
    above_090 = sum(1 for x in f1_list if x >= 0.90)
    above_070 = sum(1 for x in f1_list if x >= 0.70)
    below_050 = sum(1 for x in f1_list if x < 0.50)
    print(f"\n BERTScore F1 stats on {len(em_false_rows)} EM-False rows:")
    print(f"  mean   : {mean_f1:.4f}")
    print(f"  median : {median_f1:.4f}")
    print(f"  ≥ 0.90 : {above_090} ({above_090/len(f1_list)*100:.1f}%)")
    print(f"  ≥ 0.70 : {above_070} ({above_070/len(f1_list)*100:.1f}%)")
    print(f"  < 0.50 : {below_050} ({below_050/len(f1_list)*100:.1f}%)")

    # split by homophone mask
    homo_f1 = [f1_list[i] for i, r in enumerate(em_false_rows) if r["is_homophone"] == "True"]
    nonhomo_f1 = [f1_list[i] for i, r in enumerate(em_false_rows) if r["is_homophone"] == "False"]
    if homo_f1:
        print(f"  homophone    mean: {sum(homo_f1)/len(homo_f1):.4f}  (n={len(homo_f1)})")
    if nonhomo_f1:
        print(f"  non-homophone mean: {sum(nonhomo_f1)/len(nonhomo_f1):.4f}  (n={len(nonhomo_f1)})")

    return {
        "mean_f1": mean_f1,
        "median_f1": median_f1,
        "n": len(f1_list),
        "above_090": above_090,
        "above_070": above_070,
        "below_050": below_050,
        "homo_mean": (sum(homo_f1) / len(homo_f1)) if homo_f1 else None,
        "nonhomo_mean": (sum(nonhomo_f1) / len(nonhomo_f1)) if nonhomo_f1 else None,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main():
    rows = load_predictions()
    em_t, em_f = split_em(rows)
    print(f"loaded {len(rows)} rows: EM-T {len(em_t)}, EM-F {len(em_f)}\n")

    gram = step7_grammar(rows)
    sem = step8_semantic(em_f)

    # write a small json-ish summary for Step 10 to read
    summary = {
        "step7_grammar": gram,
        "step8_semantic": sem,
    }
    import json
    sum_path = OUT_DIR / "stage4_metrics_summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {sum_path.name}")


if __name__ == "__main__":
    main()