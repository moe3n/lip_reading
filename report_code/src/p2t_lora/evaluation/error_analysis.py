"""P2T LoRA Decoder: Error Pattern Analysis (Stage 2 + Stage 3 Options 2/3/5)"""

import sys
import os
import re
from typing import List, Dict, Optional
from collections import Counter

import jiwer

_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))
_P2T_LORA_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR         = os.path.dirname(_P2T_LORA_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _P2T_LORA_DIR)
from p2t_lora.augmentation.hard_negatives import get_homophones, get_near_homophones
from p2t_lora.evaluation.contextual_analysis import check_grammar
from p2t_lora.evaluation.llm_judge import classify_error

def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def classify_substitution(ref_word: str, hyp_word: str) -> str:
    """Classify a single substitution (ref_word -> hyp_word) as one of:"""
    ref_word, hyp_word = ref_word.upper(), hyp_word.upper()
    if ref_word == hyp_word:
        return "Equal"

    if hyp_word in get_homophones(ref_word):
        return "Homophone"

    near = get_near_homophones(ref_word, max_distance=1)
    near_words = {w for w, dist in near if dist >= 1}
    if hyp_word in near_words:
        return "Near-homophone"

    return "Other"

def resolve_substitution(reference: str,
                          original_hyp_sentence: str,
                          hyp_word: str,
                          hyp_word_idx: int,
                          category: str,
                          tokenizer=None,
                          model=None,
                          use_llm: bool = False) -> Dict:
    """Stage 3 of the P2T framework: given a substitution already classified"""
    if category not in ("Homophone", "Near-homophone"):
        return {"stage3_category": None, "stage3_subcategory": None,
                "stage3_explanation": None, "stage3_method": None}

    grammar_result = check_grammar(original_hyp_sentence, hyp_word, hyp_word_idx)
    if grammar_result["resolved"]:
        return {
            "stage3_category":    "Contextual",
            "stage3_subcategory": grammar_result["rule"],
            "stage3_explanation": grammar_result["explanation"],
            "stage3_method":      "Option 3 (grammar)",
        }

    if use_llm and tokenizer is not None and model is not None:
        llm_result = classify_error(tokenizer, model, reference, original_hyp_sentence)
        return {
            "stage3_category":    llm_result["category"],
            "stage3_subcategory": llm_result["subcategory"],
            "stage3_explanation": llm_result["explanation"],
            "stage3_method":      "Option 5 (LLM judge)",
        }

    return {"stage3_category": None, "stage3_subcategory": None,
            "stage3_explanation": None, "stage3_method": None}

def analyze_pair(reference: str, hypothesis: str,
                  tokenizer=None, model=None, use_llm: bool = False) -> Dict:
    """Run jiwer's word-level alignment on one (reference, hypothesis) pair and return the aligned operations."""
    ref_norm = normalise(reference)
    hyp_norm = normalise(hypothesis)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return {"n_hits": 0, "n_substitutions": 0, "n_deletions": 0, "n_insertions": 0,
                "substitutions": []}

    hyp_words_raw = hypothesis.split()
    can_escalate = len(hyp_words_raw) == len(hyp_words)

    out = jiwer.process_words([ref_norm], [hyp_norm])
    chunk = out.alignments[0]

    subs = []
    for c in chunk:
        if c.type != "substitute":
            continue
        ref_span = ref_words[c.ref_start_idx:c.ref_end_idx]
        hyp_span = hyp_words[c.hyp_start_idx:c.hyp_end_idx]
        for offset, (rw, hw) in enumerate(zip(ref_span, hyp_span)):
            category = classify_substitution(rw, hw)
            if can_escalate:
                hyp_idx = c.hyp_start_idx + offset
                stage3 = resolve_substitution(
                    reference, hypothesis, hyp_words_raw[hyp_idx], hyp_idx, category,
                    tokenizer=tokenizer, model=model, use_llm=use_llm,
                )
            else:
                stage3 = {"stage3_category": None, "stage3_subcategory": None,
                          "stage3_explanation": None, "stage3_method": None}
            subs.append({"ref": rw, "hyp": hw, "category": category, **stage3})

    return {
        "n_hits":          out.hits,
        "n_substitutions": out.substitutions,
        "n_deletions":     out.deletions,
        "n_insertions":    out.insertions,
        "substitutions":   subs,
    }

def error_category_report(all_refs: List[str],
                           all_hyps: List[str],
                           homo_mask: Optional[List[bool]] = None,
                           tokenizer=None,
                           model=None,
                           use_llm: bool = False) -> Dict:
    """Run analyze_pair() across an entire evaluation set and aggregate."""
    def _accumulate(refs, hyps):
        totals = {"n_hits": 0, "n_substitutions": 0, "n_deletions": 0, "n_insertions": 0}
        cat_counts = Counter()
        stage3_counts = Counter()
        stage3_method_counts = Counter()
        examples = {"Homophone": [], "Near-homophone": [], "Other": []}
        stage3_examples = []
        for ref, hyp in zip(refs, hyps):
            res = analyze_pair(ref, hyp, tokenizer=tokenizer, model=model, use_llm=use_llm)
            for k in totals:
                totals[k] += res[k]
            for s in res["substitutions"]:
                cat_counts[s["category"]] += 1
                if s["category"] in examples and len(examples[s["category"]]) < 5:
                    examples[s["category"]].append((ref, hyp, s["ref"], s["hyp"]))
                if s.get("stage3_category"):
                    stage3_counts[s["stage3_category"]] += 1
                    stage3_method_counts[s["stage3_method"]] += 1
                    if len(stage3_examples) < 8:
                        stage3_examples.append({
                            "ref_sentence": ref, "hyp_sentence": hyp,
                            "ref_word": s["ref"], "hyp_word": s["hyp"],
                            "stage3_category":    s["stage3_category"],
                            "stage3_subcategory": s["stage3_subcategory"],
                            "stage3_explanation": s["stage3_explanation"],
                            "stage3_method":      s["stage3_method"],
                        })
        return totals, cat_counts, examples, stage3_counts, stage3_method_counts, stage3_examples

    (overall_totals, overall_cats, overall_examples,
     overall_stage3, overall_stage3_methods, overall_stage3_examples) = _accumulate(all_refs, all_hyps)

    report = {
        "overall": {
            "totals": overall_totals,
            "substitution_categories": dict(overall_cats),
            "examples": overall_examples,
            "stage3_categories": dict(overall_stage3),
            "stage3_methods": dict(overall_stage3_methods),
            "stage3_examples": overall_stage3_examples,
        }
    }

    if homo_mask is not None:
        homo_refs = [r for r, m in zip(all_refs, homo_mask) if m]
        homo_hyps = [h for h, m in zip(all_hyps, homo_mask) if m]
        non_refs  = [r for r, m in zip(all_refs, homo_mask) if not m]
        non_hyps  = [h for h, m in zip(all_hyps, homo_mask) if not m]

        (h_totals, h_cats, h_ex,
         h_stage3, h_stage3_methods, h_stage3_examples) = _accumulate(homo_refs, homo_hyps)
        (n_totals, n_cats, n_ex,
         n_stage3, n_stage3_methods, n_stage3_examples) = _accumulate(non_refs, non_hyps)
        report["homophone"]     = {"totals": h_totals, "substitution_categories": dict(h_cats),
                                    "examples": h_ex, "stage3_categories": dict(h_stage3),
                                    "stage3_methods": dict(h_stage3_methods),
                                    "stage3_examples": h_stage3_examples}
        report["non_homophone"] = {"totals": n_totals, "substitution_categories": dict(n_cats),
                                    "examples": n_ex, "stage3_categories": dict(n_stage3),
                                    "stage3_methods": dict(n_stage3_methods),
                                    "stage3_examples": n_stage3_examples}

    return report

def print_error_report(report: Dict, title: str = "Error Pattern Analysis") -> None:
    width = 66
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

    for key, label in [("overall", "Overall"), ("homophone", "Homophone subset"),
                        ("non_homophone", "Non-homophone subset")]:
        if key not in report:
            continue
        sect = report[key]
        t = sect["totals"]
        cats = sect["substitution_categories"]
        n_sub = t["n_substitutions"]
        print(f"\n  -- {label} --")
        print(f"     hits={t['n_hits']}  substitutions={t['n_substitutions']}  "
              f"deletions={t['n_deletions']}  insertions={t['n_insertions']}")
        if n_sub == 0:
            print("     (no substitutions to classify)")
            continue
        for cat in ["Homophone", "Near-homophone", "Other"]:
            n = cats.get(cat, 0)
            pct = (n / n_sub * 100) if n_sub else 0.0
            print(f"     {cat:<16} {n:>5} / {n_sub}  ({pct:5.1f}% of substitutions)")

        s3_cats = sect.get("stage3_categories", {})
        if s3_cats:
            s3_total = sum(s3_cats.values())
            phon_eligible = cats.get("Homophone", 0) + cats.get("Near-homophone", 0)
            print(f"\n     Stage 3 resolution: {s3_total} / {phon_eligible} phonetically-"
                  f"explainable substitutions resolved to a P2T error category:")
            for cat3 in ["Phonological", "Lexical", "Contextual", "Semantic", "Unparseable"]:
                n3 = s3_cats.get(cat3, 0)
                if n3:
                    print(f"       {cat3:<14} {n3:>4} / {s3_total}")
            s3_methods = sect.get("stage3_methods", {})
            if s3_methods:
                method_str = ", ".join(f"{m}: {n}" for m, n in sorted(s3_methods.items()))
                print(f"       (resolved via: {method_str})")

    if "homophone" in report and report["overall"]["totals"]["n_substitutions"] > 0:
        h_cats = report["homophone"]["substitution_categories"]
        h_sub = report["homophone"]["totals"]["n_substitutions"]
        if h_sub > 0:
            phon_explained = h_cats.get("Homophone", 0) + h_cats.get("Near-homophone", 0)
            pct = phon_explained / h_sub * 100
            print(f"\n  -> Of substitution errors on homophone-containing sentences,")
            print(f"     {pct:.1f}% are phonetically explainable (exact or near-homophone).")
            print( "     High %: re-enabling contrastive training (currently disabled: see")
            print( "             dryrun.py's module docstring) is likely worth the effort.")
            print( "     Low %:  errors are mostly unrelated substitutions: more data/")
            print( "             epochs/model capacity is the more relevant lever, not")
            print( "             contrastive/hard-negative mining.")
    print("\n" + "=" * width + "\n")

def plot_error_report(report: Dict, out_path: str, title: str = "Error Pattern Analysis") -> None:
    """Bar chart of substitution error categories (Homophone / Near-homophone"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("overall", "homophone", "non_homophone") if k in report]
    labels = {"overall": "Overall", "homophone": "Homophone", "non_homophone": "Non-Homophone"}
    categories = ["Homophone", "Near-homophone", "Other"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(keys))
    width = 0.25
    for i, cat in enumerate(categories):
        heights = []
        for key in keys:
            n_sub = report[key]["totals"]["n_substitutions"]
            n_cat = report[key]["substitution_categories"].get(cat, 0)
            heights.append(n_cat / n_sub * 100 if n_sub else 0.0)
        ax.bar([xi + i * width for xi in x], heights, width, label=cat)

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels([labels[k] for k in keys])
    ax.set_ylabel("% of substitution errors")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    import sys as _sys

    refs = [
        "I COULD LABEL THIS AS MEAT",
        "THEY FOUND THE SITE ON THE COAST",
        "WHAT REALLY MAKES A CHIP IS THE CRUNCH",
        "FRESH OUT THE FRYER",
        "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF",
        "THERE IS A CAT ON THE MAT",
    ]
    hyps = [
        "I COULD LABEL THIS AS MEET",
        "THEY FOUND THE SIGHT ON THE COAST",
        "WHAT REALLY MAKES A SHIP IS THE CRUNCH",
        "FRESH OUT THE DRYER",
        "THE TRADITIONAL BANANA PAN OFTEN STAYS ON THE SHELF",
        "THEIR IS A CAT ON THE MAT",
    ]
    homo_mask = [True, True, False, False, False, True]

    use_llm = "--with-llm" in _sys.argv
    tok, mdl = None, None
    if use_llm:
        from p2t_lora.model import load_tokenizer, MODEL_NAME_DRYRUN
        from transformers import AutoModelForCausalLM
        print(f"\n--with-llm: loading judge model ({MODEL_NAME_DRYRUN}) for Stage 3 Option 5...")
        tok = load_tokenizer(MODEL_NAME_DRYRUN)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME_DRYRUN)
        mdl.resize_token_embeddings(len(tok))
        mdl.eval()

    report = error_category_report(refs, hyps, homo_mask, tokenizer=tok, model=mdl, use_llm=use_llm)
    print_error_report(report, "Smoke test: error_analysis.py")

    direct = analyze_pair(refs[-1], hyps[-1])
    their_subs = [s for s in direct["substitutions"] if s["hyp"].lower() == "their"]
    assert their_subs, "expected a THEIR substitution in the THERE->THEIR smoke case"
    assert their_subs[0]["category"] == "Homophone", \
        f"expected Option 2 to classify THERE->THEIR as Homophone, got {their_subs[0]['category']}"
    assert their_subs[0]["stage3_category"] == "Contextual", \
        f"expected Stage 3 Option 3 to resolve THEIR-as-nsubj to Contextual, got {their_subs[0]['stage3_category']}"
    assert their_subs[0]["stage3_method"] == "Option 3 (grammar)", \
        f"expected resolution via Option 3, got {their_subs[0]['stage3_method']}"
    print("  [OK] THERE->THEIR substitution correctly escalated and resolved via Stage 3 Option 3.\n")
