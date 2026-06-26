"""
CPT Decoder — Error Pattern Analysis (Stage 2 + Stage 3 Options 2/3/5)
====================================================================
Implements the automatable slice of the "P2T Three-Stage Error Analysis
Framework" (P2T Error Pattern Analysis V1.pdf): Stage 2 (phoneme/word
error-pattern analysis), Stage 3 Option 2 (dictionary-based lexical/
homophone auto-labelling, classify_substitution() below), and -- added
25 Jun 2026, per the project's "don't be constrained by dependencies"
decision -- Stage 3 Options 3 and 5 (grammar-based contextual analysis
and LLM-based classification, resolve_substitution() below, which
escalates through evaluation/contextual_analysis.py then
evaluation/llm_judge.py).

WHY THIS SLICE, AND NOT THE WHOLE FRAMEWORK
--------------------------------------------
The framework's Stage 1 (PER/WER/CER) is already live in metrics.py.
Stage 3 Options 2/3/5 are now wired into this module (see above).
Option 1 (manual annotation) needs human judgement and isn't
automatable by definition. Option 4 (semantic similarity, e.g.
BERTScore/BLEURT) still needs a new dependency (sentence-transformers
or similar) not yet added -- deferred for the same reason Option 3/5
used to be (until the real Llama 3.2:3B run produces errors worth that
investment), though that reasoning is now weaker than it was: this
project's own "don't be constrained by dependencies" call already
overrode it once for Options 3/5, so Option 4 is a candidate for the
same treatment if/when it becomes useful, not a hard blocker. The CPU
dry run's errors (small stand-in models, tens-to-hundreds of sentences)
are still dominated by undertrained-model noise, per the earlier
three-agent WER audit -- not the kind of systematic confusion this
framework is built to detect at that scale, which is why Stage 3
Option 5's LLM-judge model quality is validated separately (see
llm_judge.py's own docstring) rather than assumed reliable just because
the wiring runs cleanly.

This module is the part that's genuinely free right now: it reuses
jiwer (already a project dependency, already imported in metrics.py,
but only ever called for its scalar wer()/cer() — process_words() also
gives a full substitution/insertion/deletion breakdown with per-pair
word alignment) and hard_negatives.py's get_homophones() /
get_near_homophones() (already live, already used for training-time
hard-negative mining — repurposed here for evaluation-time
classification).

WHAT IT'S FOR
-------------
The contrastive hard-negative mechanism in this architecture is
specifically designed to fix homophone-driven confusions. This module
answers the question that determines whether that's the right lever to
keep tuning: of the substitution errors a model makes, how many are
phonetically explainable (exact or near-homophone) vs. unrelated? If
errors cluster on homophones, that's a green light to keep tuning
contrastive hyperparameters (margin, negative count, mining density).
If errors are mostly unrelated substitutions, the bottleneck is more
likely model capacity / training data scale (i.e. needs the real GPU
run with more data and a far larger model), not more hard-negative
mining.
"""

import sys
import os
import re
from typing import List, Dict, Optional
from collections import Counter

import jiwer

# This file lives in src/cpt_decoder/evaluation/. Two different absolute
# import styles are in play across this codebase and BOTH need their own
# sys.path entry, explicitly, rather than relying on whichever directory
# happens to get auto-added based on how a script was invoked (that's
# what silently broke this module's own standalone run before: dryrun.py
# only ever worked because it's launched as `cd src/cpt_decoder &&
# python3 dryrun.py`, which auto-adds src/cpt_decoder for free -- a
# script one level deeper, like this one, auto-adds evaluation/ instead,
# which is no use to either import below):
#   - "from cpt_decoder.X import ..." (this file's own style, and
#     dryrun.py's) needs src/ on sys.path.
#   - "from data.g2p import ..." (used inside hard_negatives.py,
#     get_near_homophones(), as a lazy import) needs src/cpt_decoder/ on
#     sys.path -- it's a bare "data.X", not "cpt_decoder.data.X".
_THIS_DIR        = os.path.dirname(os.path.abspath(__file__))   # .../src/cpt_decoder/evaluation
_CPT_DECODER_DIR = os.path.dirname(_THIS_DIR)                    # .../src/cpt_decoder
_SRC_DIR         = os.path.dirname(_CPT_DECODER_DIR)             # .../src
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _CPT_DECODER_DIR)
from cpt_decoder.augmentation.hard_negatives import get_homophones, get_near_homophones  # noqa: E402
# Stage 3 Options 3 and 5 -- sibling modules in this same evaluation/
# package. Imported via the same absolute "cpt_decoder.X" style as the
# hard_negatives import above (rather than a relative "from .X import"),
# for consistency with this file's existing import style, and because
# _SRC_DIR is already guaranteed on sys.path by the block above regardless
# of how this file itself was invoked (package import, `-m`, or via
# dryrun.py). Neither import forces a heavy dependency to load eagerly:
# contextual_analysis.py only imports spacy (lightweight, already a
# project dependency as of 25 Jun 2026) and lazy-loads the actual
# en_core_web_sm model on first use; llm_judge.py only imports torch
# inside classify_error()'s own function body, not at module level.
from cpt_decoder.evaluation.contextual_analysis import check_grammar  # noqa: E402
from cpt_decoder.evaluation.llm_judge import classify_error  # noqa: E402


# ── Text normalisation (mirrors metrics.py's normalise(), kept independent
#    on purpose so this module has no import-order dependency on metrics.py) ──
def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Per-pair substitution classification ──────────────────────────────────────
def classify_substitution(ref_word: str, hyp_word: str) -> str:
    """
    Classify a single substitution (ref_word -> hyp_word) as one of:
        "Homophone"      : hyp_word is an exact CMU-dict homophone of ref_word
                            (identical phoneme sequence, stress stripped)
        "Near-homophone"  : hyp_word's phoneme sequence is within edit
                            distance 1 of ref_word's, but not identical
        "Other"           : no phonetic relationship found (or ref_word isn't
                            in the CMU dictionary at all, e.g. OOV / proper noun)

    get_homophones() is an O(1)-ish dict lookup (cheap); get_near_homophones()
    brute-force-scans the full ~125k-entry CMU dictionary, so it's only called
    when the cheap exact check has already failed — keeps this affordable to
    run over an entire validation set.
    """
    ref_word, hyp_word = ref_word.upper(), hyp_word.upper()
    if ref_word == hyp_word:
        return "Equal"  # shouldn't normally be reached from a 'substitute' chunk

    if hyp_word in get_homophones(ref_word):
        return "Homophone"

    near = get_near_homophones(ref_word, max_distance=1)
    near_words = {w for w, dist in near if dist >= 1}  # distance-0 entries already
                                                         # covered by get_homophones above
    if hyp_word in near_words:
        return "Near-homophone"

    return "Other"


# ── Stage 3: resolve WHY a phonetically-explainable substitution happened ─────
def resolve_substitution(reference: str,
                          original_hyp_sentence: str,
                          hyp_word: str,
                          hyp_word_idx: int,
                          category: str,
                          tokenizer=None,
                          model=None,
                          use_llm: bool = False) -> Dict:
    """
    Stage 3 of the P2T framework: given a substitution already classified
    by Stage 3 Option 2 (classify_substitution(), above) as Homophone /
    Near-homophone / Other, attempt to resolve which of the framework's
    four top-level error categories (Phonological / Lexical / Contextual
    / Semantic) it actually belongs to. Escalates through:

        Option 3 (grammar) -- contextual_analysis.check_grammar(). Cheap,
        deterministic, no model load. Resolves closed-class word
        dependency-role mismatches (their/your/its/my/our/whose forced
        into a syntactic role only their homophone counterpart could
        fill). See that module's docstring for exactly what it can and
        can't catch, and why (the THERE/TOO/TWO / open-class-word
        boundary).

        Option 5 (LLM judge) -- llm_judge.classify_error(). Only
        attempted if Option 3 left the substitution unresolved AND
        use_llm=True AND a tokenizer/model were actually supplied.
        Opt-in by design: this needs a loaded model and one generate()
        call per unresolved substitution, a real cost increase a CPU
        dry run shouldn't pay unless explicitly asked for (see
        dryrun.py's CPT_LLM_ERROR_JUDGE env var, and llm_judge.py's own
        documented finding that the CPU dry-run judge model's
        *classifications*, as opposed to its plumbing, aren't yet
        trustworthy -- this wiring doesn't change that; it just makes
        Option 5 reachable end-to-end once a better judge model is
        available).

    "Other"-category substitutions (no phonetic relationship at all) are
    passed through untouched and not escalated at all: this framework's
    homophone-driven hierarchy is specifically about explaining
    phonetically-motivated errors (see this module's top docstring), not
    a general-purpose error classifier for every kind of substitution.

    Args:
        reference              : the reference SENTENCE (used only for
                                  Option 5's "Ground Truth:" field -- not
                                  needed by Option 3, which only looks at
                                  the hypothesis side's own grammar).
        original_hyp_sentence   : the UN-normalised hypothesis sentence
                                  (apostrophes intact -- see
                                  contextual_analysis.py's docstring for
                                  why normalise()'s output can't be used
                                  here).
        hyp_word                : the substituted word, original spelling.
        hyp_word_idx             : its whitespace-split index in
                                  original_hyp_sentence. Callers must
                                  ensure this indexes the ORIGINAL
                                  sentence, not the jiwer-normalised one
                                  (see analyze_pair() below for the
                                  token-count-preservation assumption
                                  this relies on, and its safety check).
        category                : classify_substitution()'s own output
                                  for this pair ("Homophone",
                                  "Near-homophone", or "Other").
        tokenizer, model        : passed straight through to
                                  llm_judge.classify_error() if Option 5
                                  is reached. Ignored otherwise.
        use_llm                 : opt-in gate for Option 5 (see above).

    Returns:
        {"stage3_category": str or None, "stage3_subcategory": str or None,
         "stage3_explanation": str or None, "stage3_method": str or None}
        All four fields are None if this substitution wasn't escalated
        (category == "Other"), or if every attempted method left it
        unresolved / unavailable.
    """
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


# ── Per-sentence-pair error breakdown ─────────────────────────────────────────
def analyze_pair(reference: str, hypothesis: str,
                  tokenizer=None, model=None, use_llm: bool = False) -> Dict:
    """
    Run jiwer's word-level alignment on a single (reference, hypothesis) pair,
    classify every substitution found (Stage 3 Option 2), and -- for
    substitutions that came back Homophone/Near-homophone -- escalate
    through Stage 3 Options 3/5 via resolve_substitution() above to
    determine the actual P2T error category.

    Args:
        reference, hypothesis : the ORIGINAL (un-normalised) sentence pair.
        tokenizer, model, use_llm : passed straight through to
                    resolve_substitution()'s Option 5 escalation. All
                    default to "off" so existing callers (and the CPU dry
                    run by default) get Option 3-only resolution with no
                    behaviour change otherwise.

    Returns:
        {
          "n_hits": int, "n_substitutions": int, "n_deletions": int, "n_insertions": int,
          "substitutions": [ {"ref": str, "hyp": str, "category": str,
                               "stage3_category": str or None,
                               "stage3_subcategory": str or None,
                               "stage3_explanation": str or None,
                               "stage3_method": str or None}, ... ],
        }
    """
    ref_norm = normalise(reference)
    hyp_norm = normalise(hypothesis)
    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words:
        return {"n_hits": 0, "n_substitutions": 0, "n_deletions": 0, "n_insertions": 0,
                "substitutions": []}

    # Stage 3's check_grammar() needs to index into the ORIGINAL (un-
    # normalised) hypothesis sentence (apostrophes intact -- see
    # contextual_analysis.py's docstring), using the SAME whitespace-split
    # index as the normalised hyp_words list above. This holds because
    # normalise() only lowercases/strips punctuation/collapses whitespace
    # -- it never drops or reorders a token -- so index i in hyp_words
    # corresponds to index i in hypothesis.split(), UNLESS punctuation
    # stripping happens to merge/split a token differently (e.g. a
    # standalone "-" between spaces collapsing away). Guard against that
    # rather than silently misindexing: only attempt Stage 3 escalation
    # when the two splits actually have matching lengths.
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
        for offset, (rw, hw) in enumerate(zip(ref_span, hyp_span)):  # 1:1 within a substitute chunk
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


# ── Full-set error category report ────────────────────────────────────────────
def error_category_report(all_refs: List[str],
                           all_hyps: List[str],
                           homo_mask: Optional[List[bool]] = None,
                           tokenizer=None,
                           model=None,
                           use_llm: bool = False) -> Dict:
    """
    Run analyze_pair() across an entire evaluation set and aggregate.

    Args:
        all_refs  : reference sentences
        all_hyps  : decoded hypothesis sentences
        homo_mask : optional boolean list (True = sentence drawn from the
                    homophone-containing subset) — if given, the substitution
                    category breakdown is also reported split by this mask,
                    which is the comparison that actually matters for deciding
                    whether the contrastive hard-negative mechanism is doing
                    its job.
        tokenizer, model : optional, forwarded to resolve_substitution()'s
                    Stage 3 Option 5 escalation (see analyze_pair()).
                    Ignored unless use_llm=True.
        use_llm   : opt-in flag for Stage 3 Option 5 (see dryrun.py's
                    CPT_LLM_ERROR_JUDGE env var). When False (the
                    default), every Homophone/Near-homophone substitution
                    is still run through Stage 3 Option 3 (grammar —
                    cheap, no model needed); only the Option 5 escalation
                    for substitutions Option 3 leaves unresolved is
                    skipped, so a CPU dry run isn't silently slowed down
                    or changed by default.

    Returns a dict with overall totals, a substitution-category breakdown,
    a Stage 3 category breakdown (of however many substitutions got
    resolved), and (if homo_mask given) the same breakdowns split by
    homo_mask.
    """
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


# ── Pretty printer ────────────────────────────────────────────────────────────
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

        # Stage 3 (Options 3/5): of the Homophone/Near-homophone substitutions
        # above, how many got resolved to an actual P2T error category, and
        # via which method. Empty unless resolve_substitution() actually
        # resolved something (Option 3 needs no opt-in; Option 5 needs
        # use_llm=True and a model — see error_category_report()'s docstring).
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
                print(f"       (resolved via — {method_str})")

    if "homophone" in report and report["overall"]["totals"]["n_substitutions"] > 0:
        h_cats = report["homophone"]["substitution_categories"]
        h_sub = report["homophone"]["totals"]["n_substitutions"]
        if h_sub > 0:
            phon_explained = h_cats.get("Homophone", 0) + h_cats.get("Near-homophone", 0)
            pct = phon_explained / h_sub * 100
            print(f"\n  -> Of substitution errors on homophone-containing sentences,")
            print(f"     {pct:.1f}% are phonetically explainable (exact or near-homophone).")
            print( "     High %: contrastive hard-negative tuning is the right lever.")
            print( "     Low %:  errors are likely scale/capacity-driven, not lexical —")
            print( "             the real Llama run is the more relevant fix, not more")
            print( "             hard-negative mining.")
    print("\n" + "=" * width + "\n")


if __name__ == "__main__":
    # ── Smoke test with toy data mirroring metrics.py's own smoke test ────────
    # Default run (no args): Stage 3 Option 3 (grammar) only -- fast, no
    # model load, exercises the full wiring end to end including the new
    # closed-class case below. Pass --with-llm to also exercise Stage 3
    # Option 5 against the cached CPU dry-run model (slower: loads
    # MODEL_NAME_DRYRUN). Kept separate from the default run rather than
    # always-on, mirroring this module's own opt-in use_llm design and
    # llm_judge.py's already-completed standalone validation of Option 5
    # in isolation -- this block's job is to prove the WIRING works, not
    # to re-litigate judge-model quality (see llm_judge.py's docstring for
    # that finding).
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
        "I COULD LABEL THIS AS MEET",          # exact homophone substitution
        "THEY FOUND THE SIGHT ON THE COAST",    # exact homophone substitution
        "WHAT REALLY MAKES A SHIP IS THE CRUNCH",  # near-homophone (CHIP -> SHIP, edit distance 1)
        "FRESH OUT THE DRYER",                  # also near-homophone (FRYER -> DRYER, edit distance 1, not unrelated)
        "THE TRADITIONAL BANANA PAN OFTEN STAYS ON THE SHELF",  # unrelated substitution (CHIP -> BANANA)
        "THEIR IS A CAT ON THE MAT",             # homophone substitution (THERE->THEIR) that Stage 3
                                                  # Option 3 SHOULD resolve: THEIR forced into nsubj
                                                  # (poss-only word) -> Contextual/closed_class_dependency_mismatch
    ]
    homo_mask = [True, True, False, False, False, True]

    use_llm = "--with-llm" in _sys.argv
    tok, mdl = None, None
    if use_llm:
        from cpt_decoder.model import load_tokenizer, MODEL_NAME_DRYRUN
        from transformers import AutoModelForCausalLM
        print(f"\n--with-llm: loading judge model ({MODEL_NAME_DRYRUN}) for Stage 3 Option 5...")
        tok = load_tokenizer(MODEL_NAME_DRYRUN)
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME_DRYRUN)
        mdl.resize_token_embeddings(len(tok))
        mdl.eval()

    report = error_category_report(refs, hyps, homo_mask, tokenizer=tok, model=mdl, use_llm=use_llm)
    print_error_report(report, "Smoke test — error_analysis.py")

    # Explicit assertion on the one case this smoke test is actually meant
    # to prove: THERE->THEIR must come back Homophone (Option 2) AND get
    # resolved to Contextual via Option 3 (Option 5 never needed for it).
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
