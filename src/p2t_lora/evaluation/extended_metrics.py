"""
P2T LoRA Decoder — Extended Metrics (mirrors Mira Fleite's prompting-baseline suite)
================================================================================
Adds the metric layers documented in Mira Fleite's progress reports
(`Other student's work/mira's/Summary_report_of_first_3_weeks.pdf`) that
`metrics.py`/`error_analysis.py` didn't already cover, so her prompting-only
results and our fine-tuned/prompting-only results land in directly
comparable tables:

    sid_breakdown()          — word- and char-level Substitution/Insertion/
                                Deletion counts (her Section 2.2).
    allophonic_error_rate()  — error breakdown by place/manner/voicing (AER,
                                her Section 3.3) via a hardcoded ARPAbet
                                feature table — same "manual feature mapping"
                                approach she used, not panphon.
    weighted_per()            — WPER (her Section 3.2), two variants:
                                "heuristic" (same ARPAbet table, weights
                                place=0.4/manner=0.4/voicing=0.2 as she used)
                                and "panphon" (real articulatory feature
                                vectors via the panphon library).
    grammar_error_rate()      — typos/grammar/casing/punctuation breakdown
                                via language_tool_python (her Section 3.4).
                                Needs a local Java runtime -- language_tool_
                                python runs a bundled LanguageTool server on
                                first use. Raises a clear error if Java isn't
                                found rather than silently falling back to
                                the public API (that would send decoded
                                sentences to a third-party server).
    semantic_similarity()     — BERTScore (her Section 3.5).

PER itself is NOT duplicated here — `run_baseline.py`'s `phoneme_error_rate()`
and this module's `weighted_per(..., method="panphon")` both already do
plain-vs-weighted PER; a third copy isn't needed.

All functions take sentence lists (`refs`, `hyps` — English text) and do their
own phonemization via `data/g2p.py`, so callers never touch ARPAbet directly.
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import jiwer

# Same sys.path dance as error_analysis.py in this package (see that file's
# comment for why both entries are needed regardless of invocation style).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_P2T_LORA_DIR = os.path.dirname(_THIS_DIR)
_SRC_DIR = os.path.dirname(_P2T_LORA_DIR)
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, _P2T_LORA_DIR)

from p2t_lora.data import g2p  # noqa: E402
from p2t_lora.evaluation.metrics import normalise  # noqa: E402


# ── ARPAbet feature table ────────────────────────────────────────────────────
# (place, manner, voicing) per stress-stripped ARPAbet phoneme. Vowels have no
# place of articulation in the consonant sense; manner="vowel" is enough to
# distinguish "vowel vs consonant" confusions in the AER breakdown, mirroring
# Mira's own finding that the model confuses vowels with consonants.
ARPABET_FEATURES: Dict[str, Tuple[str, str, str]] = {
    # Vowels — all voiced, no consonantal place/manner.
    **{v: ("vowel", "vowel", "voiced") for v in
       ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
        "IH", "IY", "OW", "OY", "UH", "UW")},
    # Stops
    "P": ("bilabial", "stop", "voiceless"),
    "B": ("bilabial", "stop", "voiced"),
    "T": ("alveolar", "stop", "voiceless"),
    "D": ("alveolar", "stop", "voiced"),
    "K": ("velar", "stop", "voiceless"),
    "G": ("velar", "stop", "voiced"),
    # Affricates
    "CH": ("postalveolar", "affricate", "voiceless"),
    "JH": ("postalveolar", "affricate", "voiced"),
    # Fricatives
    "F": ("labiodental", "fricative", "voiceless"),
    "V": ("labiodental", "fricative", "voiced"),
    "TH": ("dental", "fricative", "voiceless"),
    "DH": ("dental", "fricative", "voiced"),
    "S": ("alveolar", "fricative", "voiceless"),
    "Z": ("alveolar", "fricative", "voiced"),
    "SH": ("postalveolar", "fricative", "voiceless"),
    "ZH": ("postalveolar", "fricative", "voiced"),
    "HH": ("glottal", "fricative", "voiceless"),
    # Nasals
    "M": ("bilabial", "nasal", "voiced"),
    "N": ("alveolar", "nasal", "voiced"),
    "NG": ("velar", "nasal", "voiced"),
    # Liquids / glides
    "L": ("alveolar", "liquid", "voiced"),
    "R": ("alveolar", "liquid", "voiced"),
    "W": ("labiovelar", "glide", "voiced"),
    "Y": ("palatal", "glide", "voiced"),
}

# ARPAbet -> IPA, for feeding panphon (which works on IPA segments). Diphthongs
# map to 2-character IPA sequences; panphon's segmenter handles that natively.
ARPABET_TO_IPA: Dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "g", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def _phonemize(sentence: str) -> List[str]:
    """Sentence -> flat stress-stripped ARPAbet phoneme list, via our own G2P
    (OOV words are silently skipped by g2p.sentence_to_phoneme_list — same
    behaviour PER already accepts elsewhere in this project)."""
    return g2p.sentence_to_phoneme_list(sentence, stress=False)


def _to_ipa(phonemes: List[str]) -> str:
    """ARPAbet list -> one IPA string, unknown symbols dropped."""
    return "".join(ARPABET_TO_IPA.get(p, "") for p in phonemes)


# ── SID breakdown (word- and char-level) ─────────────────────────────────────
def sid_breakdown(refs: List[str], hyps: List[str]) -> Dict[str, Dict]:
    """Substitution/Insertion/Deletion/Hit counts at word and char level.
    Thin wrapper around jiwer.process_words/process_characters — both already
    project dependencies, already used internally by error_analysis.py.

    jiwer compares tokens by exact string match, so refs/hyps are normalised
    (lowercase, punctuation stripped -- metrics.py's normalise(), same
    convention as every other metric in this project) before alignment.
    Skipping this would count "THE"/"The." as a substitution purely on case
    and punctuation, wildly overstating every count here."""
    refs_n = [normalise(r) for r in refs]
    hyps_n = [normalise(h) for h in hyps]
    word_out = jiwer.process_words(refs_n, hyps_n)
    char_out = jiwer.process_characters(refs_n, hyps_n)
    return {
        "word": {"hits": word_out.hits, "substitutions": word_out.substitutions,
                  "insertions": word_out.insertions, "deletions": word_out.deletions,
                  "wer": word_out.wer},
        "char": {"hits": char_out.hits, "substitutions": char_out.substitutions,
                  "insertions": char_out.insertions, "deletions": char_out.deletions,
                  "cer": char_out.cer},
    }


# ── Shared ARPAbet-level alignment (feeds both AER and heuristic WPER) ──────
def _phoneme_substitutions(refs: List[str], hyps: List[str]) -> Tuple[List[Tuple[str, str]], int, int, int]:
    """Align ref/hyp phoneme sequences (jiwer, phonemes-as-tokens) across the
    whole ref/hyp list. Returns (substitution pairs, n_insertions, n_deletions,
    n_hits) pooled over every sentence."""
    ref_ph = [" ".join(_phonemize(r)) for r in refs]
    hyp_ph = [" ".join(_phonemize(h)) for h in hyps]
    out = jiwer.process_words(ref_ph, hyp_ph)

    subs: List[Tuple[str, str]] = []
    n_ins = n_del = n_hits = 0
    for ref_words, hyp_words, chunks in zip(out.references, out.hypotheses, out.alignments):
        for c in chunks:
            if c.type == "substitute":
                ref_span = ref_words[c.ref_start_idx:c.ref_end_idx]
                hyp_span = hyp_words[c.hyp_start_idx:c.hyp_end_idx]
                for rw, hw in zip(ref_span, hyp_span):
                    subs.append((rw, hw))
            elif c.type == "insert":
                n_ins += c.hyp_end_idx - c.hyp_start_idx
            elif c.type == "delete":
                n_del += c.ref_end_idx - c.ref_start_idx
            elif c.type == "equal":
                n_hits += c.ref_end_idx - c.ref_start_idx
    return subs, n_ins, n_del, n_hits


# ── AER (Allophonic Error Rate) ──────────────────────────────────────────────
def allophonic_error_rate(refs: List[str], hyps: List[str]) -> Dict:
    """Error breakdown by which articulatory feature differs (place/manner/
    voicing), matching Mira's AER (Section 3.3). Only substitutions are
    classified (insertions/deletions have no "which feature differs" — they
    have no matching reference/hypothesis phoneme to compare)."""
    subs, n_ins, n_del, n_hits = _phoneme_substitutions(refs, hyps)

    counts = Counter()
    n_classified = 0
    for rp, hp in subs:
        rf = ARPABET_FEATURES.get(rp)
        hf = ARPABET_FEATURES.get(hp)
        if rf is None or hf is None:
            continue  # unmapped symbol (shouldn't happen with in-vocab ARPAbet)
        n_classified += 1
        if rf[0] != hf[0]:
            counts["place"] += 1
        if rf[1] != hf[1]:
            counts["manner"] += 1
        if rf[2] != hf[2]:
            counts["voicing"] += 1

    return {
        "n_substitutions": len(subs), "n_classified": n_classified,
        "n_insertions": n_ins, "n_deletions": n_del, "n_hits": n_hits,
        "place": counts["place"], "manner": counts["manner"], "voicing": counts["voicing"],
        "place_pct": counts["place"] / n_classified * 100 if n_classified else 0.0,
        "manner_pct": counts["manner"] / n_classified * 100 if n_classified else 0.0,
        "voicing_pct": counts["voicing"] / n_classified * 100 if n_classified else 0.0,
    }


def _dominant_feature(ref: str, hyp: str) -> Optional[str]:
    """For a single (ref, hyp) sentence pair, which articulatory dimension
    accounts for the most phoneme substitutions: "vowel" (the reference
    phoneme has no consonant place/manner/voicing to compare), "manner",
    "place", or "voicing". None if there are no classifiable substitutions."""
    subs, *_ = _phoneme_substitutions([ref], [hyp])
    counts = Counter()
    for rp, hp in subs:
        rf, hf = ARPABET_FEATURES.get(rp), ARPABET_FEATURES.get(hp)
        if rf is None or hf is None:
            continue
        if rf[1] == "vowel":
            counts["vowel"] += 1
        elif rf[1] != hf[1]:
            counts["manner"] += 1
        elif rf[0] != hf[0]:
            counts["place"] += 1
        elif rf[2] != hf[2]:
            counts["voicing"] += 1
    return counts.most_common(1)[0][0] if counts else None


# Sentence length thresholds for the Long/Short categories below, taken from
# this project's own corpus (sentphonemepairs_LRS2_original.csv) 15th/85th
# percentiles (4 and 11 words) -- not copied from Mira Fleite's report, whose
# thresholds came from her own curated sample of the same corpus and landed
# close to these by coincidence, not derivation.
SHORT_WORDS = 4
LONG_WORDS = 11


def error_type_breakdown(refs: List[str], hyps: List[str],
                          truncated: Optional[List[bool]] = None) -> List[str]:
    """Classify each (ref, hyp) pair into exactly one dominant error type,
    adapted from Mira Fleite's curated-set categories (Vowel/Manner/
    Hallucination/Homophone/Long/Short/Truncation, her Week 4 report) --
    but derived entirely from our own data and thresholds. Her categories
    were built around a prompting-only setup; a fine-tuned model's error
    profile is different, so the category definitions here are ours, not
    a copy of hers, even though the category names are kept for direct
    comparison.

    Precedence (first match wins): exact match -> truncation -> hallucination
    (no word in the hypothesis lines up with the reference at all) ->
    homophone -> vowel -> manner/place/voicing -> long -> short -> other.

    truncated: optional per-row flag for whether the INPUT prompt was
    truncated at generation time -- this module has no way to know the
    token budget used, so callers that care about this category must
    supply it (see zero-shot/run_baseline.py's required_input_len()).
    Defaults to all False.
    """
    from p2t_lora.evaluation.error_analysis import classify_substitution

    truncated = truncated or [False] * len(refs)
    labels = []
    for ref, hyp, trunc in zip(refs, hyps, truncated):
        ref_n, hyp_n = normalise(ref), normalise(hyp)
        if ref_n == hyp_n:
            labels.append("Exact match")
            continue
        if trunc:
            labels.append("Truncation")
            continue

        word_out = jiwer.process_words([ref_n], [hyp_n])
        if word_out.hits == 0:
            labels.append("Hallucination")
            continue

        ref_words, hyp_words = ref_n.split(), hyp_n.split()
        homo_hit = any(
            classify_substitution(rw, hw) == "Homophone"
            for c in word_out.alignments[0] if c.type == "substitute"
            for rw, hw in zip(ref_words[c.ref_start_idx:c.ref_end_idx],
                              hyp_words[c.hyp_start_idx:c.hyp_end_idx])
        )
        if homo_hit:
            labels.append("Homophone")
            continue

        dominant = _dominant_feature(ref, hyp)
        if dominant == "vowel":
            labels.append("Vowel")
            continue
        if dominant is not None:
            labels.append("Manner")
            continue

        if len(ref_words) > LONG_WORDS:
            labels.append("Long")
        elif len(ref_words) < SHORT_WORDS:
            labels.append("Short")
        else:
            labels.append("Other")
    return labels


def error_type_summary(labels: List[str]) -> Dict[str, Dict]:
    """Count + percentage per label from error_type_breakdown(), same shape
    as Mira's "Error Type / Number of Samples / Percentage" table."""
    counts = Counter(labels)
    n = len(labels)
    return {label: {"count": c, "pct": c / n * 100 if n else 0.0}
            for label, c in counts.most_common()}


def top_confusions(refs: List[str], hyps: List[str], category: str = "Homophone",
                    n: int = 10) -> List[Tuple[Tuple[str, str], int]]:
    """Most frequent (ref_word, hyp_word) substitution pairs in the given
    category ("Homophone" or "Near-homophone") -- Mira's "Top N Homophone
    Confusions" table (to->too, when->win, ...), built from our own data."""
    from p2t_lora.evaluation.error_analysis import classify_substitution

    counts = Counter()
    for ref, hyp in zip(refs, hyps):
        ref_n, hyp_n = normalise(ref), normalise(hyp)
        ref_words, hyp_words = ref_n.split(), hyp_n.split()
        word_out = jiwer.process_words([ref_n], [hyp_n])
        for c in word_out.alignments[0]:
            if c.type != "substitute":
                continue
            for rw, hw in zip(ref_words[c.ref_start_idx:c.ref_end_idx],
                              hyp_words[c.hyp_start_idx:c.hyp_end_idx]):
                if classify_substitution(rw, hw) == category:
                    counts[(rw, hw)] += 1
    return counts.most_common(n)


# ── WPER (Weighted Phoneme Error Rate) ───────────────────────────────────────
_HEURISTIC_WEIGHTS = {"place": 0.4, "manner": 0.4, "voicing": 0.2}  # same as Mira's heuristic matrix


def _heuristic_wper(refs: List[str], hyps: List[str]) -> float:
    """Weighted PER using the same ARPAbet feature table as AER: each
    substitution costs the fraction of (place/manner/voicing) that differ,
    weighted 0.4/0.4/0.2; insertions/deletions cost 1 (fully wrong), hits
    cost 0. Normalised by total reference phoneme count (same convention as
    plain PER)."""
    subs, n_ins, n_del, n_hits = _phoneme_substitutions(refs, hyps)

    weighted_errors = float(n_ins + n_del)
    for rp, hp in subs:
        rf = ARPABET_FEATURES.get(rp)
        hf = ARPABET_FEATURES.get(hp)
        if rf is None or hf is None:
            weighted_errors += 1.0  # unmapped symbol: treat as a full miss
            continue
        cost = sum(w for (a, b, w) in zip(rf, hf, _HEURISTIC_WEIGHTS.values()) if a != b)
        weighted_errors += cost

    total_ref_phones = n_hits + n_del + len(subs)
    return weighted_errors / total_ref_phones if total_ref_phones else 0.0


def _panphon_wper(refs: List[str], hyps: List[str]) -> float:
    """Weighted PER via panphon's real articulatory feature vectors
    (panphon.distance.Distance.feature_error_rate), operating on whole-
    sentence IPA strings built from our own ARPAbet phonemization.

    Windows note: panphon's FeatureTable reads a bundled CSV without an
    explicit encoding; on a system whose locale isn't UTF-8 this raises
    UnicodeDecodeError. Fix: run with PYTHONUTF8=1 set (or `python -X utf8`)
    -- see requirements-cpt-decoder.txt's panphon entry.
    """
    try:
        import panphon.distance as pp_distance
    except UnicodeDecodeError as e:
        raise RuntimeError(
            "panphon failed to load its feature table (Windows locale/"
            "encoding issue). Re-run with PYTHONUTF8=1 set in the "
            "environment, e.g. `set PYTHONUTF8=1` (cmd) or "
            "`$env:PYTHONUTF8=\"1\"` (PowerShell) before python."
        ) from e

    dist = pp_distance.Distance()
    ref_ipa = [_to_ipa(_phonemize(r)) for r in refs]
    hyp_ipa = [_to_ipa(_phonemize(h)) for h in hyps]
    return dist.feature_error_rate(hyp_ipa, ref_ipa)


def weighted_per(refs: List[str], hyps: List[str], method: str = "heuristic") -> float:
    """WPER, matching Mira's Section 3.2. method: "heuristic" (free, no new
    dependency, her "Heuristic matrix" option) or "panphon" (her "PanPhon"
    option, needs the panphon package + PYTHONUTF8=1 on Windows)."""
    if method == "heuristic":
        return _heuristic_wper(refs, hyps)
    if method == "panphon":
        return _panphon_wper(refs, hyps)
    raise ValueError(f"weighted_per: method must be 'heuristic' or 'panphon', got {method!r}")


# ── Grammar error rate ────────────────────────────────────────────────────────
def grammar_error_rate(hyps: List[str]) -> Dict:
    """Typos/grammar/casing/punctuation breakdown via language_tool_python
    (her Section 3.4). Needs a local Java runtime -- language_tool_python
    starts a bundled LanguageTool server on first use and raises if Java
    isn't found. We don't fall back to the public API: that would send every
    decoded sentence to a third-party server, which isn't this function's
    call to make silently."""
    import language_tool_python

    try:
        tool = language_tool_python.LanguageTool("en-US")
    except Exception as e:
        raise RuntimeError(
            "language_tool_python couldn't start its local LanguageTool "
            "server -- this needs a local Java runtime (JRE 17+). Install "
            "Java and re-run; the public API is intentionally not used as "
            "a fallback (it would send every hypothesis sentence to a "
            "third-party server)."
        ) from e

    type_counts = Counter()
    n_with_errors = 0
    total_errors = 0
    try:
        for h in hyps:
            matches = tool.check(h)
            if matches:
                n_with_errors += 1
            total_errors += len(matches)
            for m in matches:
                type_counts[m.category or "OTHER"] += 1
    finally:
        tool.close()

    n = len(hyps)
    return {
        "n_samples": n, "n_with_errors": n_with_errors,
        "pct_with_errors": n_with_errors / n * 100 if n else 0.0,
        "total_errors": total_errors,
        "avg_errors_per_sample": total_errors / n if n else 0.0,
        "by_type": dict(type_counts),
    }


# ── Semantic similarity (BERTScore) ──────────────────────────────────────────
def semantic_similarity(refs: List[str], hyps: List[str], lang: str = "en") -> float:
    """Mean BERTScore F1 between references and hypotheses (her Section 3.5).
    First call downloads the scoring model (~400MB) -- expected to be slow
    once, fast after."""
    from bert_score import score as bert_score

    _, _, f1 = bert_score(hyps, refs, lang=lang, verbose=False)
    return f1.mean().item()


if __name__ == "__main__":
    # ── Self-check against real corpus rows (not fabricated examples) ──────
    from p2t_lora.data import loader as data_loader

    corpus = data_loader.load_original_phoneme_text_pairs()["sentence"].tolist()
    identical = corpus[:5]
    ref, hyp = corpus[:5], corpus[5:10]

    assert sid_breakdown(identical, identical)["word"]["substitutions"] == 0
    assert weighted_per(identical, identical, method="heuristic") == 0.0
    assert weighted_per(ref, hyp, method="heuristic") > 0.0

    print("SID:", sid_breakdown(ref, hyp))
    print("AER:", allophonic_error_rate(ref, hyp))
    print("WPER heuristic:", weighted_per(ref, hyp, method="heuristic"))
    try:
        print("WPER panphon:", weighted_per(ref, hyp, method="panphon"))
    except RuntimeError as e:
        print("WPER panphon skipped:", e)
    print("OK: identical rows score zero error, distinct rows score non-zero.")
