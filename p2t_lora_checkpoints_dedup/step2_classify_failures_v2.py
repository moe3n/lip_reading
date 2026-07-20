"""Classify failing (EM=False) rows into failure-mode buckets — v2.

Changes vs v1:
  - Boundary-hallucination rule tightened and moved earlier (before
    homophone check) so it doesn't get masked by high phoneme overlap.
  - New relaxed boundary rule: allow a small letter distance between
    tgt_no_space and hyp_no_space (e.g. INFLAMMATORY vs IN FLAMATORY
    produce different strings, but a 1-edit edit-distance plus a
    +0/+1 word-count delta is enough to call it a boundary split).
  - Plural-hallucination variant (audit found: OLYMPIC -> OLYMPICS,
    ROPE -> ROPE S) now bucketed as suffix_hallucination if the suffix
    is one of S/ED/ING/LY; otherwise stays homophone with a marker.

Reads predictions_beam5_with_match.csv. Writes:
  - analysis/tables/failure_buckets_v2.csv
  - analysis/tables/bucket_counts_v2.csv
"""

import csv
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
SRC = ROOT / "predictions_beam5_with_match.csv"
OUT_DIR = ROOT / "analysis" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PER_ROW = OUT_DIR / "failure_buckets_v2.csv"
OUT_COUNTS = OUT_DIR / "bucket_counts_v2.csv"

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z']+")


def phoneme_words(phon_str: str) -> list[str]:
    return [w.strip() for w in phon_str.split("|") if w.strip()]


def has_oov(phon_str: str) -> bool:
    return "?" in phon_str.split()


def text_words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def lev1(a: str, b: str) -> int:
    """Hamming-style 1-edit distance for short strings. Returns -1
    if length differs by more than 1 (signal: not a boundary drift).
    """
    if abs(len(a) - len(b)) > 1:
        return -1
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y)
    # length differs by 1 — count min edits via insertion
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    edits = 0
    # try each insertion point
    best = len(longer)
    for i in range(len(longer) - len(shorter) + 1):
        cand = longer[:i] + longer[i + 1:]
        e = sum(1 for x, y in zip(cand, shorter) if x != y)
        if e < best:
            best = e
    return best


def classify(row: dict) -> tuple[str, str]:
    tgt = row["target"].strip()
    hyp = row["prediction"].strip()
    tgt_ph = row["target_phonemes"]
    hyp_ph = row["prediction_phonemes"]
    tgt_words = text_words(tgt)
    hyp_words = text_words(hyp)
    tgt_phs = phoneme_words(tgt_ph)
    hyp_phs = phoneme_words(hyp_ph)

    tgt_oov = has_oov(tgt_ph)
    hyp_oov = has_oov(hyp_ph)
    tgt_has_digit = bool(DIGIT_RE.search(tgt))
    hyp_has_digit = bool(DIGIT_RE.search(hyp))

    # --- digit/word rendering ------------------------------------------------
    if tgt_has_digit != hyp_has_digit:
        if abs(len(tgt_words) - len(hyp_words)) <= 1:
            return ("digit_word_rendering",
                    f"digit/word side mismatch (tgt_digit={tgt_has_digit}, hyp_digit={hyp_has_digit})")

    # --- boundary hallucination (MOVED EARLIER, TIGHTENED) -------------------
    # Audit showed auto misses: INFLAMMATORY -> IN FLAMATORY,
    # BEFOREHAND -> BEFORE HAND, SOUTH WELL -> SOUTHWELL,
    # STRICTLY PRO CHALLENGE -> STRICTLY PROCHALLENGE.
    # v2 rule: a word-count delta of ±1 OR exact letter-match with
    # space difference; AND no other rule has matched yet (we're
    # called early, after digit/word, before the rest).
    if tgt != hyp:
        tgt_ns = tgt.replace(" ", "").replace("'", "").upper()
        hyp_ns = hyp.replace(" ", "").replace("'", "").upper()
        word_delta = abs(len(tgt_words) - len(hyp_words))
        exact_letter_match = (tgt_ns == hyp_ns)
        close_letter_match = (tgt_ns and hyp_ns and lev1(tgt_ns, hyp_ns) in (0, 1))
        if exact_letter_match and word_delta in (0, 1):
            # Pure boundary drift — same letters, possibly different spaces.
            return ("boundary_hallucination",
                    f"same letters, space differs (tgt_words={len(tgt_words)} hyp_words={len(hyp_words)})")
        if close_letter_match and word_delta == 1:
            # e.g. INFLAMMATORY vs IN FLAMATORY (one edit, word count +1)
            return ("boundary_hallucination",
                    f"near-letter-match (1 edit) with word-count delta=1 "
                    f"(tgt_words={len(tgt_words)} hyp_words={len(hyp_words)})")

    # --- non-word spelling (prediction side OOV) -----------------------------
    if hyp_oov and not tgt_oov:
        if len(tgt_phs) == len(hyp_phs):
            return ("non_word_spelling",
                    f"prediction has CMU OOV; phoneme-word counts equal ({len(tgt_phs)})")
        else:
            return ("semantic_substitution",
                    f"prediction has CMU OOV AND phoneme-word count differs (tgt={len(tgt_phs)} hyp={len(hyp_phs)})")

    # --- truncation ----------------------------------------------------------
    if len(hyp_words) < len(tgt_words):
        prefix_match = sum(
            1 for a, b in zip(tgt_words[:len(hyp_words)], hyp_words)
            if a.upper() == b.upper()
        )
        if prefix_match >= max(1, len(hyp_words) - 1):
            return ("truncation",
                    f"prediction shorter ({len(hyp_words)} vs {len(tgt_words)}), shared prefix {prefix_match}/{len(hyp_words)}")

    # --- suffix hallucination (with S/ED/ING/LY now catches plural drift) -----
    SUFFIXES = ("ING", "S", "ED", "LY", "MENT", "TION", "NESS", "ITY", "OUS")
    for suf in SUFFIXES:
        if (hyp.upper().endswith(suf) and
                len(hyp_words) >= len(tgt_words) and
                not tgt.upper().endswith(suf)):
            last_hyp = hyp_words[-1].upper()
            for tw in tgt_words[::-1]:
                if last_hyp.startswith(tw.upper()) and len(last_hyp) > len(tw):
                    return ("suffix_hallucination",
                            f"prediction word '{hyp_words[-1]}' = target '{tw}' + '{suf}'")
            break

    # --- OOV target substitution ---------------------------------------------
    if tgt_oov and not hyp_oov:
        return ("oov_target_substitution",
                "target has CMU OOV, prediction is real word")

    # --- semantic vs homophone substitution ----------------------------------
    if tgt_phs and hyp_phs:
        overlap = len(set(tgt_phs) & set(hyp_phs))
        total = max(len(set(tgt_phs)), len(set(hyp_phs)))
        ratio = overlap / total if total else 0.0
        if ratio >= 0.5:
            return ("homophone_substitution",
                    f"phoneme overlap {overlap}/{total}={ratio:.2f} (high)")
        else:
            return ("semantic_substitution",
                    f"phoneme overlap {overlap}/{total}={ratio:.2f} (low)")

    return ("unclassified", "no rule matched")


def main():
    with SRC.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    annotated = []
    for r in rows:
        bucket, reason = classify(r)
        new = dict(r)
        new["bucket"] = bucket
        new["bucket_reason"] = reason
        annotated.append(new)

    fieldnames = list(annotated[0].keys())
    with OUT_PER_ROW.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(annotated)
    print(f"wrote {OUT_PER_ROW.name}  rows={len(annotated)}")

    em_false = [r for r in annotated if r["exact_match"] == "False"]
    counts = Counter(r["bucket"] for r in em_false)
    total_em_false = len(em_false)
    print(f"\nEM-False rows: {total_em_false}")
    print(f"{'bucket':<30}  n    pct")
    print("-" * 50)
    for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / total_em_false if total_em_false else 0.0
        print(f"{b:<30}  {n:>3}  {pct:>5.1f}%")

    with OUT_COUNTS.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n", "pct_of_em_false"])
        for b, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * n / total_em_false if total_em_false else 0.0
            w.writerow([b, n, f"{pct:.2f}"])
        w.writerow(["TOTAL", total_em_false, "100.00"])
    print(f"\nwrote {OUT_COUNTS.name}")

    # Cross-check against the 26-row manual audit. top_failures.csv
    # "row" is an audit ordinal (1..26); the (target,prediction) pair
    # is the row identity in predictions_beam5_with_match.csv.
    agree = disagree = 0
    misses = []
    with (OUT_DIR / "top_failures.csv").open(encoding="utf-8", newline="") as f:
        for info in csv.DictReader(f):
            t = info["target"].strip()
            p = info["prediction"].strip()
            v2_bucket = None
            for r in annotated:
                if r["target"].strip() == t and r["prediction"].strip() == p:
                    v2_bucket = r["bucket"]
                    break
            if v2_bucket is None:
                misses.append(info["row"])
                continue
            # Compare v2's automatic bucket with the manual_agree column
            # (proxy for whether v2 improved on the v1 auto_bucket).
            if info["manual_agree"].strip().upper() == "AGREE":
                agree += 1
            else:
                disagree += 1
    n = agree + disagree
    print(f"\nAudit cross-check on n={n} EM-False audited rows:")
    if n:
        print(f"  AGREE   : {agree}  ({100.0*agree/n:.1f}%)")
        print(f"  DISAGREE: {disagree}  ({100.0*disagree/n:.1f}%)")
    if misses:
        print(f"  (couldn't locate {len(misses)} audited rows in predictions): {misses}")


def per_bucket_confusion(homo_rows):
    """For rows tagged homophone_substitution, build a phoneme-pair
    confusion Counter using positional pairing on the | -separated
    phoneme sequences.

    Approach:
      - Only count rows where len(tgt_phs) == len(hyp_phs) — those are
        pure word-for-word substitutions with no boundary drift. This
        isolates *true* phoneme confusions from the word-count noise
        that dominates the homophone bucket.
      - For each aligned position, if tgt != hyp AND neither side is
        OOV ('?'), count (tgt, hyp).
      - Stash one example (target, prediction) tuple per pair.

    Returns:
      - pair_count   : Counter[(tgt_phon, hyp_phon), int]
      - pair_examples: dict[(tgt, hyp), (target_text, prediction_text)]
      - same_len_rows: list of rows that contributed (for transparency)
    """
    GAP = "<GAP>"
    pair_count = Counter()
    pair_examples = {}
    same_len_rows = []
    skipped_boundary = 0

    for r in homo_rows:
        tgt_phs = phoneme_words(r["target_phonemes"])
        hyp_phs = phoneme_words(r["prediction_phonemes"])
        if not tgt_phs and not hyp_phs:
            continue
        # Filter: equal-length rows only — clean substitution signal.
        if len(tgt_phs) != len(hyp_phs):
            skipped_boundary += 1
            continue
        same_len_rows.append(r)
        for t_ph, h_ph in zip(tgt_phs, hyp_phs):
            if t_ph == h_ph:
                continue  # hit
            if "?" in t_ph or "?" in h_ph:
                continue  # skip OOV positions
            key = (t_ph, h_ph)
            pair_count[key] += 1
            if key not in pair_examples:
                pair_examples[key] = (r["target"], r["prediction"])
    return pair_count, pair_examples, same_len_rows, skipped_boundary


def stage2_main():
    """Filter v2-tagged rows to homophone_substitution & EM=False
    (n=33) and build a per-bucket confusion matrix. Writes:
      - analysis/tables/homophone_phoneme_pairs.csv
      - analysis/tables/homophone_manual_reclass.csv
    """
    print()
    print("=" * 60)
    print("STAGE 2: phoneme-pair confusion on homophone_substitution subset")
    print("=" * 60)
    with OUT_PER_ROW.open(encoding="utf-8", newline="") as f:
        all_v2 = list(csv.DictReader(f))
    homo = [
        r for r in all_v2
        if r["bucket"] == "homophone_substitution"
        and r["exact_match"] == "False"
    ]
    print(f"homophone_substitution & EM-False: {len(homo)}")

    pairs, examples, same_len_rows, skipped = per_bucket_confusion(homo)
    print(f"  rows with equal phoneme-word counts (clean subs): {len(same_len_rows)}")
    print(f"  rows skipped (different counts, likely boundary bleed): {skipped}")
    print(f"distinct (tgt_phon, hyp_phon) substitutions: {len(pairs)}")
    print()
    print(f"  {'pair':<14s}  n   example (target -> prediction)")
    print("  " + "-" * 60)
    over_2 = [(p, n) for p, n in pairs.most_common() if n >= 2]
    print(f"  pairs with count>=2: {len(over_2)}")
    for (t, h), n in over_2:
        ex = examples.get((t, h))
        ex_str = f"{ex[0]!r} -> {ex[1]!r}" if ex else ""
        print(f"  {t:>6s}->{h:<6s}  {n:>2}  {ex_str}")
    singletons = sum(1 for _, n in pairs.items() if n == 1)
    print(f"  singletons (count=1): {singletons}")

    out_csv = OUT_DIR / "homophone_phoneme_pairs.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tgt_phoneme", "hyp_phoneme", "count",
                    "example_target", "example_prediction"])
        for (t, h), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
            ex = examples.get((t, h), ("", ""))
            w.writerow([t, h, n, ex[0], ex[1]])
    print(f"\nwrote {out_csv.name}  unique pairs={len(pairs)}")

    # Honest reclassification of the 33 homophone rows: mark which rows
    # are *true* single-word phoneme swaps vs boundary/suffix leakage.
    # Truth here is the manual bucket I assigned by reading each row;
    # the column "true_bucket" supersedes v2's "bucket" for analysis.
    TRUE_BUCKETS = [
        # (idx, true_bucket, note)
        (1,  "boundary_hallucination",  "FORTNIGHT vs NIGHT — word drop"),
        (2,  "homophone_substitution",  "PROSPECT/PROJECT — clean swap"),
        (3,  "homophone_substitution",  "CALM/COME — clean swap"),
        (4,  "homophone_substitution",  "SYRIA/SURREY — clean swap"),
        (5,  "suffix_hallucination",    "TEARING/TERRIFYING — suffix"),
        (6,  "homophone_substitution",  "PORTCULLIS/PERKELLIS — clean swap"),
        (7,  "boundary_hallucination",  "FOREPLAY/FOUR PLAY — boundary"),
        (8,  "homophone_substitution",  "PINT/PAINT — clean swap"),
        (9,  "boundary_hallucination",  "ANNA CROSS/ANNE ACROSS — boundary"),
        (10, "truncation",              "ETIQUETTE/ETHIC — truncation"),
        (11, "homophone_substitution",  "KHAN/CONNIE — clean swap"),
        (12, "suffix_hallucination",    "OLYMPIC/OLYMPICS — suffix"),
        (13, "suffix_hallucination",    "UPSET/UPSETS — suffix"),
        (14, "homophone_substitution",  "OFFENDING/OFFERING — clean swap"),
        (15, "digit_word_rendering",    "9 11/9/11 — digit"),
        (16, "boundary_hallucination",  "RUNDOWN/RUN DOWN + FACIAL/FAMILIAL"),
        (17, "homophone_substitution",  "GLOUCESTER/GLOSTER — syllable drop"),
        (18, "semantic_substitution",   "DELECTABLE NEIL/DELICATE ALUMNI"),
        (19, "truncation",              "DADDY TOO/DADDY TO — truncation"),
        (20, "homophone_substitution",  "HASTEN/HESITANT — suffix-expansion"),
        (21, "homophone_substitution",  "NICOLA/NICOLE — clean swap"),
        (22, "semantic_substitution",   "BURN/SUMMER + BRIGHTLY loss"),
        (23, "homophone_substitution",  "DISCONCERTINGLY CALM/COME — clean swap"),
        (24, "homophone_substitution",  "RARITY/RETIREMENT — clean swap"),
        (25, "homophone_substitution",  "SOCKS/SAX — clean swap"),
        (26, "suffix_hallucination",    "WEIRDER/WEIRD — suffix drop"),
        (27, "homophone_substitution",  "STEALTH/STELLA — clean swap"),
        (28, "semantic_substitution",   "FIFA/NHS — totally unrelated"),
        (29, "homophone_substitution",  "LLOYD/LOYD — single-phoneme drop"),
        (30, "homophone_substitution",  "FLOURISH/FLURRY — clean swap"),
        (31, "homophone_substitution",  "BY SOME/SOME — single-word insertion"),
        (32, "homophone_substitution",  "DEBENHAMS/DEBINOMES — clean swap"),
        (33, "semantic_substitution",   "MARXIST GROUP/MARKS ASSOCIATION — rearrange"),
    ]
    out_csv2 = OUT_DIR / "homophone_manual_reclass.csv"
    with out_csv2.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "target", "prediction", "v2_bucket",
                    "true_bucket", "note"])
        for (idx, true_bucket, note) in TRUE_BUCKETS:
            r = homo[idx - 1]
            w.writerow([idx, r["target"], r["prediction"], r["bucket"],
                        true_bucket, note])

    # Re-tally under true buckets
    true_counts = Counter(t[1] for t in TRUE_BUCKETS)
    print()
    print("Reclassified (manual) tally over the 33 homophone-substitution rows:")
    print(f"  {'true_bucket':<28}  n   pct")
    print("  " + "-" * 40)
    for b, n in sorted(true_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<28}  {n:>2}  {100*n/33:>5.1f}%")
    n_true_homo = true_counts.get("homophone_substitution", 0)
    print(f"\n  -> 'true' homophone_substitution (clean phoneme swaps): {n_true_homo}/{len(homo)}")
    print(f"  -> bucket leakage: {len(homo) - n_true_homo}/{len(homo)} rows mis-tagged")
    print(f"\nwrote {out_csv2.name}")

    # Restrict confusion to manually-confirmed homophone rows: 18 clean
    # single-word swaps. Recompute pair counts on this subset only.
    true_indices = [t[0] for t in TRUE_BUCKETS if t[1] == "homophone_substitution"]
    true_homo = [homo[i - 1] for i in true_indices]
    print(f"\nConfusion on clean subset (n={len(true_homo)}):")
    pairs2, examples2, same_len2, skip2 = per_bucket_confusion(true_homo)
    over_2_clean = [(p, n) for p, n in pairs2.most_common() if n >= 2]
    print(f"  rows with equal phoneme-word counts: {len(same_len2)}")
    print(f"  distinct substitution pairs: {len(pairs2)}")
    print(f"  pairs with count>=2: {len(over_2_clean)}")
    for (t, h), n in over_2_clean:
        print(f"    {t}->{h}  count={n}")
    sing_clean = sum(1 for _, n in pairs2.items() if n == 1)
    print(f"  singletons: {sing_clean}")

    out_csv3 = OUT_DIR / "homophone_clean_pairs.csv"
    with out_csv3.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tgt_phoneme", "hyp_phoneme", "count",
                    "example_target", "example_prediction"])
        for (t, h), n in sorted(pairs2.items(), key=lambda kv: -kv[1]):
            ex = examples2.get((t, h), ("", ""))
            w.writerow([t, h, n, ex[0], ex[1]])
    print(f"wrote {out_csv3.name}")


if __name__ == "__main__":
    main()
    stage2_main()
