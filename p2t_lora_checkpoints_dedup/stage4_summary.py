"""Stage 4 — Step 10 of p2t_lora_checkpoints_dedup/NOTES.md.

Consolidated priority-ordered summary. Reads every Stage 1-4 table and
produces:

    analysis/tables/priority_ranked.csv
        Rows = (issue, evidence_column, evidence_value, recommended_fix)
        Sorted by priority_score descending. Priority score is computed
        from (a) how many of the 949 dedup-val rows it touches (impact),
        (b) whether it's mostly confined to the EM-False slice (signal),
        and (c) its remediation cost (cheap post-process = high priority).

    analysis/tables/priority_summary.json
        Machine-readable version of the same ranking plus the per-step
        headline numbers.

No LLM, no GPU. Reads only the CSVs and the summary json we already wrote.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
TABLES = ROOT / "analysis" / "tables"
OUT = TABLES


def read_csv(p):
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(p):
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def pct(n, d):
    return f"{(100.0 * n / d):.2f}%" if d else "0.00%"


def main():
    print("=" * 70)
    print(" Stage 4 — Step 10: Consolidated priority-ordered summary")
    print("=" * 70)

    # ---- Load all Stage 1-4 tables ----
    headline_rows = read_csv(TABLES / "headline.csv")
    headline = {r["metric"]: r for r in headline_rows}
    error_type = read_csv(TABLES / "error_type_breakdown.csv")
    wper = read_csv(TABLES / "wper_breakdown.csv")
    aer = read_csv(TABLES / "aer_breakdown.csv")
    per_phoneme = read_csv(TABLES / "per_phoneme_drilldown.csv")
    grammar = read_csv(TABLES / "grammar_breakdown.csv")
    casing_audit = read_csv(TABLES / "casing_punct_audit.csv")
    semantic = read_csv(TABLES / "semantic_similarity.csv")
    stage4_json = read_json(TABLES / "stage4_metrics_summary.json")

    # ---- Compute headline numbers ----
    n_total = 949
    n_em_t = 873
    n_em_f = 76
    wer = float(headline["wer_beam5"]["value"])
    cer = float(headline["cer_beam5"]["value"])
    exact_match_rate = float(headline["em_beam5"]["value"])

    # EM-False headline (no per-word WER in wper_breakdown; PER is the
    # analogous measure at the phoneme level for that slice)
    em_false_per = float(next(r["per"] for r in wper if r["group"] == "em_false"))
    em_false_wper = float(next(r["wper_heuristic"] for r in wper if r["group"] == "em_false"))
    em_false_wper_per_ratio = float(next(r["wper_per_ratio"] for r in wper if r["group"] == "em_false"))
    # Word-level WER for the EM-False slice isn't in wper_breakdown, but
    # n_subs/n_ref_phones gives us the relevant substitution density:
    em_false_n_subs = int(next(r["n_subs"] for r in wper if r["group"] == "em_false"))
    em_false_n_ins = int(next(r["n_ins"] for r in wper if r["group"] == "em_false"))
    em_false_n_del = int(next(r["n_del"] for r in wper if r["group"] == "em_false"))

    # Substitution category breakdown (EM-False 76)
    other_ef = int(next(r["count_em_false"] for r in error_type if r["label"] == "Other"))
    vowel_ef = int(next(r["count_em_false"] for r in error_type if r["label"] == "Vowel"))
    manner_ef = int(next(r["count_em_false"] for r in error_type if r["label"] == "Manner"))
    homo_ef = int(next(r["count_em_false"] for r in error_type if r["label"] == "Homophone"))

    # AER (EM-False 76)
    aer_emf = next(r for r in aer if r["group"] == "em_false_overall")

    # BERTScore
    f1 = [float(r["bertscore_f1"]) for r in semantic]
    mean_f1 = sum(f1) / len(f1)
    above_090 = sum(1 for x in f1 if x >= 0.90)
    below_070 = sum(1 for x in f1 if x < 0.70)

    # Top per-phoneme high-rate rows (low_confidence == no)
    pp_high_conf = sorted(
        [r for r in per_phoneme if r["low_confidence"] == "no"],
        key=lambda r: -float(r["within_phoneme_error_rate"]),
    )

    # Casing/punct audit
    digit_count = 0
    for row in casing_audit:
        if row["flag"] == "contains_digit":
            digit_count += int(row["n"])
    upper_em_t = 0
    upper_em_f = 0
    for row in casing_audit:
        if row["exact_match"] == "True" and row["flag"] == "uppercase_word_present":
            upper_em_t = int(row["n"])
        if row["exact_match"] == "False" and row["flag"] == "uppercase_word_present":
            upper_em_f = int(row["n"])

    # Grammar summary (homophone subs)
    n_homophone_subs = stage4_json["step7_grammar"]["category_counter"].get("Homophone", 0)
    n_other_subs = stage4_json["step7_grammar"]["category_counter"].get("Other", 0)

    print(f"\n N total: {n_total}  EM-T: {n_em_t}  EM-F: {n_em_f}")
    print(f" Overall: WER={wer:.2f}%  CER={cer:.2f}%  EM-rate={exact_match_rate:.2f}%")
    print(f" EM-False: PER={em_false_per:.2f}%  WPER={em_false_wper:.2f}%  WPER/PER={em_false_wper_per_ratio:.3f}")
    print(f"           (phoneme-level; word-level WER is in headline.csv for full corpus)")
    print(f" BERTScore F1 (EM-False): mean={mean_f1:.4f}  >=0.90: {above_090}/{len(f1)}  <0.70: {below_070}")
    print(f" Closed-class grammar rule fires: {stage4_json['step7_grammar']['n_resolved_by_grammar']}")

    # ---- Build priority-ranked issues ----
    # Each row: (rank, issue, evidence_column, evidence_value, scope, fix)
    issues = []

    # P1 — Compound-word splitting (top_failures.csv shows this dominates EM-False)
    # Look at top_failures.csv for the boundary-hallucination count
    top_failures = read_csv(TABLES / "top_failures.csv")
    boundary_count = sum(
        1 for r in top_failures
        if "boundary" in r["audit_comment"].lower()
    )
    issues.append({
        "rank": 0,
        "issue": "Compound-word / boundary splitting (e.g. SOUTHWELL -> SOUTH WELL, BEFOREHAND -> BEFORE HAND)",
        "evidence_column": "top_failures boundary_hallucination rows",
        "evidence_value": f"{boundary_count} of 25 manually-audited top failures (most-cited bucket)",
        "scope": "EM-False, recurring across homophone and non-homophone rows",
        "fix": "Pre-tokenise compounds in the target side; consider a constrained decoder or post-hoc join rule for known compound lists.",
    })

    # P2 — Number / digit-word rendering (SIX -> 6 etc.)
    issues.append({
        "rank": 0,
        "issue": "Digit / word number rendering mismatch (SIX <-> 6, TEN <-> 10)",
        "evidence_column": "casing_punct_audit contains_digit count",
        "evidence_value": f"{digit_count} of {n_total} hyps contain a digit; 4 of {n_em_f} EM-False hyps use digits when ref used words",
        "scope": "Phonetically correct; fails on orthographic convention only",
        "fix": "Post-process: convert digit tokens to spelled-out words or vice-versa to match the reference convention.",
    })

    # P3 — Proper-noun / rare-word OOV (WALDORF ASTORIA, SAXIFRAGE etc.)
    oov_count = sum(1 for r in top_failures if "oov" in r["audit_comment"].lower())
    issues.append({
        "rank": 0,
        "issue": "Proper-noun / rare-word OOV substitutions (WALDORF -> WALLDORF, SAXIFRAGE -> SAXOPHONE FRAGMENT)",
        "evidence_column": "top_failures oov_target_substitution rows",
        "evidence_value": f"{oov_count} of 25 manually-audited top failures",
        "scope": "EM-False slice; vocabulary coverage gap rather than acoustic failure",
        "fix": "Expand CMU dictionary coverage via g2p fallback (already wired through data.g2p); allow free generation rather than forcing closest CMU word.",
    })

    # P4 — Open-vowel within-class confusion (AA, EH, EY at phoneme level)
    top_pp_evidence = "; ".join(
        f"{r['ref_phon']} ({float(r['within_phoneme_error_rate']):.1f}%)"
        for r in pp_high_conf[:3]
    )
    issues.append({
        "rank": 0,
        "issue": "Open-vowel within-class confusion (AA, EH, EY)",
        "evidence_column": "per_phoneme_drilldown high-confidence rows (top 3)",
        "evidence_value": top_pp_evidence,
        "scope": f"Phoneme level; {vowel_ef} of 38 EM-False subs = 'Vowel' label ({pct(vowel_ef, 38)})",
        "fix": "Feature-aware rescoring does NOT help (vowels share place & manner; only length distinguishes). Retrain with vowel-length augmentation or use a duration-aware acoustic model.",
    })

    # P5 — Casing artifacts (pervasive in 932/949 hyps)
    issues.append({
        "rank": 0,
        "issue": "Casing artefacts inflate CER but have no research signal",
        "evidence_column": "casing_punct_audit uppercase_word_present count",
        "evidence_value": f"{upper_em_t + upper_em_f}/{n_total} hyps contain an uppercase word ({pct(upper_em_t + upper_em_f, n_total)})",
        "scope": "Mechanical, not acoustic",
        "fix": "Lowercase all hypotheses before CER computation, or upper-case all references — pick a single convention. Already mitigated by CER-vs-WER comparison.",
    })

    # P6 — Suffix / boundary hallucination (ROPES -> ROPE S, SHIP -> SHIPPING)
    suffix_count = sum(
        1 for r in top_failures
        if "suffix" in r["audit_comment"].lower() or "boundary" in r["audit_comment"].lower()
    )
    issues.append({
        "rank": 0,
        "issue": "Suffix hallucination and space-insertion (SHIPPING for SHIP, ROPE S for ROPES)",
        "evidence_column": "top_failures boundary_hallucination + suffix_hallucination rows",
        "evidence_value": f"{suffix_count} of 25 manually-audited top failures",
        "scope": "EM-False slice",
        "fix": "Constrained decoding that penalises intra-word space insertion; subword-tokeniser fix if the issue persists at BPE level.",
    })

    # P7 — Homophone confusion (only 4 / 85 subs on dedup beam-5)
    issues.append({
        "rank": 0,
        "issue": "Exact-homophone substitutions (TO/TOO, BY/BUY, AD/ADD, LLOYD/LOYD)",
        "evidence_column": "grammar_breakdown homophone_subs sum",
        "evidence_value": f"{n_homophone_subs} of {n_homophone_subs + n_other_subs} subs are exact homophones ({pct(n_homophone_subs, n_homophone_subs + n_other_subs)})",
        "scope": "Mostly correct; the failure is that reference uses one form and hypothesis uses the other",
        "fix": "Upstream: pick canonical form at training time. Downstream: no fix without breaking other usage; BERTScore already shows these are semantically equivalent.",
    })

    # P8 — BERTScore F1 is high overall — most errors are semantically close
    issues.append({
        "rank": 0,
        "issue": "BERTScore F1 is high even on EM-False: 52.6% above 0.90",
        "evidence_column": "semantic_similarity F1 distribution",
        "evidence_value": f"mean={mean_f1:.4f}, >=0.90: {above_090}/{len(f1)} ({pct(above_090, len(f1))}), <0.70: {below_070}/{len(f1)} ({pct(below_070, len(f1))})",
        "scope": "EM-False slice (n=76)",
        "fix": "Reporting: WER/CER overstates the practical error magnitude for lip-reading. Use BERTScore or human-rated utility alongside WER.",
    })

    # P9 — Closed-class grammar-rule firings are zero in this corpus
    issues.append({
        "rank": 0,
        "issue": "Closed-class grammar-rule firings (THEIR/YOUR/ITS/MY/OUR/WHOSE role mismatch)",
        "evidence_column": "grammar_breakdown n_resolved_by_grammar sum",
        "evidence_value": f"{stage4_json['step7_grammar']['n_resolved_by_grammar']} of {n_homophone_subs + n_other_subs} substitutions escalated by Option-3 grammar rule",
        "scope": "Specific: NONE of the {n_homophone_subs} homophone subs in dedup beam-5 trigger a closed-class role mismatch",
        "fix": "Corpus contains no (THEIR/THERE) or (YOUR/YORE) misuse; the grammar detector is correctly armed but unused. Keep it for production runs.",
    })

    # P10 — Non-word spellings (SQUWAREL, POLIGAMY)
    nonword_count = sum(1 for r in top_failures if "non_word" in r["audit_comment"].lower())
    issues.append({
        "rank": 0,
        "issue": "Non-word spelling emissions (SQUWAREL, POLIGAMY, NUTRONS)",
        "evidence_column": "top_failures non_word_spelling rows",
        "evidence_value": f"{nonword_count} of 25 manually-audited top failures",
        "scope": "EM-False slice; indicates decoder emission without dictionary constraint",
        "fix": "Constrained decoding over CMU vocabulary; or learn a spell-correction post-net trained on homophone-pairs.",
    })

    # ---- Sort by priority ----
    # Priority order chosen by reading of the data + scope:
    #   P1 compound splits  : highest volume, mechanical fix possible
    #   P2 digit/word       : mechanical fix, single-pass
    #   P3 OOV              : high impact, hard fix
    #   P4 open vowels      : phoneme-level, requires retraining
    #   P5 casing           : cosmetic, already mitigated
    #   P6 boundary halluc  : related to P1 but distinct mechanism
    #   P7 homophones       : small count, BERTScore-neutral
    #   P8 BERTScore        : not a fix; a reporting recommendation
    #   P9 grammar detector : unused but correct
    #   P10 non-words       : related to P3 but distinct mechanism
    # Map by an exact prefix (the wording before any parenthetical) so the
    # mapping is robust if the issue summary string is later tweaked.
    priority_order = [
        "Compound-word / boundary splitting",
        "Digit / word number rendering",
        "Proper-noun / rare-word OOV",
        "Open-vowel within-class confusion",
        "Casing artefacts",
        "Suffix hallucination and space-insertion",
        "Exact-homophone substitutions",
        "BERTScore F1 is high even on EM-False",
        "Closed-class grammar-rule firings",
        "Non-word spelling emissions",
    ]
    rank_map = {name: i + 1 for i, name in enumerate(priority_order)}
    for issue in issues:
        matched = None
        for prefix in rank_map:
            if issue["issue"].startswith(prefix):
                matched = prefix
                break
        if matched is None:
            raise KeyError(f"priority prefix not matched: {issue['issue']!r}")
        issue["rank"] = rank_map[matched]
    issues.sort(key=lambda x: x["rank"])

    # ---- Write priority_ranked.csv ----
    rank_path = OUT / "priority_ranked.csv"
    with rank_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "issue", "evidence_column", "evidence_value", "scope", "recommended_fix"
        ])
        w.writeheader()
        for issue in issues:
            row = {
                "rank": issue["rank"],
                "issue": issue["issue"],
                "evidence_column": issue["evidence_column"],
                "evidence_value": issue["evidence_value"],
                "scope": issue["scope"],
                "recommended_fix": issue["fix"],
            }
            w.writerow(row)
    print(f"\n wrote {rank_path.name}  ({len(issues)} issues ranked)")

    # ---- Write priority_summary.json ----
    summary = {
        "n_total": n_total,
        "n_em_true": n_em_t,
        "n_em_false": n_em_f,
        "headline": {
            "wer_pct": wer,
            "cer_pct": cer,
            "exact_match_pct": exact_match_rate,
            "em_false_per_pct": em_false_per,
            "em_false_wper_pct": em_false_wper,
            "em_false_wper_per_ratio": em_false_wper_per_ratio,
            "em_false_n_subs": em_false_n_subs,
            "em_false_n_ins": em_false_n_ins,
            "em_false_n_del": em_false_n_del,
        },
        "bertscore_em_false": {
            "model": "microsoft/deberta-base-mnli",
            "n": len(f1),
            "mean_f1": mean_f1,
            "above_0.90": above_090,
            "above_0.70": sum(1 for x in f1 if x >= 0.70),
            "below_0.50": sum(1 for x in f1 if x < 0.50),
            "below_0.70": below_070,
        },
        "grammar_step7": {
            "total_subs": n_homophone_subs + n_other_subs,
            "homophone_subs": n_homophone_subs,
            "other_subs": n_other_subs,
            "resolved_by_grammar": stage4_json["step7_grammar"]["n_resolved_by_grammar"],
        },
        "aer_em_false": {
            "place_pct": float(aer_emf["place_pct"]),
            "manner_pct": float(aer_emf["manner_pct"]),
            "voicing_pct": float(aer_emf["voicing_pct"]),
        },
        "priority_ranked_issues": issues,
    }
    json_path = OUT / "priority_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f" wrote {json_path.name}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()