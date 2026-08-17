"""P2T LoRA Decoder: Evaluation Metrics"""

import re
from typing import List, Dict, Tuple, Optional
import jiwer
import sacrebleu

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_error_rate(references: List[str],
                    hypotheses: List[str]) -> float:
    """Word Error Rate (WER)."""
    refs = [normalise(r) for r in references]
    hyps = [normalise(h) for h in hypotheses]
    return jiwer.wer(refs, hyps)

def character_error_rate(references: List[str],
                          hypotheses: List[str]) -> float:
    """Character Error Rate (CER)."""
    refs = [normalise(r) for r in references]
    hyps = [normalise(h) for h in hypotheses]
    return jiwer.cer(refs, hyps)

def bleu4_score(references: List[str],
                hypotheses: List[str]) -> float:
    """BLEU-4 score (sacrebleu implementation)."""
    refs_norm = [[normalise(r) for r in references]]
    hyps_norm = [normalise(h) for h in hypotheses]
    result = sacrebleu.corpus_bleu(hyps_norm, refs_norm)
    return result.score / 100.0

def exact_match(references: List[str],
                hypotheses: List[str]) -> float:
    """Exact Match Accuracy: proportion of perfectly decoded sentences."""
    correct = sum(
        normalise(r) == normalise(h)
        for r, h in zip(references, hypotheses)
    )
    return correct / len(references) if references else 0.0

def homophone_disambiguation_rate(references: List[str],
                                   hypotheses: List[str],
                                   homophone_pairs: List[Tuple[str, str]]) -> float:
    """Novel metric: proportion of homophone-containing sentences where"""
    homo_set = set()
    for w1, w2 in homophone_pairs:
        homo_set.add((w1.lower(), w2.lower()))
        homo_set.add((w2.lower(), w1.lower()))

    total_homo_words = 0
    correct_homo = 0

    for ref, hyp in zip(references, hypotheses):
        ref_words = normalise(ref).split()
        hyp_words = normalise(hyp).split()
        min_len = min(len(ref_words), len(hyp_words))
        for i in range(min_len):
            rw, hw = ref_words[i], hyp_words[i]
            if (rw, hw) in homo_set or rw == hw:
                if any(rw in (p[0], p[1]) for p in homophone_pairs):
                    total_homo_words += 1
                    if rw == hw:
                        correct_homo += 1

    if total_homo_words == 0:
        return 1.0
    return correct_homo / total_homo_words

def evaluate(references: List[str],
             hypotheses: List[str],
             label: str = "Overall") -> Dict[str, float]:
    """Compute all metrics for a set of reference/hypothesis pairs."""
    if not references:
        return {}

    results = {
        "label":        label,
        "n_sentences":  len(references),
        "WER":          word_error_rate(references, hypotheses),
        "CER":          character_error_rate(references, hypotheses),
        "BLEU-4":       bleu4_score(references, hypotheses),
        "Exact_Match":  exact_match(references, hypotheses),
    }
    return results

def stratified_evaluate(all_refs: List[str],
                         all_hyps: List[str],
                         homo_mask: List[bool]) -> Dict[str, Dict]:
    """Run evaluation on three groups simultaneously:"""
    homo_refs  = [r for r, m in zip(all_refs, homo_mask) if m]
    homo_hyps  = [h for h, m in zip(all_hyps, homo_mask) if m]
    non_refs   = [r for r, m in zip(all_refs, homo_mask) if not m]
    non_hyps   = [h for h, m in zip(all_hyps, homo_mask) if not m]

    return {
        "overall":       evaluate(all_refs, all_hyps, label="Overall"),
        "homophone":     evaluate(homo_refs, homo_hyps, label="Homophone"),
        "non_homophone": evaluate(non_refs, non_hyps, label="Non-Homophone"),
    }

def print_results(results: Dict[str, Dict],
                  title: str = "Evaluation Results") -> None:
    """Print a formatted evaluation table."""
    width = 62
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
    print(f"  {'Subset':<22} {'N':>6}  {'WER':>7}  {'CER':>7}  {'BLEU-4':>7}  {'EM':>7}")
    print("-" * width)

    for key in ["overall", "homophone", "non_homophone"]:
        r = results.get(key, {})
        if not r:
            continue
        print(f"  {r['label']:<22} {r['n_sentences']:>6}  "
              f"{r['WER']*100:>6.2f}%  "
              f"{r['CER']*100:>6.2f}%  "
              f"{r['BLEU-4']:>7.4f}  "
              f"{r['Exact_Match']*100:>6.2f}%")

    print("=" * width)

    if "homophone" in results and "non_homophone" in results:
        h = results["homophone"]
        n = results["non_homophone"]
        if h and n:
            wer_gap = (h["WER"] - n["WER"]) * 100
            em_gap  = (n["Exact_Match"] - h["Exact_Match"]) * 100
            print(f"\n  >> WER gap  (homo - non-homo): {wer_gap:+.2f}%")
            print(f"  >> EM gap   (non-homo - homo): {em_gap:+.2f}%")
    print()

def save_results(results: Dict[str, Dict],
                 path: str,
                 model_name: str = "baseline") -> None:
    """Save results to a CSV for tracking across experiments."""
    import csv, datetime
    rows = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    for key, r in results.items():
        if not r:
            continue
        rows.append({
            "timestamp":    timestamp,
            "model":        model_name,
            "subset":       r["label"],
            "n":            r["n_sentences"],
            "WER":          round(r["WER"] * 100, 4),
            "CER":          round(r["CER"] * 100, 4),
            "BLEU4":        round(r["BLEU-4"], 4),
            "ExactMatch":   round(r["Exact_Match"] * 100, 4),
        })
    file_exists = False
    try:
        with open(path, "r") as f:
            file_exists = bool(f.read(1))
    except FileNotFoundError:
        pass

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved to: {path}")

if __name__ == "__main__":
    print("Running metrics smoke test...\n")

    refs  = [
        "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF",
        "THEY FOUND THE SITE ON THE COAST",
        "I COULD LABEL THIS AS MEAT",
        "FRESH OUT THE FRYER",
        "WHAT REALLY MAKES A CHIP IS THE CRUNCH",
    ]
    hyps_perfect = refs[:]
    hyps_errors  = [
        "THE TRADITIONAL CHIP PAN OFTEN STAYS ON A SHELF",
        "THEY FOUND THE SIGHT ON THE COAST",
        "I COULD LABEL THIS AS MEET",
        "FRESH OUT THE FRYER",
        "WHAT REALLY MAKES A CHIP IS THE CRUNCH",
    ]
    homo_mask = [False, True, True, False, False]

    print("--- Perfect predictions ---")
    r1 = stratified_evaluate(refs, hyps_perfect, homo_mask)
    print_results(r1, "Perfect Baseline")

    print("--- With homophone errors ---")
    r2 = stratified_evaluate(refs, hyps_errors, homo_mask)
    print_results(r2, "With Errors")
