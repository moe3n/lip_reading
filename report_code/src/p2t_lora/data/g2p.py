"""P2T LoRA Decoder: Grapheme-to-Phoneme (G2P) Pipeline"""

import re
import nltk
from typing import List, Optional

try:
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()

def strip_stress(phonemes: List[str]) -> List[str]:
    """Remove numeric stress markers from ARPAbet phonemes."""
    return [re.sub(r"[012]", "", p) for p in phonemes]

def keep_stress(phonemes: List[str]) -> List[str]:
    """Return phonemes as-is (stress preserved)."""
    return phonemes

def word_to_phonemes(word: str,
                     stress: bool = True,
                     variant: int = 0) -> Optional[List[str]]:
    """Look up a word in the CMU dict."""
    key = word.lower().strip()
    variants = CMU_DICT.get(key)
    if variants is None:
        return None
    phones = variants[min(variant, len(variants) - 1)]
    return keep_stress(phones) if stress else strip_stress(phones)

def word_to_phoneme_str(word: str, stress: bool = True) -> Optional[str]:
    """Return phonemes as a space-separated string."""
    phones = word_to_phonemes(word, stress=stress)
    return " ".join(phones) if phones else None

def sentence_to_arpabet(sentence: str,
                         stress: bool = True,
                         word_boundary: str = "<space>",
                         unk_token: str = "<UNK>") -> str:
    """Convert a full sentence to an ARPAbet phoneme sequence."""
    clean = re.sub(r"[^A-Za-z\s']", "", sentence).upper().strip()
    words = clean.split()

    phoneme_groups = []
    for word in words:
        phones = word_to_phonemes(word, stress=stress)
        if phones:
            phoneme_groups.append(" ".join(phones))
        else:
            phoneme_groups.append(unk_token)

    seq = f" {word_boundary} ".join(phoneme_groups)
    return f"<SOS> {seq} <EOS>"

def sentence_to_phoneme_list(sentence: str,
                              stress: bool = True) -> List[str]:
    """Return all phonemes for a sentence as a flat list (no word boundaries)."""
    clean = re.sub(r"[^A-Za-z\s']", "", sentence).upper().strip()
    words = clean.split()
    phonemes = []
    for word in words:
        phones = word_to_phonemes(word, stress=stress)
        if phones:
            phonemes.extend(phones)
    return phonemes

def coverage_check(sentences: List[str]) -> dict:
    """Check what proportion of words in a list of sentences are"""
    total_words = 0
    covered = 0
    unknown_words = set()

    for sent in sentences:
        clean = re.sub(r"[^A-Za-z\s']", "", sent).upper().strip()
        words = clean.split()
        for w in words:
            total_words += 1
            if w.lower() in CMU_DICT:
                covered += 1
            else:
                unknown_words.add(w)

    return {
        "total_words": total_words,
        "covered": covered,
        "coverage_pct": covered / total_words * 100 if total_words else 0,
        "unknown_count": len(unknown_words),
        "unknown_sample": sorted(list(unknown_words))[:20],
    }

def phoneme_edit_distance(seq1: List[str], seq2: List[str]) -> int:
    """Compute edit distance (Levenshtein) between two phoneme sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

if __name__ == "__main__":
    print("=== G2P Pipeline Demo ===\n")

    test_sentences = [
        "THROUGH WHAT THEY CALL A KNIFE BLOCK",
        "THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF",
        "I COULD LABEL THIS ON THE INGREDIENTS AS MEAT",
    ]

    for sent in test_sentences:
        arpabet = sentence_to_arpabet(sent)
        print(f"Input   : {sent}")
        print(f"ARPAbet : {arpabet}\n")

    print("=== Homophone Demo ===\n")
    pairs = [("meet", "meat"), ("sight", "site"), ("their", "there"), ("knight", "night")]
    for w1, w2 in pairs:
        p1 = word_to_phoneme_str(w1)
        p2 = word_to_phoneme_str(w2)
        match = "✓ IDENTICAL" if p1 == p2 else "✗ DIFFERENT"
        print(f"  {w1:10s} -> {p1}")
        print(f"  {w2:10s} -> {p2}  {match}\n")
