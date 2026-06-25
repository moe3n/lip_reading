"""
CPT Decoder — Hard Negative Generator
=======================================
Generates phonetically near-identical alternative sentences
for contrastive training.

For each sentence containing a homophone word, we:
  1. Look up all CMU Dictionary words that share the same phoneme sequence
  2. Filter to alternatives that are semantically distinct (different spelling)
  3. Substitute them into the sentence to create hard negative candidates

These hard negatives are used in the contrastive loss:
    L_contrast = max(0, margin - score(correct) + score(hard_negative))
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import nltk

try:
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()
except LookupError:
    nltk.download("cmudict", quiet=True)
    from nltk.corpus import cmudict
    CMU_DICT = cmudict.dict()


# ── Build reverse phoneme index ───────────────────────────────────────────────
def build_phoneme_to_words_index(stress: bool = False) -> Dict[str, List[str]]:
    """
    Build a reverse index: phoneme_sequence -> [word1, word2, ...]
    This lets us quickly find all words that sound the same.

    Args:
        stress : If True, include stress markers in the key

    Returns:
        Dict mapping phoneme strings to lists of words
    """
    index = defaultdict(list)
    for word, pronunciations in CMU_DICT.items():
        for phones in pronunciations:
            if not stress:
                key = " ".join(re.sub(r"[012]", "", p) for p in phones)
            else:
                key = " ".join(phones)
            index[key].append(word.upper())
    # deduplicate
    return {k: list(set(v)) for k, v in index.items()}


# Build index once at module load
PHONEME_INDEX = build_phoneme_to_words_index(stress=False)


# ── Homophone lookup ──────────────────────────────────────────────────────────
def get_homophones(word: str) -> List[str]:
    """
    Return all words that share the same phoneme sequence as the input word.
    The input word itself is excluded from the results.

    Example:
        get_homophones("meet") -> ["MEAT", "METE"]
        get_homophones("sight") -> ["CITE", "SITE"]
    """
    key_word = word.lower()
    if key_word not in CMU_DICT:
        return []

    results = set()
    for phones in CMU_DICT[key_word]:
        phoneme_key = " ".join(re.sub(r"[012]", "", p) for p in phones)
        alts = PHONEME_INDEX.get(phoneme_key, [])
        for alt in alts:
            if alt.upper() != word.upper():
                results.add(alt.upper())

    return sorted(results)


def get_near_homophones(word: str, max_distance: int = 1) -> List[Tuple[str, int]]:
    """
    Return words with phoneme sequences within `max_distance` edits
    of the input word's phoneme sequence. Useful for catching
    near-miss confusions beyond exact homophones.

    Returns list of (word, distance) tuples sorted by distance.
    """
    from data.g2p import phoneme_edit_distance, word_to_phonemes
    key_word = word.lower()
    if key_word not in CMU_DICT:
        return []

    ref_phones = word_to_phonemes(word, stress=False)
    if not ref_phones:
        return []

    candidates = []
    seen = set()
    # Only check words of similar length to keep search tractable
    ref_len = len(ref_phones)
    for cand_word, pronunciations in CMU_DICT.items():
        if cand_word == key_word:
            continue
        for phones in pronunciations:
            phones_clean = [re.sub(r"[012]", "", p) for p in phones]
            if abs(len(phones_clean) - ref_len) > max_distance:
                continue
            dist = phoneme_edit_distance(ref_phones, phones_clean)
            if dist <= max_distance and cand_word.upper() not in seen:
                candidates.append((cand_word.upper(), dist))
                seen.add(cand_word.upper())

    return sorted(candidates, key=lambda x: x[1])


# ── Sentence-level hard negatives ─────────────────────────────────────────────
def generate_hard_negatives(sentence: str,
                              max_per_word: int = 3,
                              max_total: int = 5) -> List[Dict]:
    """
    Generate hard negative sentences for a given input.

    For each word in the sentence that has homophones, substitute
    each homophone alternative to create a near-miss sentence.

    Args:
        sentence    : Input English sentence (uppercase)
        max_per_word: Max alternatives per homophone word
        max_total   : Max total hard negatives to return

    Returns:
        List of dicts:
          {
            "sentence"     : original sentence,
            "negative"     : hard negative sentence,
            "changed_word" : original word,
            "alt_word"     : substituted word,
            "position"     : word index,
          }
    """
    clean = re.sub(r"[^\w\s']", "", sentence).upper().strip()
    words = clean.split()
    negatives = []

    for i, word in enumerate(words):
        homophones = get_homophones(word)
        if not homophones:
            continue
        for alt in homophones[:max_per_word]:
            new_words = words[:i] + [alt] + words[i+1:]
            negatives.append({
                "sentence":     sentence,
                "negative":     " ".join(new_words),
                "changed_word": word,
                "alt_word":     alt,
                "position":     i,
            })
        if len(negatives) >= max_total:
            break

    return negatives[:max_total]


def build_homophone_lookup_table() -> Dict[str, List[str]]:
    """
    Build a complete lookup table of all homophones in the CMU Dict.
    Returns {word -> [homophone1, homophone2, ...]}

    Used to pre-filter which sentences need contrastive training.
    """
    table = {}
    for word in CMU_DICT.keys():
        alts = get_homophones(word)
        if alts:
            table[word.upper()] = alts
    return table


def sentence_contains_homophone(sentence: str,
                                  lookup: Optional[Dict] = None) -> bool:
    """Check if a sentence contains at least one word with homophones."""
    if lookup is None:
        lookup = build_homophone_lookup_table()
    clean = re.sub(r"[^\w\s']", "", sentence).upper().strip()
    return any(w in lookup for w in clean.split())


if __name__ == "__main__":
    print("=" * 55)
    print("  HARD NEGATIVE GENERATOR DEMO")
    print("=" * 55)

    # Homophone pairs demo
    test_words = ["meet", "sight", "their", "knight", "weather", "piece", "write"]
    print("\n── Homophones found in CMU Dict ──\n")
    for word in test_words:
        alts = get_homophones(word)
        if alts:
            print(f"  {word.upper():<12} → {', '.join(alts)}")

    # Sentence-level hard negatives
    print("\n── Hard Negatives for example sentences ──\n")
    sentences = [
        "THEY FOUND THE SITE ON THE COAST",
        "I COULD LABEL THIS ON THE INGREDIENTS AS MEAT",
        "WHAT REALLY MAKES A CHIP IS THE CRUNCH",
    ]
    for sent in sentences:
        negs = generate_hard_negatives(sent)
        print(f"  Original : {sent}")
        if negs:
            for neg in negs:
                print(f"  Negative : {neg['negative']}  "
                      f"[{neg['changed_word']} → {neg['alt_word']}]")
        else:
            print("  Negative : (none — no homophones found)")
        print()

    # Stats
    print("── Building full homophone lookup table... ──")
    table = build_homophone_lookup_table()
    print(f"  Words with homophones in CMU Dict: {len(table):,}")
    sample = list(table.items())[:8]
    for word, alts in sample:
        print(f"  {word:<15} → {', '.join(alts[:4])}")
