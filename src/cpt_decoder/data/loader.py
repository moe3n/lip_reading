"""
CPT Decoder — Data Loader
=========================
Loads the LRS2 phoneme-text dataset and the homophone/non-homophone
sentence partitions. Handles both the full dataset (when available)
and the current partial files.
"""

import pandas as pd
import os
import re
from typing import Tuple, Optional


# ── Paths ─────────────────────────────────────────────────────────────────────
def _find_data_dir() -> str:
    """
    Locate the folder holding the LRS2 CSVs by walking up from this file
    and checking known sibling/ancestor locations. This avoids hardcoding
    a fixed number of '..' hops, which breaks every time the project
    folders get reorganised.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    marker = "sentences_with_homophones_37374.csv"
    candidate_names = ("data", "uploads")

    cur = here
    for _ in range(6):
        cur = os.path.dirname(cur)
        for name in candidate_names:
            candidate = os.path.join(cur, name)
            if os.path.isfile(os.path.join(candidate, marker)):
                return candidate

    # Fall back to the legacy assumption (kept so the function never
    # raises here; load_* calls below will raise a clear FileNotFoundError
    # instead if this guess is also wrong).
    return os.path.join(here, "..", "..", "..", "data")


DATA_DIR = _find_data_dir()

FILES = {
    "full":             os.path.join(DATA_DIR, "sentphonemepairs_LRS2_original.csv"),
    "with_homophones":  os.path.join(DATA_DIR, "sentences_with_homophones_37374.csv"),
    "without_homophones": os.path.join(DATA_DIR, "sentences_without_homophones_10790.csv"),
    # Added 21 Jun 2026: full 48,164-sentence corpus with REAL (not
    # self-generated) phoneme transcriptions, in original BBC-Oxford LRS2
    # order. Headerless, two columns (sentence, phoneme sequence). See
    # load_original_phoneme_text_pairs() below.
    "original_order":  os.path.join(DATA_DIR, "sentphonemepairs_LRS2_original.csv"),
}


# ── Phoneme sequence cleaning ─────────────────────────────────────────────────
def clean_phoneme_seq(seq: str) -> str:
    """
    Normalise an ARPAbet phoneme sequence.
    Removes <SOS>, <EOS>, strips stress markers, normalises spaces.

    Example:
        '<SOS> TH R UW1 <space> W AH1 T <EOS>'
        -> 'TH R UW <space> W AH T'
    """
    seq = seq.strip()
    seq = re.sub(r"<SOS>|<EOS>", "", seq)       # remove sentence markers
    seq = re.sub(r"[012]", "", seq)              # strip stress digits
    seq = re.sub(r"\s+", " ", seq).strip()
    return seq


def clean_sentence(text: str) -> str:
    """Normalise a sentence to uppercase, stripped."""
    return text.strip().upper()


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_phoneme_text_pairs(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the full phoneme-text dataset.
    Expects columns: Sentence, Phoneme Transcription

    Returns a DataFrame with:
        sentence        – cleaned uppercase text
        phonemes        – cleaned ARPAbet sequence (with <space> word boundaries)
        phonemes_raw    – original phoneme string
    """
    if path is None:
        path = FILES["full"]

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna()

    df["sentence"]     = df["Sentence"].apply(clean_sentence)
    df["phonemes_raw"] = df["Phoneme Transcription"].astype(str)
    df["phonemes"]     = df["Phoneme Transcription"].apply(clean_phoneme_seq)

    return df[["sentence", "phonemes", "phonemes_raw"]].reset_index(drop=True)


def load_original_phoneme_text_pairs(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the full 48,164-sentence LRS2 corpus with its REAL phoneme
    transcriptions, in original BBC-Oxford LRS2 order
    (sentphonemepairs_LRS2_original.csv).

    Unlike load_phoneme_text_pairs() (sentences.csv, a 9-row toy sample)
    and the CMU-dict self-generated phonemes from data/g2p.py, this file
    is headerless and was not produced by this project's G2P module — a
    spot check (21 Jun 2026) found it resolves words our G2P chokes on,
    notably British spellings the standard CMU dict doesn't carry
    (COLOUR, FLAVOUR, ...): self-generated G2P leaves an <UNK> token on
    5.95% of these 48,164 sentences (2,864 rows), while this file has
    zero unresolved-word markers across the whole corpus. Prefer this
    as the phoneme source going forward; data/g2p.py's self-generation
    remains useful for ad-hoc text not in this corpus.

    Returns the same shape as load_phoneme_text_pairs():
        sentence, phonemes (cleaned), phonemes_raw (original string)
    """
    if path is None:
        path = FILES["original_order"]

    df = pd.read_csv(path, header=None, names=["Sentence", "Phoneme Transcription"])
    df = df.dropna()

    df["sentence"]     = df["Sentence"].apply(clean_sentence)
    df["phonemes_raw"] = df["Phoneme Transcription"].astype(str)
    df["phonemes"]     = df["Phoneme Transcription"].apply(clean_phoneme_seq)

    return df[["sentence", "phonemes", "phonemes_raw"]].reset_index(drop=True)


def load_homophone_sentences(path: Optional[str] = None) -> pd.DataFrame:
    """Load sentences that contain at least one homophone-prone word."""
    if path is None:
        path = FILES["with_homophones"]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["sentence"] = df["Sentence"].apply(clean_sentence)
    return df[["sentence"]].reset_index(drop=True)


def load_non_homophone_sentences(path: Optional[str] = None) -> pd.DataFrame:
    """Load sentences that contain no homophone-prone words."""
    if path is None:
        path = FILES["without_homophones"]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["sentence"] = df["Sentence"].apply(clean_sentence)
    return df[["sentence"]].reset_index(drop=True)


def load_stratified_split(phoneme_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a phoneme-text DataFrame, split into:
        homophone_df     – rows whose sentence appears in the homophone set
        non_homophone_df – rows whose sentence does not

    Returns (homophone_df, non_homophone_df)
    """
    homo_sentences  = set(load_homophone_sentences()["sentence"])
    non_homo_sentences = set(load_non_homophone_sentences()["sentence"])

    mask_homo    = phoneme_df["sentence"].isin(homo_sentences)
    mask_non_homo = phoneme_df["sentence"].isin(non_homo_sentences)

    homophone_df     = phoneme_df[mask_homo].reset_index(drop=True)
    non_homophone_df = phoneme_df[mask_non_homo].reset_index(drop=True)

    # Rows in neither set (edge cases) go to non-homophone by default
    neither = phoneme_df[~mask_homo & ~mask_non_homo]
    non_homophone_df = pd.concat([non_homophone_df, neither], ignore_index=True)

    return homophone_df, non_homophone_df


# ── Summary ───────────────────────────────────────────────────────────────────
def dataset_summary(phoneme_df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded dataset."""
    homo_df, non_homo_df = load_stratified_split(phoneme_df)

    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total sentences            : {len(phoneme_df):>8,}")
    print(f"  Homophone sentences        : {len(homo_df):>8,}  ({len(homo_df)/len(phoneme_df)*100:.1f}%)")
    print(f"  Non-homophone sentences    : {len(non_homo_df):>8,}  ({len(non_homo_df)/len(phoneme_df)*100:.1f}%)")
    print(f"  Avg words per sentence     : {phoneme_df['sentence'].apply(lambda x: len(x.split())).mean():>8.1f}")
    print(f"  Avg phonemes per sentence  : {phoneme_df['phonemes'].apply(lambda x: len(x.split())).mean():>8.1f}")
    print("=" * 55)


if __name__ == "__main__":
    print("Loading phoneme-text pairs...")
    df = load_phoneme_text_pairs()
    print(f"\nLoaded {len(df)} pairs.\n")
    print(df.head(3).to_string())
    print()
    dataset_summary(df)
