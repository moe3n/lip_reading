"""
CPT Decoder — Stage 3 Option 3: Grammar-Based Contextual Analysis
====================================================================
Implements "Option 3: Grammar-Based Context Analysis" from the "P2T
Three-Stage Error Analysis Framework" (P2T Error Pattern Analysis V1.pdf).
The framework names spaCy / Stanza / LanguageTool as candidate tools and
gives one worked example: "Yesterday I went home" -> "Yesterday I will go
home", labelled Contextual / Tense inconsistency.

WHAT THIS MODULE ACTUALLY CATCHES, AND WHAT IT DOESN'T (validated, not
assumed — 25 Jun 2026)
--------------------------------------------------------------------
Before writing the rules below, I ran spaCy's en_core_web_sm against both
of the PDF's own running examples to see whether a context-aware grammar
check could resolve them automatically:

  "I see the problem"  vs  "I sea the problem"
      -> spaCy's statistical tagger tags "sea" as VERB/ROOT in this
         context, identical to "see". The parse is internally consistent
         either way. POS/dependency parsing CANNOT catch this case — it's
         a content-word (open-class) homophone, and the tagger leans on
         surrounding context to guess a plausible tag for any token,
         correct word or not. This needs Stage 3 Option 5 (LLM-based
         classification, see llm_judge.py), not grammar.

  "their is a cat"  (should be "there is a cat")
      -> spaCy tags "their" correctly as PRP$ (possessive pronoun) but
         assigns it dep_="nsubj" — a possessive determiner standing in
         as a clause subject. That's a hard violation: "their" is
         lexically restricted to the "poss" dependency role (it always
         modifies a following head noun) no matter what surrounds it.
         This is catchable, reliably, because the violation is a property
         of the closed-class word itself, not of statistical context.

The dividing line this revealed: closed-class function-word homophones
(determiners/pronouns/contractions — there/their/they're, your/you're,
its/it's, whose/who's, to/too/two) have a fixed, narrow set of legal
syntactic roles that holds regardless of context, so a dependency-role
mismatch is strong, low-risk evidence of a genuine contextual error.
Open-class content-word homophones (see/sea, meet/meat, sight/site, ...)
do not have this property — the tagger will happily rationalise either
spelling. So this module only ever fires on the closed-class list below;
everything else returns resolved=False and should be escalated to
Stage 3 Option 5.

A SECOND, NARROWER FINDING: not every closed-class homophone set actually
benefits from this method (validated 25 Jun 2026, after the first smoke
test surfaced 3 unexpected non-flags)
--------------------------------------------------------------------
THEIR/YOUR/ITS/MY/OUR/WHOSE work because forcing them into the wrong
slot (e.g. "their" as a clause subject) produces a dependency label
genuinely outside their one legitimate role ("poss"). Re-tested this
against "your is broken", "its is broken", "my is here", "our is
bigger", "whose is this here" — all correctly parse to nsubj/nsubjpass,
none of which is "poss", so all correctly flag. Confirmed not a fluke
specific to "their is a cat".

THERE/TOO/TWO do NOT share this property, despite also being
closed-class. Tested "I left there book", "I went too the shop", "I
have too books", "I went two the shop", "too is broken": in every case
spaCy's parser falls back to the SAME dependency label it would assign
the word when used correctly (THERE -> advmod attaching to the nearest
verb, TOO -> advmod attaching to the nearest verb, TWO -> nummod
attaching to the nearest following noun) *regardless of whether that
usage is actually grammatical*. E.g. "too is broken" -- nonsensical,
"too" can't be a sentence subject -- still parses as advmod(broken),
never forced into a genuinely mismatched label the way "their is a
cat" forces THEIR into nsubj. The parser effectively has nowhere else
to put these three words, so a dep-role check can't distinguish correct
from incorrect usage for them. THERE/TOO/TWO are therefore EXCLUDED
from _VALID_DEPS below: keeping them would silently overclaim a
detection capability that doesn't exist for this word class. Real
their/your/its/my/our/whose-type errors: catchable here. Real
there/too/two-type errors: must escalate to Stage 3 Option 5.

WHY OPERATE ON THE ORIGINAL SENTENCE, NOT THE JIWER-NORMALISED ONE
--------------------------------------------------------------------
error_analysis.py's normalise() strips punctuation (re.sub(r"[^\w\s]",
"", text)) before word-alignment, which deletes apostrophes. That's fine
for alignment, but it means "IT'S" and "ITS" collapse to the identical
string "ITS" by the time classify_substitution() sees them — exactly the
distinction this module exists to resolve. So check_grammar() below is
designed to be called with the ORIGINAL (un-normalised) hypothesis
sentence, recovering the apostrophe'd word at the same whitespace-split
index, not the alignment-time normalised one.
"""

import spacy
from typing import Dict, Optional

_NLP = None  # lazy-loaded singleton — importing this module shouldn't
             # force-load the spaCy model until it's actually needed.


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# ── Closed-class confusable words and their only legal dependency roles ──────
# Validated empirically against en_core_web_sm (see module docstring).
# HIS/HER are deliberately excluded: both have a legitimate standalone
# (non-"poss") use ("I saw her", "this is his"), so a dep-role mismatch
# wouldn't reliably indicate an error for them.
# THERE/TOO/TWO are deliberately excluded too, for a different reason:
# spaCy's parser falls back to the same dep label (advmod / nummod)
# whether or not the word is used correctly, so a mismatch check has
# nothing to detect for this trio — see module docstring's second
# finding. Escalate substitutions involving them to Stage 3 Option 5.
_VALID_DEPS = {
    "THEIR": {"poss"},
    "YOUR":  {"poss"},
    "ITS":   {"poss"},
    "MY":    {"poss"},
    "OUR":   {"poss"},
    "WHOSE": {"poss"},
}


def _word_char_span(sentence: str, word_idx: int) -> Optional[tuple]:
    """
    Return the (start_char, end_char) span of the word_idx-th
    whitespace-split word in `sentence`, searched left-to-right so
    repeated words resolve to the correct occurrence.
    """
    words = sentence.split()
    if word_idx < 0 or word_idx >= len(words):
        return None
    pos = 0
    for i, w in enumerate(words):
        start = sentence.index(w, pos)
        end = start + len(w)
        if i == word_idx:
            return start, end
        pos = end
    return None


def check_grammar(original_hyp_sentence: str, hyp_word: str, hyp_word_idx: int) -> Dict:
    """
    Check whether `hyp_word` at whitespace-index `hyp_word_idx` in
    `original_hyp_sentence` is used in a syntactic role its own lexical
    class permits.

    Args:
        original_hyp_sentence : the UN-normalised hypothesis sentence
                                  (apostrophes intact) — see module
                                  docstring for why this matters.
        hyp_word               : the substituted word, in its original
                                  (apostrophe'd) spelling.
        hyp_word_idx            : its whitespace-split index in
                                  original_hyp_sentence.

    Returns:
        {"resolved": bool, "rule": str or None, "explanation": str or None}
        resolved=True means this module found a closed-class dependency
        violation — i.e. the substitution is explainable as a Contextual/
        grammatical error without needing Stage 3 Option 5.
        resolved=False covers two different cases: the word's role IS
        valid here (grammar can't explain an error that isn't there from
        a syntax point of view), or the word isn't in the closed-class
        table at all (open-class content word — grammar alone isn't a
        reliable signal, see docstring). Either way, escalate to Option 5.
    """
    key = hyp_word.upper().strip(".,!?;:")
    if key not in _VALID_DEPS:
        return {"resolved": False, "rule": None, "explanation": None}

    span = _word_char_span(original_hyp_sentence, hyp_word_idx)
    if span is None:
        return {"resolved": False, "rule": None, "explanation": None}
    start, end = span

    doc = _get_nlp()(original_hyp_sentence)
    covering = [t for t in doc if t.idx < end and (t.idx + len(t.text)) > start]
    if not covering:
        return {"resolved": False, "rule": None, "explanation": None}

    valid_deps = _VALID_DEPS[key]
    observed_deps = {t.dep_ for t in covering}

    if observed_deps & valid_deps:
        # The word's role here is one its own class permits — grammar
        # finds nothing wrong, so it can't be the thing explaining this
        # substitution. Hand off to Option 5.
        return {"resolved": False, "rule": "grammar_consistent", "explanation": None}

    observed_str = ", ".join(sorted(observed_deps)) or "(none)"
    valid_str = ", ".join(sorted(valid_deps))
    return {
        "resolved": True,
        "rule": "closed_class_dependency_mismatch",
        "explanation": (
            f"'{hyp_word}' is lexically restricted to the dependency "
            f"role(s) [{valid_str}] (it can only modify or stand in for "
            f"a following head noun / verb in that fixed way), but the "
            f"parser places it here as [{observed_str}] — a role only "
            f"a homophone counterpart could fill. Category: Contextual."
        ),
    }


if __name__ == "__main__":
    # ── Smoke test — mirrors the PDF's own worked examples plus the
    #    closed-class cases this module is actually built for ───────────
    cases = [
        ("there is a cat",            "their", 0, True),   # poss forced into nsubj -> flag
        ("I left their book",         "their", 2, False),  # legit poss use
        ("your is broken",            "your",  0, True),   # poss forced into nsubjpass -> flag
        ("I gave your the keys",      "your",  2, False),  # legit poss use
        ("I left there book",         "there", 2, False),  # NOT catchable: "there" always
                                                             # parses advmod regardless of fit
        ("I went too the shop",       "too",   2, False),  # NOT catchable: same reason
        ("I have too books",          "too",   2, False),  # NOT catchable: same reason
        ("I sea the problem",         "sea",   1, False),  # NOT catchable: open-class word
    ]
    print("\n" + "=" * 66)
    print("  Smoke test — contextual_analysis.py (Stage 3 Option 3)")
    print("=" * 66)
    for sentence, word, idx, expect_flag in cases:
        result = check_grammar(sentence, word, idx)
        status = "FLAGGED " if result["resolved"] else "no flag "
        ok = "OK " if result["resolved"] == expect_flag else "**MISMATCH**"
        print(f"\n  [{status}] {ok}  \"{sentence}\"  (word='{word}' idx={idx})")
        if result["explanation"]:
            print(f"            {result['explanation']}")
    print("\n" + "=" * 66 + "\n")
