"""Stage 3, Option 1: manual error annotation of the noise model's failures,
using the framework's five-category scheme, ending in a frequency table.

The category for each failing prediction was assigned by reading the pair and
applying the framework's definitions with this precedence when more than one
could apply:

  Contextual  a grammatical error: tense, number agreement, word order, or a
              missing/added function word, with the content words otherwise right
  Homophone   the wrong word has the same or near-identical pronunciation
  Semantic    a fluent, grammatical sentence whose meaning has changed
  Other       a non-word or misspelling, a mangled or wrong proper noun, a
              number written the other way (six/6), a split or merged word, or a
              dropped word
  Lexical     a wrong but real word not covered above (the default for real-word
              substitutions)

Labels are listed in the same order as failing_rows.csv. Editing a label here
and re-running updates the table, so a human reviewer can adjust borderline
calls directly.
"""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path("p2t_lora_checkpoints_noise/analysis")
FAIL = ROOT / "failing_rows.csv"

# One label per failing row, in file order.
LABELS = [
    "Homophone",    # FORTNIGHT -> FOUR NIGHTS
    "Lexical",      # ROPES -> ROLES
    "Other",        # SAXIFRAGE -> SAXON FRAGMENT (rare word split)
    "Lexical",      # SQUIRREL -> SQUAMOUS
    "Lexical",      # KEY -> LINK
    "Lexical",      # FLAWED -> FLUID
    "Other",        # BIZARRE THEORY -> BIZARRE (truncation)
    "Lexical",      # PREVAILING -> PERSISTING
    "Contextual",   # EXTRAORDINARY -> EXTRAORDINARILY (word form)
    "Other",        # WALDORF ASTORIA -> WALLDORF STORY IS (proper noun)
    "Other",        # 10 -> TEN (number)
    "Other",        # SIMILAR -> SIMILER (non-word)
    "Other",        # UPSTAIRS -> UP STAIRS (split)
    "Other",        # IN SPADES -> ITS INSPACES (non-word)
    "Other",        # COSMETICALLY -> CAUSE METICULOUSLY (split/mangle)
    "Lexical",      # APPLIANCES -> APPLICANTS
    "Other",        # TECHNO -> TECH NO (split)
    "Other",        # CAGED EGGS -> CAKE DEGAS (mangle)
    "Lexical",      # ACKNOWLEDGED -> ANNULLED
    "Other",        # GARRISON -> GARISON (non-word)
    "Other",        # GARRISON -> GARISON (non-word)
    "Other",        # TEARING -> TERRING (non-word)
    "Other",        # PORTCULLIS -> PERKETELLIS (non-word)
    "Homophone",    # FOREPLAY -> FOUR PLAY
    "Lexical",      # WITHER -> WITHSTAND
    "Homophone",    # DESSERT -> DESERT
    "Other",        # SIX -> 6 (number)
    "Other",        # MELTING SEA ICE -> MELTING ICE (deletion)
    "Other",        # CROSSOVER -> CROSS OVER (split)
    "Other",        # ISLAM ALLOWS POLYGAMY -> EL SALVADOR LOWS POLICY (mangle)
    "Lexical",      # CURRENTLY -> CONTINUALLY
    "Contextual",   # WHERE I'M GOING -> WHERE AM I GOING (word order)
    "Other",        # TEN -> 10 (number)
    "Homophone",    # ANNA -> ANNE
    "Contextual",   # ROCKET -> ROCKETS (agreement)
    "Lexical",      # UNDER -> AND
    "Other",        # ASHCROFT -> ASHKROFT (proper noun)
    "Other",        # BEFOREHAND -> BEFORE HAND (split)
    "Lexical",      # TRANSITION -> TRANSLATION
    "Other",        # TEN CUBIC METRES -> TAKE A METRE RULER (mangle)
    "Lexical",      # ETIQUETTE -> SECRET
    "Homophone",    # KHAN -> CON
    "Other",        # TRIBAL -> TRIBLE (non-word)
    "Contextual",   # UPSET -> UPSETS (agreement)
    "Other",        # PUERTO RICAN -> PERHAPS A RECON (mangle)
    "Homophone",    # LEONARD COHEN -> LEARNED COUSIN
    "Other",        # BARON ASH -> BARRON AS (non-word/truncation)
    "Contextual",   # SURPRISES HAPPEN -> SURPRISE IS HAPPENING
    "Homophone",    # KNOT -> NOT
    "Lexical",      # TOWER -> TOUR
    "Lexical",      # OFFENCES -> OFFENSIVES
    "Other",        # BRONTE SISTERS -> BRONTES (truncation)
    "Lexical",      # SYLLABLES -> SILHOUETTES
    "Lexical",      # MARITIME -> MARRIAGE
    "Other",        # OFFENDING -> AFFENDING (non-word)
    "Other",        # 9 11 -> 911 (number)
    "Other",        # MILLENNIUM -> MALAYA (OOV/proper)
    "Lexical",      # FACIAL -> FASHIONABLE
    "Contextual",   # FACE -> FACED (tense)
    "Homophone",    # GLOUCESTER -> GLOSTER
    "Lexical",      # DELECTABLE -> DIRECTABLE
    "Other",        # TOO -> 2 (number)
    "Lexical",      # RESCUING -> REQUESTING
    "Contextual",   # THROUGH ANTIBODY -> THROUGH AN ANTIBODY (article)
    "Other",        # COWELL -> KELL (proper noun)
    "Lexical",      # HASTEN -> HAD
    "Lexical",      # CHAMPAGNE -> CHAMPION
    "Contextual",   # DO BALLROOM -> DO A BALLROOM (article)
    "Other",        # JENNY ECLAIR -> GENIE CLARKE (proper noun)
    "Other",        # ANTIQUARIANS -> ANTIQUERS (non-word)
    "Other",        # PURIFICATION -> PURE AFFECTION (split/mangle)
    "Other",        # NICOLA -> NICHOLAS (proper noun)
    "Other",        # ROCHELLE -> ROCKET (proper noun)
    "Lexical",      # BILLY -> REALLY
    "Other",        # PRO CHALLENGE -> PROCHALLENGE (merge)
    "Contextual",   # SAUSAGES IN -> SAUSAGE IS IN
    "Contextual",   # UNFOLD -> UNFOLDED (tense)
    "Other",        # URANIUM -> YOUR ANTIQUES (mangle)
    "Other",        # CALM/DISCONCERTINGLY -> COME/DISCONCERTEDLY (non-word)
    "Homophone",    # TO -> TWO
    "Other",        # SHEER SIZE -> SIZE (deletion)
    "Lexical",      # BANANAS -> BUNNIES
    "Homophone",    # NEUTRONS -> NEWTONS
    "Lexical",      # UNIVERSE -> UNIVERSITY
    "Other",        # 3 -> THREE (number)
    "Lexical",      # IMMENSITY -> IMMINENCE
    "Lexical",      # RARITY -> REPUTATION
    "Lexical",      # SOCKS -> SAX
    "Other",        # SCARECROW -> SCARCROW (non-word)
    "Semantic",     # RECTIFY WRONGS -> CONTACT US
    "Lexical",      # CALM -> COME
    "Lexical",      # DAY -> DAUGHTER
    "Other",        # ADVERTISEMENT -> ADVERTISMENT (non-word)
    "Lexical",      # BAT -> BATH
    "Lexical",      # FATEFUL -> FATAL
    "Lexical",      # STEALTH -> STRENGTH
    "Lexical",      # CHOOSE -> KNOW
    "Other",        # ELIZABETH -> ISABELLA (proper noun)
    "Other",        # ELIZABETH -> ISABELLA (proper noun)
    "Lexical",      # FIFA -> ME
    "Semantic",     # LET LOOSE -> LET ME TELL YOU
    "Other",        # LLOYD -> LOYD (proper noun)
    "Homophone",    # BY -> BUY
    "Other",        # DATABASE -> DATA BASE (split)
    "Other",        # DENISE -> DENNIS (proper noun)
    "Lexical",      # DRYING -> DRIVING
    "Other",        # MARXIST GROUP -> MARKS AND SERVICES GROUP (mangle)
]


def main():
    rows = list(csv.DictReader(open(FAIL, encoding="utf-8")))
    assert len(rows) == len(LABELS), f"{len(rows)} rows but {len(LABELS)} labels"

    with open(ROOT / "stage3_manual_annotation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "prediction", "category"])
        for r, lab in zip(rows, LABELS):
            w.writerow([r["target"], r["prediction"], lab])

    counts = Counter(LABELS)
    n = len(LABELS)
    order = ["Other", "Lexical", "Homophone", "Contextual", "Semantic"]
    print(f"Stage 3 Option 1 — manual annotation, noise model ({n} failing predictions)\n")
    print(f"{'Error Type':<12}{'Count':>6}{'Percent':>9}")
    print("-" * 27)
    for cat in order:
        c = counts.get(cat, 0)
        print(f"{cat:<12}{c:>6}{c / n * 100:>8.1f}%")
    print("-" * 27)
    print(f"{'Total':<12}{n:>6}{100.0:>8.1f}%")


if __name__ == "__main__":
    main()
