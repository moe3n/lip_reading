"""Stage 3 Option 1: manual error annotation of the CLEAN model's failures,
same five-category scheme and precedence as the noise annotation, so the two
frequency tables compare directly.

Rows are in the order of tables/semantic_similarity.csv (the clean failing set).
"""

import csv
from collections import Counter
from pathlib import Path

ROOT = Path("p2t_lora_checkpoints_dedup/analysis")
FAIL = ROOT / "tables" / "semantic_similarity.csv"

LABELS = [
    "Other",       # 1  UNDERSTAND LEASEHOLD -> UNDERSTANDLY SELL (non-word)
    "Other",       # 2  FORTNIGHT -> NIGHT (truncation)
    "Other",       # 3  ROPES -> ROPE S (split)
    "Other",       # 4  SAXIFRAGE -> SAXOPHONE FRAGMENT (OOV split)
    "Other",       # 5  SQUIRREL -> SQUWAREL (non-word)
    "Contextual",  # 6  SHIP -> SHIPPING (form)
    "Other",       # 7  BIZARRE THEORY -> BIZARRE (truncation)
    "Other",       # 8  WALDORF ASTORIA -> WALLDORF STORY A (proper noun)
    "Other",       # 9  10 -> TEN (number)
    "Lexical",     # 10 PROSPECT -> PROJECT
    "Lexical",     # 11 CALM -> COME
    "Other",       # 12 ACKNOWLEDGED -> ANNOLLED (non-word)
    "Other",       # 13 SYRIA -> SURREY (proper noun)
    "Other",       # 14 GARRISON -> GARISON (non-word)
    "Lexical",     # 15 TEARING -> TERRIFYING
    "Other",       # 16 PORTCULLIS -> PERKELLIS (non-word)
    "Homophone",   # 17 FOREPLAY -> FOUR PLAY
    "Other",       # 18 INFLAMMATORY -> IN FLAMATORY (split/non-word)
    "Other",       # 19 INFLAMMATORY -> IN FLAMMATORY (split)
    "Other",       # 20 SIX -> 6 (number)
    "Lexical",     # 21 PINT -> PAINT
    "Other",       # 22 SOUTH WELL -> SOUTHWELL (merge)
    "Other",       # 23 MELTING SEA ICE -> MELTING ICE (deletion)
    "Other",       # 24 CROSSOVER -> CROSS OVER (split)
    "Other",       # 25 POLYGAMY -> POLIGAMY (non-word)
    "Other",       # 26 TEN -> 10 (number)
    "Homophone",   # 27 ANNA -> ANNE
    "Other",       # 28 BEFOREHAND -> BEFORE HAND (split)
    "Other",       # 29 TEN -> 10 (number)
    "Lexical",     # 30 ETIQUETTE -> ETHIC
    "Other",       # 31 KHAN -> CONNIE (proper noun)
    "Contextual",  # 32 OLYMPIC -> OLYMPICS (number)
    "Contextual",  # 33 UPSET -> UPSETS (agreement)
    "Other",       # 34 PUERTO RICAN STYLE -> PWARE TO RECONSTRUCT (mangle)
    "Other",       # 35 WEDDINGS -> WEEDINGS (non-word)
    "Other",       # 36 BARON ASH -> BARRENASH (non-word)
    "Other",       # 37 BRONTE -> BRONTE-diacritic (spelling)
    "Other",       # 38 MARITIME -> MARATIME (non-word)
    "Lexical",     # 39 OFFENDING -> OFFERING
    "Other",       # 40 9 11 -> 9/11 (number)
    "Other",       # 41 MILLENNIUM -> MOLYNEUX (OOV/proper)
    "Lexical",     # 42 FACIAL -> FAMILIAL
    "Homophone",   # 43 GLOUCESTER -> GLOSTER
    "Lexical",     # 44 DELECTABLE -> DELICATE
    "Homophone",   # 45 TOO -> TO
    "Lexical",     # 46 HASTEN -> HESITANT
    "Other",       # 47 JENNY ECLAIR -> GENIE CLARKE (proper noun)
    "Other",       # 48 NICOLA -> NICOLE (proper noun)
    "Other",       # 49 ROCHELLE/MARVIN -> ROSEHELLE/MERVYN (proper noun)
    "Other",       # 50 TOP LINE -> TOPLINE (merge)
    "Other",       # 51 WAXED -> WACKED (non-word)
    "Other",       # 52 PRO CHALLENGE -> PROCHALLENGE (merge)
    "Homophone",   # 53 SOME BURN -> SUMMER (sound-alike merge)
    "Other",       # 54 VICTORIAN -> INVICTORIAN (merge/non-word)
    "Other",       # 55 URANIUM -> URIANIA (non-word)
    "Lexical",     # 56 CALM -> COME
    "Other",       # 57 NEUTRONS -> NUTRONS (non-word)
    "Other",       # 58 3 -> THREE (number)
    "Lexical",     # 59 IMMENSITY -> IMMANENCE
    "Lexical",     # 60 RARITY -> RETIREMENT
    "Lexical",     # 61 SOCKS -> SAX
    "Other",       # 62 SCARECROW -> SCARCROW (non-word)
    "Other",       # 63 RECTIFY WRONGS -> WRECK A FIRE ONGS (mangle)
    "Contextual",  # 64 WEIRDER -> WEIRD (form)
    "Other",       # 65 ADVERTISEMENT -> ADVERTISMENT (non-word)
    "Lexical",     # 66 STEALTH -> STELLA
    "Other",       # 67 QUEEN -> QWEEN (non-word)
    "Other",       # 68 FIFA -> NHS (proper noun)
    "Other",       # 69 LLOYD -> LOYD (non-word)
    "Lexical",     # 70 FLOURISH -> FLURRY
    "Homophone",   # 71 BY -> BUY
    "Other",       # 72 DATABASE -> DATA BASE (split)
    "Other",       # 73 DEBENHAMS -> DEBINOMES (non-word)
    "Contextual",  # 74 I HONESTLY -> HONESTLY (subject drop)
    "Other",       # 75 SPREAD UPON -> SPREAD UP ON (split)
    "Other",       # 76 MARXIST GROUP -> MARKS ASSOCIATION (mangle)
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
    print(f"Stage 3 Option 1 - manual annotation, CLEAN model ({n} failing predictions)\n")
    print(f"{'Error Type':<12}{'Count':>6}{'Percent':>9}")
    print("-" * 27)
    for cat in ["Other", "Lexical", "Homophone", "Contextual", "Semantic"]:
        c = counts.get(cat, 0)
        print(f"{cat:<12}{c:>6}{c / n * 100:>8.1f}%")
    print("-" * 27)
    print(f"{'Total':<12}{n:>6}{100.0:>8.1f}%")


if __name__ == "__main__":
    main()
