# Put exact_match.py next to your script, 
# then from exact_match import exact_match and call exact_match(refs, hyps).



import re


def normalise(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(refs, hyps):
    if not refs:
        return 0.0
    correct = 0
    for r, h in zip(refs, hyps):
        if normalise(r) == normalise(h):
            correct += 1
    return correct / len(refs)
