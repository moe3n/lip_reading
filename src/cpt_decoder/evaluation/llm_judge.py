"""
CPT Decoder — Stage 3 Option 5: LLM-Based Error Classification
====================================================================
Implements "Option 5: LLM-Based Error Classification" from the "P2T
Three-Stage Error Analysis Framework" (P2T Error Pattern Analysis V1.pdf).
The framework's own worked example (verbatim, p.5-6 of the PDF):

    Prompt:
        Ground Truth: I see the problem.
        Prediction: I sea the problem.

        Classify the error as
        1. Phonological
        2. Lexical
        3. Contextual
        4. Semantic

        Explain why.

    Output:
        Category: Lexical
        Subcategory: Homophone
        Explanation: The predicted word is a homophone of the correct
        word but changes the lexical choice.

This is the module check_grammar() in contextual_analysis.py escalates
to when it returns resolved=False — either because the substitution
involves an open-class homophone (see/sea: not a closed-class word at
all) or a closed-class word whose dependency role doesn't reliably
distinguish correct from incorrect usage (there/too/two: see that
module's docstring). An LLM judge can use full sentence context (and,
in principle, surrounding-sentence context) to make the lexical vs.
contextual vs. semantic call that pure syntax can't.

ONE ADDITION BEYOND THE PDF'S LITERAL PROMPT
--------------------------------------------------------------------
The PDF shows the desired Category/Subcategory/Explanation output as a
worked example, not as an explicit instruction inside the prompt. A
0.5B-1B instruct model run zero-shot won't reliably reproduce that
exact structure without being told to. _build_prompt() below adds one
line -- "Respond in exactly this format:" followed by the three labelled
fields -- so _parse_response() can extract structured fields
automatically across many sentence pairs. The four-option classification
list and "Explain why." instruction are verbatim from the PDF; only the
output-format instruction is this module's own addition, and it's kept
as a clearly separate, minimal scaffold rather than rewording anything
the PDF specifies.

SCOPE LIMITATION: SENTENCE-LEVEL CONTEXT ONLY, NOT PARAGRAPH-LEVEL
--------------------------------------------------------------------
Earlier in this project's methodology discussion, Option 5 was
recommended specifically because an LLM judge can reason at the
paragraph level, not just the sentence level (unlike Option 3's
grammar check, which is inherently sentence-bound). That reach requires
actually passing surrounding-sentence context into the prompt. This
project's real dataset (sentphonemepairs_LRS2_original.csv) has only
two columns -- sentence text and phoneme transcription -- with no
video-id, clip-sequence, speaker-id, or timestamp column to verify
which sentences are contiguous/from the same discourse. Treating
adjacent CSV rows as a paragraph would be an unverified assumption, not
a finding, so classify_error() below deliberately takes only a single
(ground_truth, prediction) pair: it gets Option 5's better single-
sentence reasoning (full clause/sentence structure, not just local
dependency roles), but NOT genuine cross-sentence discourse reasoning.
If a verified paragraph/discourse grouping becomes available upstream
of this CSV, classify_error() would need a third "Context:" field
added to the prompt -- the signature is intentionally narrow so that
doesn't get silently faked.

EMPIRICAL FINDING: THE CPU DRY-RUN JUDGE MODEL IS NOT RELIABLE (25 Jun 2026)
--------------------------------------------------------------------
Ran this module's own smoke test against the cached MODEL_NAME_DRYRUN
(Qwen2.5-0.5B-Instruct). The prompt/parsing pipeline itself worked --
every call returned a cleanly-parsed Category/Subcategory/Explanation,
no "Unparseable" results. But the *classifications were wrong*,
including on the PDF's own two canonical worked examples:
  - "I see the problem" -> "I sea the problem" should be Lexical /
    Homophone (the PDF's own stated answer). The 0.5B model said
    Semantic, and its explanation hallucinated a different sentence
    ("I see the issue") that was never in the prompt.
  - "Yesterday I went home" -> "Yesterday I will go home" should be
    Contextual / Tense inconsistency (the PDF's own Option 3 example).
    The 0.5B model again said Semantic.
  - Both project-data examples (CHIP->SHIP near-homophone,
    CHIP->BANANA unrelated substitution) were also called Semantic.
All four cases got the same label. This looks like the 0.5B model
defaulting to "Semantic" as a safe-seeming guess rather than actually
discriminating between the four categories -- consistent with this
project's standing finding that small CPU stand-in models are useful
for exercising a pipeline's *shape*, not for trusting their outputs
(same caveat already on record for the dry run's decoder itself).
**Don't trust Option-5 category labels produced by the CPU dry-run
judge model.** This module's plumbing (prompt, parsing, generation) is
validated; the 0.5B model's judgement is not. Re-validate this
specifically once a judge model can actually run on the uni PC GPU.

A SECOND ISSUE THIS SURFACED, RELEVANT TO THE REAL RUN SPECIFICALLY:
model.py's MODEL_NAME_TARGET ("meta-llama/Llama-3.2-3B") is the BASE
model, not an instruct/chat checkpoint -- correct for the CPT decoder's
own phoneme-to-text generation task, which doesn't need chat-instruct
behaviour. But it means disable_adapter() reuse of the real decoder for
Option 5 judging (see next section) would run the zero-shot
"Respond in exactly this format" instruction through a non-instruct
base model, which is a much weaker setup for instruction-following
than a model actually tuned for it, on top of whatever capability
issues exist regardless of tuning. Recommendation for task #49: on the
uni PC, point classify_error() at a small but genuinely instruct-tuned
model dedicated to judging (e.g. Llama-3.2-3B-Instruct, or even the
existing MODEL_NAME_DRYRUN Qwen2.5-0.5B-Instruct as a cheap second
pass) rather than disable_adapter()-ing the production base-Llama
decoder. disable_adapter() is still the right call when the underlying
model genuinely is instruct-tuned (true for MODEL_NAME_DRYRUN) -- it
just doesn't rescue a base model that was never instruct-tuned to begin
with.

WHY disable_adapter() WHEN A PEFT-WRAPPED MODEL IS PASSED IN
--------------------------------------------------------------------
Task #49 wires this module into dryrun.py, which already has a
tokenizer + LoRA-wrapped model loaded for the thing actually being
evaluated. Reusing that loaded model avoids a second multi-hundred-MB
download/load in a memory-constrained sandbox (see model.py's CPU_DTYPE
comment re: 3.8GiB RAM). But the LoRA adapter is either randomly
initialised (CPU dry run, no training yet) or trained for the decoder's
own task (real run) -- neither is what we want answering "what kind of
linguistic error is this", which is a generic instruction-following
task closer to the base instruct model's pretraining. peft's
PeftModel.disable_adapter() context manager runs the forward/generate
pass through the un-adapted base weights, so classify_error() uses it
whenever the passed-in model exposes it, and falls back to a plain
.generate() call otherwise (e.g. a bare AutoModelForCausalLM with no
adapter, as in this module's own standalone smoke test).
"""

import re
import contextlib
from typing import Dict, Optional

ALLOWED_CATEGORIES = {"Phonological", "Lexical", "Contextual", "Semantic"}


def _build_prompt(ground_truth: str, prediction: str) -> str:
    """
    Builds the user-turn text. Verbatim from the PDF's Option 5 prompt
    (Ground Truth / Prediction / four-option list / "Explain why."),
    plus one added output-format instruction -- see module docstring.
    """
    return (
        f"Ground Truth: {ground_truth}\n"
        f"Prediction: {prediction}\n\n"
        "Classify the error as\n"
        "1. Phonological\n"
        "2. Lexical\n"
        "3. Contextual\n"
        "4. Semantic\n\n"
        "Explain why.\n\n"
        "Respond in exactly this format:\n"
        "Category: <Phonological, Lexical, Contextual, or Semantic>\n"
        "Subcategory: <a short label, e.g. Homophone, Near-homophone, "
        "Tense inconsistency, Word order, Word choice, or None>\n"
        "Explanation: <one sentence>"
    )


def _parse_response(text: str) -> Dict[str, Optional[str]]:
    """
    Extract Category/Subcategory/Explanation from the model's free-text
    response. Tolerant of minor formatting drift (extra whitespace,
    missing trailing field) since small instruct models don't always
    follow a requested format exactly -- but does NOT guess a category
    that wasn't actually said. An unparseable or out-of-vocabulary
    category is reported as such (category="Unparseable") rather than
    silently defaulted, so it surfaces for manual review instead of
    polluting the aggregate counts.
    """
    cat_match = re.search(r"Category:\s*([A-Za-z]+)", text)
    sub_match = re.search(r"Subcategory:\s*(.+)", text)
    exp_match = re.search(r"Explanation:\s*(.+)", text, re.DOTALL)

    category = cat_match.group(1).strip() if cat_match else None
    if category not in ALLOWED_CATEGORIES:
        category = "Unparseable"

    subcategory = sub_match.group(1).strip().splitlines()[0].strip() if sub_match else None
    explanation = exp_match.group(1).strip().splitlines()[0].strip() if exp_match else None

    return {"category": category, "subcategory": subcategory, "explanation": explanation}


def classify_error(tokenizer,
                    model,
                    ground_truth: str,
                    prediction: str,
                    max_new_tokens: int = 100) -> Dict:
    """
    Run the Option 5 LLM-judge prompt for one (ground_truth, prediction)
    sentence pair and return a parsed classification.

    Args:
        tokenizer    : a loaded HF tokenizer (e.g. from model.load_tokenizer()).
        model        : a loaded HF causal LM, optionally PEFT-wrapped. If it
                        exposes disable_adapter(), this function uses it (see
                        docstring) so judging isn't done through an untrained
                        or task-specific LoRA adapter.
        ground_truth : the reference sentence.
        prediction   : the decoded/hypothesis sentence.
        max_new_tokens : generation budget; ~100 covers the three short
                        labelled fields with room for a one-sentence explanation.

    Returns:
        {"category": str, "subcategory": str or None, "explanation": str or None,
         "raw_output": str}
        category is one of ALLOWED_CATEGORIES or "Unparseable". raw_output is
        always included so an unparseable or surprising response can be
        inspected manually rather than discarded.
    """
    import torch  # local import: keep this module importable without torch
                   # for cases where only _build_prompt/_parse_response are
                   # being unit-tested.

    prompt_text = _build_prompt(ground_truth, prediction)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    adapter_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else contextlib.nullcontext()

    with torch.no_grad(), adapter_ctx:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                # greedy -- deterministic, reproducible for a dissertation
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    parsed = _parse_response(raw_output)
    parsed["raw_output"] = raw_output
    return parsed


if __name__ == "__main__":
    # ── Smoke test against the PDF's own worked example, plus two
    #    project-relevant cases (near-homophone, unrelated substitution) ──
    import sys
    import os

    # This file lives in src/cpt_decoder/evaluation/ -- "from cpt_decoder.X
    # import ..." needs src/ on sys.path, which is 3 directories up from
    # here (evaluation -> cpt_decoder -> src), not 2. See error_analysis.py
    # for the same off-by-one and why it matters (it additionally needs a
    # second sys.path entry for a "data.X" bare import this file doesn't use).
    _src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _src_dir)
    from cpt_decoder.model import load_tokenizer, MODEL_NAME_DRYRUN  # noqa: E402
    from transformers import AutoModelForCausalLM  # noqa: E402

    print(f"\nLoading judge model ({MODEL_NAME_DRYRUN}) — base instruct model, "
          f"no LoRA wrapping needed for this standalone smoke test...")
    tok = load_tokenizer(MODEL_NAME_DRYRUN)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_NAME_DRYRUN)
    mdl.resize_token_embeddings(len(tok))
    mdl.eval()

    cases = [
        ("I see the problem", "I sea the problem"),                       # PDF's own example -> expect Lexical/Homophone
        ("Yesterday I went home", "Yesterday I will go home"),            # PDF's Option 3 example -> expect Contextual/Tense
        ("WHAT REALLY MAKES A CHIP IS THE CRUNCH",
         "WHAT REALLY MAKES A SHIP IS THE CRUNCH"),                       # near-homophone, project data
        ("THE TRADITIONAL CHIP PAN OFTEN STAYS ON THE SHELF",
         "THE TRADITIONAL BANANA PAN OFTEN STAYS ON THE SHELF"),          # unrelated substitution, project data
    ]

    print("\n" + "=" * 70)
    print("  Smoke test — llm_judge.py (Stage 3 Option 5)")
    print("=" * 70)
    for gt, pred in cases:
        result = classify_error(tok, mdl, gt, pred)
        print(f"\n  Ground Truth: {gt}")
        print(f"  Prediction:   {pred}")
        print(f"  -> Category: {result['category']}   Subcategory: {result['subcategory']}")
        print(f"     Explanation: {result['explanation']}")
        print(f"     [raw_output: {result['raw_output']!r}]")
    print("\n" + "=" * 70 + "\n")
