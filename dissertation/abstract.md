Automated lip-reading recovers text from silent video, and its sentence-level accuracy is
constrained chiefly by phoneme-to-text conversion, the stage that maps a recognised phoneme
sequence to words. This dissertation investigates phoneme-to-text conversion on the
Oxford-BBC Lip Reading Sentences 2 (LRS2) corpus. It addresses whether a decoder-only large
language model, adapted by parameter-efficient fine-tuning, outperforms prompting and
published encoder-decoder models; how much of its accuracy persists once training-to-evaluation
sentence overlap is removed; whether training on deliberately corrupted phonemes improves
robustness to the imperfect input of a visual front-end; and whether homophones are the
dominant error type. A Llama-3.2-3B model was fine-tuned with QLoRA and compared against a
zero-shot baseline of the same model and a recurrent network trained from scratch, using
conventional, semantic, and staged error-analysis evaluation. Prompting was ineffective,
yielding near-zero exact match, whereas the fine-tuned model achieved 88.82% exact match on
the held-out test set, exceeding the published encoder-decoder result of 85.11%. A
contamination audit established that approximately one evaluation sentence in eight is
memorised from training; the deduplicated accuracy, which excludes these sentences, is 87.44%.
The clean-trained decoder was highly sensitive to input corruption, whereas noise-augmented
training raised exact match under a 5% corruption rate from between 19% and 46% to between 58%
and 81%, at a cost of approximately three points on clean input. Homophones accounted for
around 5% of errors, corroborating prior work. The study concludes that decoder-only
fine-tuning is effective for this stage, and that rigorous evaluation and robustness are more
consequential than homophone-specific handling.
