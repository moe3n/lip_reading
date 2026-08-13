Automated lip-reading recovers text from silent video, and its accuracy is limited most by
the stage that converts recognised phonemes into words. This dissertation studies that stage
on the Oxford-BBC Lip Reading Sentences 2 corpus. It asks whether a decoder-only
large language model, adapted by parameter-efficient fine-tuning, outperforms prompting and
published encoder-decoder work, how much of its accuracy survives once training-to-evaluation
sentence overlap is removed, whether training on deliberately corrupted phonemes improves its
robustness to the imperfect input a visual front-end produces, and whether homophones are the
dominant error type. A Llama-3.2-3B model was fine-tuned with QLoRA on a single graphics card,
compared against a zero-shot baseline of the same model and a recurrent baseline with no
language knowledge, and evaluated with conventional, semantic, and staged error-analysis
methods. Prompting failed, reaching near-zero exact match, while the fine-tuned model reached
88.82% exact match on the held-out test set, exceeding the published encoder-decoder result of
85.11%. A contamination audit found that roughly one evaluation sentence in eight is memorised
from training, and the honest deduplicated accuracy is 87.44%. The clean-trained decoder proved
fragile under corrupted input, and noise-augmented training raised accuracy under a light
corruption from between 19% and 46% to between 58% and 81%, at a clean-input cost of about three
points. The error analysis found homophones to be a small minority of errors, around 5%, which
corroborates earlier work. The project concludes that decoder-only fine-tuning solves this stage
effectively, and that honest evaluation and robustness matter more than homophone-specific
handling.
