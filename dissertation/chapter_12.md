# Chapter 12. Conclusion

This chapter evaluates the project against the aim and objectives it set out to achieve, states its contributions, and reflects on what was learned and what remains to be done.

## 12.1 Achievement of the aim and objectives

The aim was to develop and evaluate a decoder-only large language model for phoneme-to-text conversion in lip-reading, to establish its accuracy on a deduplicated evaluation, to measure and improve its robustness to imperfect input, and to characterise its errors. On the evidence of the results, the aim was met.

Each objective was achieved. The zero-shot baseline established that the pretrained model cannot perform the task without training, reaching near-zero exact match. The fine-tuned model, adapted with QLoRA, reached 88.82% exact match on the held-out test set, which exceeds the published encoder-decoder result of 85.11% on the same test set. The contamination audit found that roughly one evaluation sentence in eight is memorised from training, and the deduplicated figure of 87.44% is reported as the measure of genuine generalisation. The noise-augmentation experiment showed that the clean-trained decoder is fragile under corrupted input and that noise-augmented training restores robustness for a small clean-input cost. The three-stage error analysis characterised the residual errors and tested the homophone assumption directly.

## 12.2 Contributions

The project makes four contributions to the phoneme-to-text stage of lip-reading. It shows that a decoder-only language model fine-tuned with QLoRA outperforms the published encoder-decoder model on this task, on a single graphics card. It contributes a contamination audit and a deduplicated evaluation for LRS2, giving an unbiased measure of task difficulty that a figure computed on the standard split overstates. It introduces noise-augmented training for this stage and quantifies the robustness it buys against the clean-input cost it carries. It applies a staged error analysis to two models and shows that homophones are a small minority of the errors, which corroborates and extends the earlier finding on this task.

## 12.3 Critical reflection and future work

Two starting assumptions did not survive contact with the evidence, and recognising that is the main lesson of the project. Homophones were expected to be the central difficulty and turned out to be a minor one, which redirected the work toward accuracy, deduplicated evaluation, and robustness. A single accuracy number was expected to describe the model, and it turned out that the number depended heavily on how the data was split and how the model decoded, which is why the deduplication and the decoding control became central to the method. The value of the work is as much in measuring these things carefully as in the accuracy achieved.

The project also has clear limits that set the direction for further work. The headline figures are on the validation and test sets of a single corpus with no speaker identifiers, so a speaker-independent evaluation on a second corpus would strengthen the claims. The noise augmentation applies a fixed corruption to each example rather than resampling it each epoch, and the fifth error-classification method, using an instructed language model, was implemented but not run for want of a reliable judge model; both are natural extensions. The most important direction is to integrate the decoder with a real visual front-end, so that the robustness result can be tested against genuine front-end errors rather than simulated ones. That step would turn the deployment claim made here into a measured one, and it is the route by which this work could become a publishable contribution to automated lip-reading.
