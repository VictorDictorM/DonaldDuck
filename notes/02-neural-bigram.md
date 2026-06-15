---
stage: neural-bigram
status: verified
date: 2026-06-15
---

## What was done

Built `neural_bigram.py`: a single `nn.Embedding(vocab_size, vocab_size)` layer
trained with cross-entropy loss + Adam (lr=0.1), full-batch (all consecutive
token pairs in `data/sample.txt` per step), 500 steps.

The embedding table IS the model — row `i` gives the logit distribution over
next tokens for input token `i`. No separate projection layer needed since
embed_dim == vocab_size.

Generation: feed the last generated token id into the model, softmax the
logits, sample with `torch.multinomial`, append, repeat — same sampling
approach as `bigram.py`, just drawing from learned softmax probs instead of
smoothed counts.

## What's been verified

- `x`/`y` shapes and forward-pass logits shape `(N, vocab_size)` as expected.
- Initial loss ≈ 4.80, close to `ln(vocab_size)` (uniform-guess baseline) —
  confirms untrained model starts at chance level.
- Loss decreases monotonically from 4.80 -> 0.82 and plateaus by ~step 300 —
  confirms the train loop (forward -> loss -> backward -> step) works.
- Top-3 words after "great" and a 25-token generated sample look qualitatively
  similar in style to the counting bigram's output.

## Observations / assumptions not deeply checked

- Final neural loss (0.82) is much lower than the counting bigram's
  Laplace-smoothed NLL on the same data (2.97). This is expected: the neural
  model converges toward the *unsmoothed* MLE (no +1 penalty for rare/unseen
  pairs), so it fits the training distribution more tightly. Not verified
  against a held-out set — with this little data, "low loss" here likely
  means memorization rather than generalization, but that's fine for this
  stage's goal (get the training loop working).
- lr=0.1 with Adam for 500 steps was chosen by feel (single linear layer,
  small/near-convex problem) and not tuned further once the loss plateaued.
