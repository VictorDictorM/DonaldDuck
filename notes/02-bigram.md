---
stage: bigram
status: verified
date: 2026-06-14
---

## What was built

- `tokenizer.py`: word-level tokenizer over `data/sample.txt`. Lowercases
  text and splits into words/punctuation tokens via regex. Builds
  `word_to_id` / `id_to_word` from the unique tokens (61 total), with
  `encode`/`decode` functions.
- `bigram.py`: count-based bigram model (no neural net). Builds a
  `61 x 61` counts tensor from consecutive token-id pairs in the sample,
  applies +1 smoothing, and normalizes each row into a probability
  distribution `P(next word | current word)`. Includes a `generate` function
  that samples a sequence via `torch.multinomial`.

## Verification results

- `python tokenizer.py`:
  - vocab size: 61
  - round-trip ok: True
  - spot-check mappings all correct (`'the' -> 44 -> 'the'`, etc.)
- `python bigram.py`:
  - max row-sum deviation from 1.0: 1.19e-07 (all rows sum to ~1)
  - top-3 words after `'great'`: `again` (0.046), `deals` (0.031),
    `.` (0.031) — plausible given phrases like "great again" and "great
    deals" in the sample.
  - generated (25 words, starting from `"we"`):
    > we be tired make thing much be , know very you on tell . now by right
    > thank working great this deals they tell ' made

## Commentary

The generated text is, as expected, incoherent — the model only ever
conditions on a single previous word, so it produces locally plausible word
pairs (e.g. "great deals", "tell .") without any sentence-level structure.
This is the baseline the transformer (stages 3-5) should improve on.

## Carried-over assumption

Per `notes/01-data-prep.md`, this is all built on the small hand-written
sample. Vocab size, smoothing behavior, and generation quality should be
re-checked once a real transcript corpus replaces `data/sample.txt`.
