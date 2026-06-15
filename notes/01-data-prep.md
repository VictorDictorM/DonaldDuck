---
stage: data-prep
status: assumption
date: 2026-06-14
---

## What was done

Instead of scraping a full corpus of Trump speech transcripts up front, we
wrote a small hand-crafted sample (`data/sample.txt`, ~190 words) in a
Trump-speech-like rhetorical style (repetitive phrasing, signature phrases
like "tremendous", "believe me", "many people are saying"). This lets us
build and verify the tokenizer and bigram model mechanics end-to-end before
investing in real data collection.

## Assumptions (not yet checked against real data)

- A simple regex-based word/punctuation tokenizer (lowercase, split words and
  punctuation into separate tokens) is "good enough" for this stage.
- `data/sample.txt` is designed as a drop-in replacement target: when a real
  scraped corpus replaces it, `tokenizer.py` and `bigram.py` should not need
  to change — only the input file does.
- Vocab size (61 on the sample) will be much larger on a real corpus, and
  punctuation/OOV handling may need revisiting at that point.

## Deferred

Full transcript collection and cleaning is deferred to a later task.
