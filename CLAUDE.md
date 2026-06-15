# CLAUDE.md

## What we're building

A word-level, decoder-only transformer trained on Trump speech
transcripts, built from scratch in PyTorch — following the spirit of
Karpathy's nanoGPT walkthrough, but learn-as-you-go rather than
front-loaded theory.

**Goal:** understand every component well enough to explain it, not just
get a model that runs. 4-day scope, Python-comfortable, no prior ML
background.

**Build order:**
1. Data prep — collect/clean Trump speech transcripts, build a
   word-level tokenizer
2. Bigram baseline — simplest possible "given this word, predict the
   next" model, no neural net yet
3. Transformer pieces, added one at a time — embeddings, self-attention,
   multi-head attention, feed-forward, stacked into a decoder-only
   transformer
4. Training — loss function (cross-entropy), training loop,
   `.backward()`
5. Generation — sampling, temperature, check it actually sounds like him

**Status tracking:** progress notes live in `notes/`, one file per stage
(e.g. `notes/01-data-prep.md`), each with a short header:

```yaml
---
stage: data-prep
status: done | verified | in-progress | assumption
date: 2026-06-14
---
```

followed by what was done, what's been verified, and any assumptions
made that haven't been checked yet. Check `notes/` for current state
before starting work on a stage.

---

## 1. Think Before Coding
Don't guess silently. State assumptions before writing code — if
something's ambiguous, lay out the interpretations instead of picking
one. If there's a simpler way to do something, say so, even if it means
pushing back. If something's genuinely unclear, stop and ask rather than
plowing ahead.

## 2. Simplicity First
Write the minimum needed for the current stage — nothing speculative.
No extra config options, abstractions, or error handling for cases that
can't happen yet. This is a learning project: a 50-line version that's
understood beats a 200-line version that "might be useful later."

## 3. Surgical Changes
When editing existing code, change only what the task requires. Don't
tidy up unrelated code or refactor things that work, even if you'd write
them differently. If your change leaves something unused, remove that —
but leave pre-existing dead code alone (mention it instead).

## 4. Goal-Driven Execution
Before starting a stage, define what "done" looks like — a check, a
test, an expected output — and work until that's satisfied. For
multi-step work, sketch a short plan with a verification step per part,
e.g.:

```
1. Build word-level tokenizer → verify: encode/decode round-trips on a sample string
2. Build bigram model → verify: loss decreases over a few training steps
```
