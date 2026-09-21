# Real-world failures — logs that broke on their own

Everything in [`../gallery/`](../gallery/) is a fault injected deliberately, one
knob at a time, so the rules can be validated against a known cause. That is the
right way to test a rule and the wrong way to prove a tool matters, because a
reader can fairly say: you only catch failures you built yourself.

This directory is the other half. Nobody broke these runs. They broke.

## `xtts_diverged` — Coqui XTTS v2 fine-tune, LJSpeech, 2026-02-11

**This log is a truncated prefix of a longer run.** See the correction below before
relying on any interpretation on this page.

A 9.8-hour segment (35,131s wall, 1,459 logged loss points, ending at step 72,900
of the run's eventual 125,039). Judge it:

```bash
trainproof epoch examples/real_world/xtts_diverged/trainer_0_log.txt --format coqui
```

```text
[FAIL] TP-DIVERGE: Loss curve is diverging.
       Evidence: End loss 0.031 vs Min loss 0.019
```

The loss reached its minimum at step 48,350, which is 66% of the way through, and
**this log ends** 1.62x above that minimum.

### Correction (2026-09-21): this file is a prefix, and the run did not stop here

This page used to say *"Coqui's own bookkeeping agrees"*, citing
`best_model_49880.pth` against `checkpoint_70000.pth`. Two independent forensic
audits established that **this log is a prefix of
`../../evidence/xtts_coqui_feb2026/trainer_0_log.txt`** — the same continuous run,
truncated at step 72,900 of 125,039. The two are identical over that range after
line-ending normalisation and the 14-line `<TTS>` redaction this directory's
`run_meta.json` records; no numeric value, timestamp or step differs between them.
Verify it yourself:

```bash
# this file's 1,459 loss points are the first 1,459 of the full log
grep "BEST MODEL" trainer_0_log.txt | tail -1                                  # best_model_49880.pth
grep "BEST MODEL" ../../evidence/xtts_coqui_feb2026/trainer_0_log.txt | tail -1 # best_model_124700.pth
```

The continuation supersedes the bookkeeping quoted above. In the complete run the
trainer promoted `best_model_124700.pth` — step 124,700 of 125,039, **99.7% of the
way through** — and all six retained held-out `avg_loss` evaluations improve,
including the last (4.8813 → 2.5894). The epoch-aggregated training loss ends at
its own minimum.

So the sentence *"the trainer knew its best weights were roughly 23,000 steps
behind the end"* was true of the prefix and is **not true of the run**. What the
full evidence supports is that the per-micro-batch training display loss ended
above its own minimum — a property of that series, not of the model. **No audio or
perceptual evaluation of this run is retained, so nothing here shows the model got
better either.**

This artifact is kept, unchanged, as the historical record of what the earlier
claim was computed from.

This log is also the only Coqui-format fixture in the repo, so it regression-tests
that adapter against a real 580KB file rather than a synthetic one.

### On the redaction

One local filesystem prefix was replaced with `<TTS>` in 14 lines. No number,
timestamp or step was touched, and trainproof's verdict and evidence strings are
byte-identical before and after — verifiable by anyone with a similar log, and the
reason the redaction is disclosed here rather than quietly done. The unredacted
original is retained privately as the provenance record.
