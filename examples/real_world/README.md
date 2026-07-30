# Real-world failures — logs that broke on their own

Everything in [`../gallery/`](../gallery/) is a fault injected deliberately, one
knob at a time, so the rules can be validated against a known cause. That is the
right way to test a rule and the wrong way to prove a tool matters, because a
reader can fairly say: you only catch failures you built yourself.

This directory is the other half. Nobody broke these runs. They broke.

## `xtts_diverged` — Coqui XTTS v2 fine-tune, LJSpeech, 2026-02-11

A 9.8-hour fine-tune (35,131s wall, 1,459 logged loss points, ending at step
72,900). Judge it:

```bash
trainproof epoch examples/real_world/xtts_diverged/trainer_0_log.txt --format coqui
```

```text
[FAIL] TP-DIVERGE: Loss curve is diverging.
       Evidence: End loss 0.031 vs Min loss 0.019
```

The loss reached its minimum at step 48,350, which is 66% of the way through, and
the run ended 1.62x above that minimum.

The part worth checking yourself: **Coqui's own bookkeeping agrees.** The last
`BEST MODEL` line in the log is `best_model_49880.pth`, and the last checkpoint
written is `checkpoint_70000.pth`. The trainer knew its best weights were roughly
23,000 steps behind the end, and kept going for about three more hours anyway.
Nothing in the stack raised a word about it, because nothing in the stack was
looking.

```bash
grep "BEST MODEL" trainer_0_log.txt | tail -1     # best_model_49880.pth
grep -o "checkpoint_[0-9]*" trainer_0_log.txt | tail -1   # checkpoint_70000
```

This log is also the only Coqui-format fixture in the repo, so it regression-tests
that adapter against a real 580KB file rather than a synthetic one.

### On the redaction

One local filesystem prefix was replaced with `<TTS>` in 14 lines. No number,
timestamp or step was touched, and trainproof's verdict and evidence strings are
byte-identical before and after — verifiable by anyone with a similar log, and the
reason the redaction is disclosed here rather than quietly done. The unredacted
original is retained privately as the provenance record.
