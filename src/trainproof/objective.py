"""Checks on the loss objective itself, rather than on the data or the curve.

The checks here exist because of one real failure. A VALL-E-X-derived TTS model
trained for fifteen epochs with `eos_id` and `ignore_index` set to the same integer.
Every end-of-sequence target was therefore discarded before the loss saw it, and the
model was never taught to stop. The loss curve was healthy throughout; a minimal
reproduction put the broken and fixed arms at 0.0035 and 0.0034 final loss, so no
curve-shaped check can separate them. These are step-0 and first-epoch checks for
exactly that reason.
"""

from . import rules


def _to_int_list(targets):
    """Accept a torch tensor, numpy array, or nested sequence of ints."""
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu()
    if hasattr(targets, "reshape") and hasattr(targets, "tolist"):
        targets = targets.reshape(-1).tolist()
    elif hasattr(targets, "tolist"):
        targets = targets.tolist()
    out = []
    stack = [targets]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            out.append(int(item))
    return out


def check_ignore_index(num_classes, ignore_index):
    """Is the loss's ignore sentinel a real, predictable class?

    Safe sentinels sit outside [0, num_classes): PyTorch's default -100, or an id
    exactly equal to num_classes (one past the end). A sentinel INSIDE the range is
    a class the output layer can emit and the loss can never learn.
    """
    if ignore_index is None:
        return []

    ignore_index = int(ignore_index)
    num_classes = int(num_classes)

    if 0 <= ignore_index < num_classes:
        return [{
            "id": "TP-OBJ-IGNORE-INDEX-COLLISION",
            "level": "FAIL",
            "message": (
                "ignore_index is a valid class in the output layer - every target "
                "with this id is silently dropped from the loss and can never be learned"
            ),
            "evidence": (
                f"ignore_index={ignore_index}, output layer has {num_classes} classes "
                f"(valid ids 0..{num_classes - 1}). Use a sentinel outside the range, "
                f"e.g. -100 or {num_classes}."
            ),
        }]

    if ignore_index == num_classes:
        return [{
            "id": "TP-OBJ-IGNORE-INDEX-OK",
            "level": "INFO",
            "message": "ignore_index sits one past the last class, which is outside the output layer",
            "evidence": f"ignore_index={ignore_index}, classes 0..{num_classes - 1}",
        }]

    return []


def _to_rows(seq):
    """Normalise a batch of ids to a list of int rows. Returns [] if it cannot.

    The slice happens BEFORE any device transfer. Inspecting a few dozen positions must
    not copy an entire batch off the accelerator, and slicing behaves identically on a
    tensor and on a list, so no tensor library is needed to do it.
    """
    # Row slice first, while this may still be a device tensor.
    try:
        seq = seq[: rules.LABEL_ALIGNMENT_MAX_ROWS]
    except (TypeError, IndexError, KeyError):
        return []

    if hasattr(seq, "detach"):
        seq = seq.detach().cpu()
    if hasattr(seq, "tolist"):
        seq = seq.tolist()
    if not isinstance(seq, (list, tuple)) or not seq:
        return []

    def tail(row):
        # The TAIL, never the head: SFT masks the prompt on the left, so a head slice
        # of a long sequence sees only ignore_index. Contiguous, so i / i+1 survives.
        return row[-rules.LABEL_ALIGNMENT_MAX_COLS:]

    if isinstance(seq[0], (list, tuple)):
        rows = []
        for row in seq:
            if not isinstance(row, (list, tuple)):
                return []
            try:
                rows.append([int(v) for v in tail(row)])
            except (TypeError, ValueError):
                return []
        return rows
    try:
        return [[int(v) for v in tail(seq)]]
    except (TypeError, ValueError):
        return []


def check_label_alignment(input_ids, labels, ignore_index=-100, loss_shifts=None):
    """Are the labels aligned the way this loss expects, or shifted one time too many?

    In the HuggingFace causal-LM convention `labels` arrive ALIGNED with `input_ids`
    and the loss does the shift, scoring the logits at position i against the label at
    i+1. A caller who pre-shifts gets the shift twice, and the model is trained to
    predict two tokens ahead. Shapes stay valid, gradients flow, the loss falls, and
    the curve is indistinguishable from a healthy run.

    Two hypotheses are counted over every unmasked position:

        labels[i] == input_ids[i]      the labels are aligned
        labels[i] == input_ids[i + 1]  the labels are already shifted

    They are counted independently, not exclusively, because wherever a token repeats
    (`input_ids[i] == input_ids[i+1]`) a position satisfies both.

    ``loss_shifts`` is what turns an observation into a verdict, and it defaults to
    None:

      * **None -- unknown. This function will never FAIL.** Which arrangement is
        *correct* is a property of the loss, not of the tensors, and the tensors cannot
        reveal it. A custom loop that pre-shifts its labels AND pairs that with a loss
        that does not shift is training correctly. An earlier draft of this check
        failed exactly that case; a false FAIL under ``stop_on_fail`` aborts a correct
        run, which is the worst thing this library can do.
      * **True -- the loss shifts** (a confirmed HuggingFace causal LM): pre-shifted
        labels FAIL, aligned labels PASS.
      * **False -- the loss does not shift**: aligned labels FAIL, because the model is
        being trained to emit the token it was just given. This value is only ever
        supplied by a caller asserting it. **It is never inferred**, because there is no
        way to confirm a negative about someone else's loss function.

    The verdict is a vote over ROWS, not a pooled count over positions: positions inside
    one sequence are not independent observations. Rows that cannot discriminate are
    excluded from the vote rather than allowed to veto it.
    """
    rows_in = _to_rows(input_ids)
    rows_lab = _to_rows(labels)

    def not_measured(reason):
        return [{
            "id": "TP-OBJ-LABEL-SHIFT-NOT-MEASURED",
            "level": "NOT-CHECKED",
            "message": "label/input alignment was not measured",
            "evidence": reason,
        }]

    if not rows_in or not rows_lab:
        return not_measured("input_ids or labels could not be read as integer rows")
    if len(rows_in) != len(rows_lab):
        return not_measured(
            f"batch mismatch: {len(rows_in)} input rows vs {len(rows_lab)} label rows"
        )

    ignore_index = int(ignore_index)
    threshold = rules.LABEL_ALIGNMENT_MATCH_FRACTION

    total = 0
    votes = {"aligned": 0, "shifted": 0, "both": 0, "neither": 0}
    # Row counts were equalised above, so strict= expresses the invariant rather
    # than guarding against it.
    for ids, labs in zip(rows_in, rows_lab, strict=True):
        if len(ids) != len(labs):
            continue
        n_r = aligned_r = shifted_r = 0
        # The last position is excluded: input_ids[i+1] does not exist there, so it
        # could only ever feed the aligned hypothesis. Counting it would measure the
        # two hypotheses on different samples and bias the comparison toward PASS.
        for i in range(len(labs) - 1):
            if labs[i] == ignore_index:
                continue
            n_r += 1
            if labs[i] == ids[i]:
                aligned_r += 1
            if labs[i] == ids[i + 1]:
                shifted_r += 1
        total += n_r
        if n_r < rules.LABEL_ALIGNMENT_MIN_ROW_POSITIONS:
            continue
        a_r = aligned_r / n_r
        s_r = shifted_r / n_r
        if a_r >= threshold and s_r >= threshold:
            votes["both"] += 1
        elif s_r >= threshold:
            votes["shifted"] += 1
        elif a_r >= threshold:
            votes["aligned"] += 1
        else:
            votes["neither"] += 1

    judgeable = sum(votes.values())
    informative = votes["aligned"] + votes["shifted"]

    if total < rules.LABEL_ALIGNMENT_MIN_POSITIONS:
        return not_measured(
            f"only {total} comparable unmasked positions, "
            f"need {rules.LABEL_ALIGNMENT_MIN_POSITIONS}"
        )
    if not judgeable:
        return not_measured(
            f"no row carried {rules.LABEL_ALIGNMENT_MIN_ROW_POSITIONS} comparable positions"
        )
    if not informative:
        return not_measured(
            f"no row discriminated: {votes['both']} degenerate (repeated tokens), "
            f"{votes['neither']} where the labels are not a copy of input_ids"
        )
    # Informative rows must be most of what was judged. A batch where a minority of rows
    # happen to look like a copy of the inputs is not evidence about the collator.
    if informative * 2 < judgeable:
        return not_measured(
            f"only {informative} of {judgeable} judgeable rows discriminated "
            f"({votes['both']} degenerate, {votes['neither']} not a copy of input_ids)"
        )

    winner = "shifted" if votes["shifted"] >= votes["aligned"] else "aligned"
    agreement = votes[winner] / informative
    if agreement < rules.LABEL_ALIGNMENT_ROW_AGREEMENT:
        return not_measured(
            f"rows disagree: {votes['aligned']} aligned vs {votes['shifted']} shifted "
            f"of {informative} informative rows"
        )

    detail = (
        f"{votes[winner]}/{informative} informative rows (of {judgeable} judgeable, "
        f"{total} positions)"
    )

    if winner == "shifted":
        if loss_shifts is True:
            return [{
                "id": "TP-OBJ-LABEL-SHIFT-DOUBLE",
                "level": "FAIL",
                "message": (
                    "labels are pre-shifted against input_ids and this loss shifts them "
                    "again - the model is trained to predict two tokens ahead"
                ),
                "evidence": f"labels[i]==input_ids[i+1] in {detail}",
            }]
        if loss_shifts is False:
            return [{
                "id": "TP-OBJ-LABEL-SHIFT-OK",
                "level": "PASS",
                "message": "labels are pre-shifted, which is what a non-shifting loss expects",
                "evidence": f"labels[i]==input_ids[i+1] in {detail}",
            }]
        return not_measured(
            f"labels[i]==input_ids[i+1] in {detail}, but whether that is correct depends "
            "on whether this loss shifts, which cannot be read from the tensors. Pass "
            "loss_shifts=True (HuggingFace causal-LM convention) to judge it."
        )

    if loss_shifts is False:
        return [{
            "id": "TP-OBJ-LABEL-SHIFT-DOUBLE",
            "level": "FAIL",
            "message": (
                "labels are aligned with input_ids and this loss does not shift them - "
                "the model is trained to emit the token it was just given"
            ),
            "evidence": f"labels[i]==input_ids[i] in {detail}",
        }]
    if loss_shifts is True:
        return [{
            "id": "TP-OBJ-LABEL-SHIFT-OK",
            "level": "PASS",
            "message": "labels are aligned with input_ids, as the causal-LM loss expects",
            "evidence": f"labels[i]==input_ids[i] in {detail}",
        }]
    return not_measured(
        f"labels[i]==input_ids[i] in {detail}, but whether that is correct depends on "
        "whether this loss shifts, which cannot be read from the tensors. Pass "
        "loss_shifts=True (HuggingFace causal-LM convention) to judge it."
    )


class TargetCoverage:
    """Accumulates which classes ever appear as a positive target.

    A class that the output layer can emit but that is never once a target is a class
    the model has no way to learn. Feed it the same target tensor the loss receives,
    after any masking, for one epoch.
    """

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = int(num_classes)
        self.ignore_index = None if ignore_index is None else int(ignore_index)
        self.seen = set()
        self.n_targets = 0
        self.out_of_range = 0

    def observe(self, targets):
        for t in _to_int_list(targets):
            if self.ignore_index is not None and t == self.ignore_index:
                continue
            self.n_targets += 1
            if 0 <= t < self.num_classes:
                self.seen.add(t)
            else:
                self.out_of_range += 1
        return self

    def result(self):
        findings = []

        if self.out_of_range:
            findings.append({
                "id": "TP-OBJ-TARGET-OUT-OF-RANGE",
                "level": "FAIL",
                "message": "targets contain ids the output layer cannot represent",
                "evidence": f"{self.out_of_range} of {self.n_targets} targets outside 0..{self.num_classes - 1}",
            })

        missing = sorted(set(range(self.num_classes)) - self.seen)
        coverage = len(self.seen) / self.num_classes if self.num_classes else 0.0

        # A handful of never-targeted classes among otherwise broad coverage is a
        # structural exclusion. Hundreds of unseen classes is just a small sample,
        # and saying so would bury the signal we care about in noise.
        if coverage < rules.DEAD_CLASS_MIN_COVERAGE:
            findings.append({
                "id": "TP-OBJ-COVERAGE-INSUFFICIENT",
                "level": "INFO",
                "message": "too few distinct classes observed to judge dead classes",
                "evidence": f"{len(self.seen)}/{self.num_classes} classes seen over {self.n_targets} targets",
            })
        elif missing and len(missing) <= rules.DEAD_CLASS_MAX_REPORTED:
            findings.append({
                "id": "TP-OBJ-DEAD-CLASS",
                "level": "FAIL",
                "message": (
                    "class is present in the output layer but never appears as a training "
                    "target - the model cannot learn to emit it"
                ),
                "evidence": (
                    f"class(es) {missing} never targeted; {len(self.seen)}/{self.num_classes} "
                    f"other classes seen over {self.n_targets} targets"
                ),
            })
        elif not missing:
            findings.append({
                "id": "TP-OBJ-DEAD-CLASS-OK",
                "level": "PASS",
                "message": "every class in the output layer appears as a training target",
                "evidence": f"{self.num_classes} classes over {self.n_targets} targets",
            })

        return findings


def check_objective(num_classes, ignore_index=None, targets=None, input_ids=None,
                    loss_shifts=None):
    """One-shot convenience wrapper. `targets` may be a single batch or an iterable.

    `input_ids` is optional and enables the causal-LM alignment check only when the
    caller can supply the same batch the labels came from. Omitting it leaves that
    check unrun rather than guessed. `loss_shifts` defaults to None, which can observe
    an arrangement but will never fail one -- see `check_label_alignment`.
    """
    findings = list(check_ignore_index(num_classes, ignore_index))

    if input_ids is not None and targets is not None:
        findings.extend(check_label_alignment(
            input_ids, targets, -100 if ignore_index is None else ignore_index,
            loss_shifts=loss_shifts,
        ))

    if targets is not None:
        cov = TargetCoverage(num_classes, ignore_index)
        if isinstance(targets, (list, tuple)) and targets and hasattr(targets[0], "__len__"):
            for batch in targets:
                cov.observe(batch)
        else:
            cov.observe(targets)
        findings.extend(cov.result())

    return findings
