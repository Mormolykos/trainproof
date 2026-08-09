"""Checks on the loss objective itself, rather than on the data or the curve.

Both checks here exist because of one real failure. A VALL-E-X-derived TTS model
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


def check_objective(num_classes, ignore_index=None, targets=None):
    """One-shot convenience wrapper. `targets` may be a single batch or an iterable."""
    findings = list(check_ignore_index(num_classes, ignore_index))

    if targets is not None:
        cov = TargetCoverage(num_classes, ignore_index)
        if isinstance(targets, (list, tuple)) and targets and hasattr(targets[0], "__len__"):
            for batch in targets:
                cov.observe(batch)
        else:
            cov.observe(targets)
        findings.extend(cov.result())

    return findings
