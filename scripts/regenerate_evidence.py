"""Regenerate EVIDENCE_MATRIX.md from the gallery logs, at the current version.

The matrix used to be maintained by hand. It drifted: it carried verdicts from
v0.3-dev long after the rules had changed, and prose numbers in README.md
disagreed with the logs they described. A document that describes data should be
derived from that data, for the same reason the golden snapshots exist.

Every cell here is computed by running the shipped rules over the shipped logs.
The observations at the bottom are derived too -- nothing in the output is a
claim someone typed and hoped stayed true.

    uv run python scripts/regenerate_evidence.py            # write the file
    uv run python scripts/regenerate_evidence.py --check     # fail if stale (CI)
"""

import argparse
import sys
from pathlib import Path

import trainproof
from trainproof.adapters import parse_log_with_format_info
from trainproof.compare import check_compare
from trainproof.epoch import check_epoch

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "examples" / "gallery"
EVIDENCE = ROOT / "evidence"
OUT = ROOT / "EVIDENCE_MATRIX.md"

# Real training runs from frameworks other than HuggingFace. The gallery above
# is synthetic-by-design -- one knob broken at a time -- which proves the rules
# fire but not that they survive contact with logs nobody wrote for us. These
# did not exist before v0.14.0, when the rules had only ever been tested against
# trainer_state.json. `None` means "the event file in this directory".
EVIDENCE_RUNS = [
    ("Coqui XTTS v2", "xtts_coqui_feb2026", "trainer_0_log.txt", "coqui"),
    ("Coqui XTTS v2", "xtts_coqui_feb2026", None, "tfevents"),
    ("Lightning / Fish Speech", "fish_lightning_feb2026", None, "tfevents"),
    # The run v0.20.0 returned PASS and exit 0 on. Added in v0.21 with the two
    # rules it produced; see that directory's README for why five separate
    # checks each declined. Kept in the matrix because a rule justified by a
    # real log should be re-proved against that log on every regeneration.
    ("TinyLlama 1.1B SFT", "web_chat_tinyllama_apr2025", "trainer_state.json", "hf"),
]

BASELINE = "healthy"
SEED_DIRS = {42: None, 43: "seed43", 44: "seed44"}

# Hand-written history, carried forward verbatim through every regeneration.
# The tables below are derived and disposable; this is why a rule exists, which
# no amount of re-running the gallery can recover.
HISTORY = """\
Recorded against v0.3-dev, and still true: `lr_zero` seed 43 evaded the dead-run
rule because batch-order noise faked >5% improvement under lr=0, and it passes
comparison because its losses land near baseline -- the pretrained model never
moved. That drove the total-zero-LR fatality rule: lr=0 on every step means the
optimizer never stepped, so it fails from the lr column itself, immune to loss
noise. The tables above show that rule is still the only thing catching it."""


def log_for(config: str, seed: int) -> str:
    sub = SEED_DIRS[seed]
    d = GALLERY / config if sub is None else GALLERY / config / sub
    return str((d / "trainer_state.json").resolve())


def configs() -> list[str]:
    """Discovered from disk, never declared: a new run cannot be left out."""
    found = sorted(p.name for p in GALLERY.iterdir() if p.is_dir())
    return [BASELINE] + [c for c in found if c != BASELINE]


def evidence_path(subdir: str, filename: str | None) -> Path | None:
    d = EVIDENCE / subdir
    if not d.is_dir():
        return None
    if filename is not None:
        p = d / filename
        return p if p.exists() else None
    events = sorted(p for p in d.iterdir() if "tfevents" in p.name)
    return events[0] if events else None


def evidence_rows() -> list[dict]:
    """Judge each real-world log with the shipped rules, at generation time."""
    rows = []
    for framework, subdir, filename, fmt in EVIDENCE_RUNS:
        path = evidence_path(subdir, filename)
        if path is None:
            continue
        records, *_ = parse_log_with_format_info(path, fmt=fmt)
        report = check_epoch(str(path), fmt=fmt)
        steps = [r["step"] for r in records if "step" in r]
        rows.append({
            "framework": framework,
            "file": path.name,
            "fmt": fmt,
            "records": len(records),
            "span": f"{int(min(steps))}..{int(max(steps))}" if steps else "-",
            "report": report,
            "rules": rules_of(report),
        })
    return rows


def cell(report: dict) -> str:
    # INFO findings are marked. Listing them beside the rules that produced the
    # verdict reads as several grounds for it; the 2026-09-21 audits found the
    # XTTS FAIL cell -- `TP-DIVERGE`, `TP-THROUGHPUT` -- read that way, when
    # TP-THROUGHPUT is INFO and cannot contribute to any verdict. The verdict
    # itself is unchanged; only which findings support it is now visible.
    parts = []
    for f in sorted(report["findings"], key=lambda f: f["id"]):
        mark = " *(INFO)*" if f.get("level") == "INFO" else ""
        parts.append(f"`{f['id']}`{mark}")
    return f"**{report['verdict']}**<br>{', '.join(parts) or '(none)'}"


def rules_of(report: dict) -> set[str]:
    return {f["id"] for f in report["findings"]}


def build() -> str:
    cfgs = configs()
    seeds = sorted(SEED_DIRS)

    epoch = {
        (c, s): check_epoch(log_for(c, s), fmt="hf") for c in cfgs for s in seeds
    }
    compare = {
        (c, s): check_compare(log_for(c, s), log_for(BASELINE, s), fmt="hf")
        for c in cfgs
        for s in seeds
        if c != BASELINE
    }
    cross = [
        (43, 42, check_compare(log_for(BASELINE, 43), log_for(BASELINE, 42), fmt="hf")),
        (44, 42, check_compare(log_for(BASELINE, 44), log_for(BASELINE, 42), fmt="hf")),
        (44, 43, check_compare(log_for(BASELINE, 44), log_for(BASELINE, 43), fmt="hf")),
    ]

    head = " | ".join(f"seed {s}" for s in seeds)
    rule = "|".join(["---"] * (len(seeds) + 1))

    L = [
        "# Evidence matrix",
        "",
        "<!-- GENERATED by scripts/regenerate_evidence.py -- do not edit by hand. -->",
        "",
        (
            f"trainproof **{trainproof.__version__}** &middot; "
            f"{len(cfgs)} configs x {len(seeds)} seeds = {len(cfgs) * len(seeds)} runs "
            "&middot; Qwen2.5-3B-Instruct, 4-bit NF4, LoRA r=16, 300 steps"
        ),
        "",
        "Seed 42 is the log at each config root; 43 and 44 are nested beside it.",
        "",
        "## Single-run verdicts (`trainproof epoch`)",
        "",
        f"| config | {head} |",
        f"|{rule}|",
    ]
    for c in cfgs:
        L.append(f"| `{c}` | " + " | ".join(cell(epoch[(c, s)]) for s in seeds) + " |")

    L += [
        "",
        "## Comparison against same-seed healthy (`trainproof compare`)",
        "",
        f"| config | {head} |",
        f"|{rule}|",
    ]
    for c in cfgs:
        if c == BASELINE:
            continue
        L.append(f"| `{c}` | " + " | ".join(cell(compare[(c, s)]) for s in seeds) + " |")

    L += [
        "",
        "## Cross-seed baseline sanity",
        "",
        "Healthy judged against healthy from a *different* seed. Anything other than",
        "a clean pass here means the compare rules are reading seed noise as signal.",
        "",
        "| run | baseline | verdict |",
        "|---|---|---|",
    ]
    for run_s, base_s, rep in cross:
        L.append(f"| healthy seed {run_s} | healthy seed {base_s} | {cell(rep)} |")

    # ---- real runs from other frameworks --------------------------------------
    ev = evidence_rows()
    if ev:
        L += [
            "",
            "## Real runs, other frameworks (`trainproof epoch`)",
            "",
            "Training runs nobody wrote for trainproof, judged by the shipped rules.",
            "The logs are in `evidence/`. Both XTTS rows are the *same run* read by two",
            "parsers -- a text log and a binary event file. They are two readers of one",
            "logging process, not two independent measurements of the run.",
            "",
            "| framework | log | format | records | steps | verdict |",
            "|---|---|---|---|---|---|",
        ]
        for r in ev:
            L.append(
                f"| {r['framework']} | `{r['file']}` | `{r['fmt']}` | "
                f"{r['records']} | {r['span']} | {cell(r['report'])} |"
            )

    # ---- derived observations: computed, not asserted -------------------------
    obs = []

    # Two readers, one run. This used to be emitted as "two independent readers
    # agree exactly", and presented as corroboration. The 2026-09-21 forensic
    # audits established that it is neither:
    #
    #   * NOT INDEPENDENT. They read one logging process. Worse, on the XTTS run
    #     the `coqui` reader drops the six eval sections the text log contains
    #     (`avg_loss` is absent from CANONICAL_ALIASES) while the `tfevents`
    #     reader keeps them. Agreement between a sighted reader and a blind one
    #     is not corroboration.
    #   * NOT EXACT. The verdict and rule set match; coverage does not.
    #
    # So compare coverage too, and report what actually matched instead of
    # asserting a conclusion the comparison does not support.
    by_file = {}
    for r in ev:
        by_file.setdefault(r["framework"], []).append(r)
    for framework, rows in by_file.items():
        if len(rows) < 2:
            continue
        rulesets = {frozenset(r["rules"]) for r in rows}
        verdicts = {r["report"]["verdict"] for r in rows}
        fmts = ", ".join(f"`{r['fmt']}`" for r in rows)
        covs = [r["report"].get("checks", {}).get("coverage", {}) for r in rows]
        cov_txt = ", ".join(
            f"`{r['fmt']}` {c.get('checked', '?')}/{c.get('total', '?')}"
            for r, c in zip(rows, covs, strict=False)
        )
        cov_same = len({(c.get("checked"), c.get("total")) for c in covs}) == 1
        if len(rulesets) == 1 and len(verdicts) == 1:
            obs.append(
                f"{framework}: {fmts} are two readers of one logging process. They "
                f"reach the same verdict and the same rule set"
                + (
                    f", and examine the same amount of the run ({cov_txt})."
                    if cov_same
                    else f", but examine **different amounts of the run** ({cov_txt}) -- "
                    "so this is agreement between readers with different visibility, "
                    "not independent corroboration."
                )
            )
        else:
            obs.append(
                f"{framework}: {fmts} read the same run and **disagree** "
                f"(verdicts {sorted(verdicts)}). One of these parsers is wrong."
            )

    # Severity, not just FAIL: a mode that can only reach WARN on a broken run is
    # a mode that would not stop CI, and that is the interesting asymmetry.
    sev = {"PASS": 0, "WARN": 1, "FAIL": 2}
    for c in cfgs:
        if c == BASELINE:
            continue
        e_max = max(sev[epoch[(c, s)]["verdict"]] for s in seeds)
        c_max = max(sev[compare[(c, s)]["verdict"]] for s in seeds)
        worst = {v: k for k, v in sev.items()}
        if c_max > e_max:
            obs.append(
                f"`{c}`: comparison reaches **{worst[c_max]}**, the single-run rules "
                f"only **{worst[e_max]}**. A single run cannot condemn this one."
            )
        elif e_max > c_max:
            obs.append(
                f"`{c}`: the single-run rules reach **{worst[e_max]}**, comparison "
                f"only **{worst[c_max]}**. A baseline cannot condemn this one."
            )

    lonely = [
        (c, s, next(iter(rules_of(epoch[(c, s)]))))
        for c in cfgs
        for s in seeds
        if epoch[(c, s)]["verdict"] == "FAIL" and len(rules_of(epoch[(c, s)])) == 1
    ]
    for c, s, only in lonely:
        obs.append(
            f"`{c}` seed {s} is a FAIL that exactly one rule catches: {only}. "
            f"Remove that rule and this run passes."
        )

    for c in cfgs:
        verdicts = {epoch[(c, s)]["verdict"] for s in seeds}
        ruleset = {frozenset(rules_of(epoch[(c, s)])) for s in seeds}
        if len(verdicts) > 1:
            obs.append(f"`{c}`: single-run verdict is not stable across seeds ({sorted(verdicts)}).")
        elif len(ruleset) > 1:
            obs.append(
                f"`{c}`: same verdict in every seed, but not the same rules -- "
                f"which rule fires is seed-dependent."
            )

    L += ["", "## Derived observations", ""]
    L += [f"- {o}" for o in obs] or ["- (none)"]
    L += ["", "## Why the zero-LR rule exists", "", HISTORY, ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the file is stale")
    args = ap.parse_args()

    fresh = build()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != fresh:
            print(f"STALE: {OUT.name} does not match the gallery at "
                  f"trainproof {trainproof.__version__}. Regenerate it.", file=sys.stderr)
            return 1
        print(f"{OUT.name} is current at trainproof {trainproof.__version__}.")
        return 0

    OUT.write_text(fresh, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} at trainproof {trainproof.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
