"""SARIF 2.1.0 output, so findings land in GitHub PR annotations.

A training-log finding has a file but no meaningful line, so every result is
anchored to the log with a region of line 1: GitHub's code-scanning ingest
requires a physical location, and a synthetic step-derived line number would
point at nothing a reader could open.

Levels map to SARIF's three: FAIL -> error, WARN -> warning, everything else
(INFO, PASS) -> note. INFO and PASS findings are still emitted, because a run
whose PASS states which checks were skipped is exactly what a reviewer needs.
"""

SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json"

_LEVELS = {"FAIL": "error", "WARN": "warning"}


def _level(finding):
    return _LEVELS.get(finding.get("level", ""), "note")


def to_sarif(reports, version):
    """Build a SARIF log from trainproof reports (the JSON envelope's list)."""
    rules = {}
    results = []

    for report in reports:
        location = report.get("file") or report.get("path") or "training-run"
        for finding in report.get("findings", []):
            rule_id = finding.get("id")
            if not rule_id:
                continue

            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "shortDescription": {"text": finding.get("message", rule_id)},
                    "defaultConfiguration": {"level": _level(finding)},
                    "helpUri": "https://github.com/Mormolykos/trainproof/blob/main/RULES.md",
                }

            text = finding.get("message", "")
            evidence = finding.get("evidence")
            if evidence:
                text = f"{text} Evidence: {evidence}"
            source = finding.get("source")
            if source == "compare":
                text = f"{text} (vs. baseline)"

            results.append({
                "ruleId": rule_id,
                "ruleIndex": list(rules).index(rule_id),
                "level": _level(finding),
                "message": {"text": text},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": str(location).replace("\\", "/")},
                        "region": {"startLine": 1},
                    }
                }],
            })

    return {
        "$schema": SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "trainproof",
                    "version": version,
                    "informationUri": "https://github.com/Mormolykos/trainproof",
                    "rules": list(rules.values()),
                }
            },
            "results": results,
        }],
    }
