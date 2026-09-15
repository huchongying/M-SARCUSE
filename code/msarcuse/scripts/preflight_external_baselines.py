from __future__ import annotations

import json
from collections import Counter
from paper_protocol import (
    ROOT, PACKAGE_ROOT, ARTIFACT_ROOT, SEEDS, load_paper_inputs, sha256, read_rows,
)
from sarcnext_common import atomic_json


def main() -> None:
    manifest = json.loads((PACKAGE_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    for path in sorted((ROOT / "config").glob("*.json")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if relative not in expected or sha256(path) != expected[relative]["sha256"]:
            raise ValueError(f"Release config hash mismatch: {relative}")
    inputs = load_paper_inputs(require_training=True)
    config_path = ROOT / "config" / "msarcuse_external_baselines.draft.json"
    config = json.loads(config_path.read_text())
    if config["seeds"] != SEEDS:
        raise ValueError("Seed list differs from the supplied protocol")
    if config["bootstrap"]["resamples"] != 2000 or config["bootstrap"]["clustered_by"] != "episode":
        raise ValueError("Expected 2000 episode-cluster bootstrap draws")
    artifacts = []
    rec_path = ARTIFACT_ROOT / "data" / "recognition_records.jsonl"
    recognition = read_rows(rec_path)
    rec_map = {r["source_index"]: r for r in recognition}
    if len(rec_map) != len(recognition):
        raise ValueError("Duplicate recognition source indices")
    for row in recognition:
        if row["evaluation_fold"] != inputs["episodes"][row["episode_id"]]:
            raise ValueError("Recognition record uses a different fold map")
    artifacts.append({"path": str(rec_path), "sha256": sha256(rec_path)})
    for fold in range(4):
        path = ARTIFACT_ROOT / "data" / f"fold-{fold}" / "response_tasks.jsonl"
        rows = read_rows(path)
        expected_targets = {r["sample_id"]: (split, r)
                            for split, values in (
                                ("training", inputs["training"][fold]),
                                ("evaluation", [r for r in inputs["targets"] if r["fold"] == fold]))
                            for r in values}
        groups = {}
        for row in rows:
            groups.setdefault(row["sample_id"], []).append(row)
        if set(groups) != set(expected_targets):
            raise ValueError(f"Fold {fold}: task membership differs from final inputs")
        for sample, candidates in groups.items():
            split, target = expected_targets[sample]
            ref = [(c["candidate_id"], c["source_index"], c["text"], c["is_gold"])
                   for c in target["candidates"]]
            actual = [(c["candidate_id"], c["candidate_source_index"],
                       c["candidate_text"], c["is_gold"]) for c in candidates]
            if actual != ref:
                raise ValueError(f"Fold {fold}: candidate identity/order differs for {sample}")
            for row in candidates:
                if (row["split"] != split or row["fold"] != fold
                        or row["source_index"] != target["source_index"]
                        or row["episode_id"] != target["episode_id"]
                        or row["target_label"] != target["target_label"]):
                    raise ValueError(f"Fold {fold}: target metadata differs for {sample}")
            rec = rec_map[target["source_index"]]
            if rec["episode_id"] != target["episode_id"] or rec["target_label"] != target["target_label"]:
                raise ValueError("Recognition/response membership mismatch")
        artifacts.append({"path": str(path), "sha256": sha256(path)})
    for filename in ("common_training_targets.json", "method_neutral_tasks.json"):
        path = ARTIFACT_ROOT / "manifests" / filename
        report = json.loads(path.read_text())
        if report["input_config_sha256"] != inputs["input_config_sha256"]:
            raise ValueError("Task manifest is stale")
        artifacts.append({"path": str(path), "sha256": sha256(path)})
    output = ARTIFACT_ROOT / "manifests" / "preflight.json"
    atomic_json(output, {"schema_version": 2, "status": "passed",
                         "input_config_sha256": inputs["input_config_sha256"],
                         "config_sha256": sha256(config_path),
                         "fold_target_counts": inputs["fold_counts"],
                         "episodes": len(inputs["episodes"]),
                         "artifacts": artifacts})
    print(json.dumps({"status": "passed", "output": str(output)}))


if __name__ == "__main__":
    main()
