from __future__ import annotations

import json
from paper_protocol import ARTIFACT_ROOT, load_paper_inputs, sha256
from sarcnext_common import atomic_json, write_jsonl


def main() -> None:
    inputs = load_paper_inputs(require_training=True)
    summaries = []
    for fold, rows in inputs["training"].items():
        path = ARTIFACT_ROOT / "data" / f"fold-{fold}" / "selector_training_targets.jsonl"
        write_jsonl(path, rows)
        summaries.append({"fold": fold, "path": str(path), "sha256": sha256(path),
                          "targets": len(rows),
                          "training_and_evaluation_episode_overlap": False})
    atomic_json(ARTIFACT_ROOT / "manifests" / "common_training_targets.json",
                {"schema_version": 2, "status": "materialized",
                 "input_config_sha256": inputs["input_config_sha256"],
                 "folds": summaries})
    print(json.dumps({"status": "materialized", "folds": len(summaries)}))


if __name__ == "__main__":
    main()
