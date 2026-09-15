from __future__ import annotations

import copy
import json
import pandas as pd
from paper_protocol import ARTIFACT_ROOT, ROOT, checked_input, load_paper_inputs, sha256
from prepare_sarcnext import eligible_records
from sarcnext_common import atomic_json, write_jsonl


def candidate_rows(examples: list[dict], split: str, fold: int):
    for example in examples:
        turns = "\n".join(f"{t['speaker']}: {t['text']}" for t in example["context_turns"])
        context = f"{turns}\nRecorded next speaker: {example['next_speaker']}"
        for candidate in example["candidates"]:
            yield {"fold": fold, "split": split, "sample_id": example["sample_id"],
                   "dialogue_pair_id": example["dialogue_pair_id"],
                   "dialogue_id": example["dialogue_id"], "episode_id": example["episode_id"],
                   "source_index": example["source_index"], "target_label": example["target_label"],
                   "context_text": context, "candidate_id": candidate["candidate_id"],
                   "candidate_text": candidate["text"],
                   "candidate_source_index": candidate["source_index"],
                   "candidate_source_dialogue_id": candidate["source_dialogue_id"],
                   "is_gold": candidate["is_gold"], "target_audio": example["target_audio"]}


def main() -> None:
    inputs = load_paper_inputs(require_training=True)
    frame = pd.read_pickle(checked_input(inputs["config"]["dataset"]))
    audio = pd.read_pickle(checked_input(inputs["config"]["audio_features"]))
    if len(frame) != 11200 or len(audio) != 11200:
        raise ValueError("Expected the development-only 11200-row snapshots")
    helper = json.loads((ROOT / "config" / "msarcuse_episode_fold0.json").read_text())
    helper["pilot"] = copy.deepcopy(helper["evaluation"])
    records = eligible_records(frame, audio, helper)
    unknown = {r["episode_id"] for r in records} - set(inputs["episodes"])
    if unknown:
        raise ValueError("Final episode map must cover all eligible development episodes")
    record_map = {r["source_index"]: r for r in records}
    for target in inputs["targets"] + [t for rows in inputs["training"].values() for t in rows]:
        source = record_map[target["source_index"]]
        for key in ("episode_id", "dialogue_id", "target_label", "context_turns", "next_speaker"):
            if target[key] != source[key]:
                raise ValueError(f"Final target differs from source: {target['sample_id']} {key}")
        if not (source["target_audio"] == target["target_audio"]).all():
            raise ValueError(f"Audio features differ: {target['sample_id']}")
        for candidate in target["candidates"]:
            row = frame.iloc[candidate["source_index"]]
            if (str(row["Episode_label"]) != candidate["source_dialogue_id"]
                    or str(row["text"]).strip() != candidate["text"].strip()):
                raise ValueError(f"Candidate/source mismatch: {target['sample_id']}")
            if candidate["is_gold"] and (candidate["source_index"] != target["source_index"] + 1
                                         or candidate["text"].strip() != source["response"]):
                raise ValueError(f"Gold is not the observed next reply: {target['sample_id']}")
    recognition_path = ARTIFACT_ROOT / "data" / "recognition_records.jsonl"
    recognition = [
        {"source_index": r["source_index"], "dialogue_id": r["dialogue_id"],
         "episode_id": r["episode_id"], "evaluation_fold": inputs["episodes"][r["episode_id"]],
         "target_label": r["target_label"], "text": r["detector_text"],
         "target_audio": r["target_audio"].astype(float).tolist()}
        for r in records]
    write_jsonl(recognition_path, recognition)
    summaries = []
    for fold in range(4):
        train_path = ARTIFACT_ROOT / "data" / f"fold-{fold}" / "selector_training_targets.jsonl"
        if not train_path.is_file():
            raise FileNotFoundError("Run materialize_external_baseline_training.py first")
        training = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
        if training != inputs["training"][fold]:
            raise ValueError("Materialized training targets differ from final inputs")
        evaluation = [t for t in inputs["targets"] if t["fold"] == fold]
        path = ARTIFACT_ROOT / "data" / f"fold-{fold}" / "response_tasks.jsonl"
        rows = list(candidate_rows(training, "training", fold))
        rows.extend(candidate_rows(evaluation, "evaluation", fold))
        write_jsonl(path, rows)
        summaries.append({"fold": fold, "path": str(path), "sha256": sha256(path),
                          "rows": len(rows), "evaluation_targets": len(evaluation)})
    atomic_json(ARTIFACT_ROOT / "manifests" / "method_neutral_tasks.json",
                {"schema_version": 2, "status": "materialized",
                 "input_config_sha256": inputs["input_config_sha256"],
                 "recognition_records": {"path": str(recognition_path),
                                         "sha256": sha256(recognition_path)},
                 "response_tasks": summaries})
    print(json.dumps({"status": "materialized", "folds": len(summaries)}))


if __name__ == "__main__":
    main()
