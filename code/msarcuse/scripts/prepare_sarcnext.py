from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / 'config' / 'sarcnext.json'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarcnext_common import atomic_json, dialogue_bucket, episode_id_from_dialogue, load_json, rendered_prefix, sha256_file, split_bucket, stable_int, write_jsonl

def validate_config(config: dict) -> None:
    if config['experiment_family'] != 'SARCNEXT-BRIDGE-1':
        raise ValueError('Wrong experiment family')
    data = config['data']
    if data['permitted_development_end_exclusive'] != 11200:
        raise ValueError('The permitted development boundary must remain 11200')
    if data['sealed_confirmation_range'] != [11200, 14000] or data['sealed_official_test_range'] != [14000, 15576]:
        raise ValueError('Frozen confirmation/test ranges changed')
    policy = config['evaluation_policy']
    forbidden = ('new_human_annotation', 'llm_judge', 'human_evaluation_claim', 'sealed_confirmation_for_selection', 'sealed_official_test_for_selection', 'paid_services')
    if any((policy[key] for key in forbidden)):
        raise ValueError('SarcNext scope boundary changed')
    if config['pilot']['conditions'] != ['direct', 'self_multimodal', 'oracle']:
        raise ValueError('Frozen condition order changed')

def clean_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None

def context_turns(frame: pd.DataFrame, index: int, maximum: int) -> list[dict]:
    dialogue = str(frame.at[index, 'Episode_label'])
    start = index
    while start > 0 and str(frame.at[start - 1, 'Episode_label']) == dialogue:
        start -= 1
    start = max(start, index - maximum + 1)
    turns = []
    for row_index in range(start, index + 1):
        text = clean_text(frame.at[row_index, 'text'])
        if text is not None:
            turns.append({'speaker': str(frame.at[row_index, 'Speaker']), 'text': text})
    return turns

def eligible_records(frame: pd.DataFrame, audio: pd.Series, config: dict) -> list[dict]:
    pilot = config['pilot']
    data = config['data']
    split_unit = pilot.get('development_split_unit', data.get('development_split_unit', 'dialogue'))
    records = []
    for index in range(len(frame) - 1):
        dialogue = str(frame.at[index, 'Episode_label'])
        if dialogue != str(frame.at[index + 1, 'Episode_label']):
            continue
        target = clean_text(frame.at[index, 'text'])
        response = clean_text(frame.at[index + 1, 'text'])
        if target is None or response is None:
            continue
        vector = np.asarray(audio.iloc[index], dtype=np.float32)
        if vector.shape != (128,) or not np.isfinite(vector).all():
            raise ValueError(f'Invalid target audio vector at permitted source index {index}')
        turns = context_turns(frame, index, pilot['context_turns'])
        if not turns:
            continue
        records.append({'source_index': index, 'dialogue_id': dialogue, 'episode_id': episode_id_from_dialogue(dialogue), 'target_label': int(frame.at[index, 'Sarcasm']), 'next_speaker': str(frame.at[index + 1, 'Speaker']), 'response': response, 'context_turns': turns, 'detector_text': '\n'.join((f"{turn['speaker']}: {turn['text']}" for turn in turns)), 'target_audio': vector, 'is_pilot_development': split_bucket(dialogue, data['split_salt'], pilot['development_hash_modulus'], split_unit) == pilot['development_hash_bucket']})
    return records

def detector_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {'samples': int(len(labels)), 'label_counts': {str(key): int(value) for key, value in sorted(Counter(labels.tolist()).items())}, 'accuracy': float(accuracy_score(labels, predictions)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)), 'f1': float(f1_score(labels, predictions, zero_division=0)), 'confusion_matrix_labels_0_1': confusion_matrix(labels, predictions, labels=[0, 1]).tolist()}

def fit_detectors(records: list[dict], config: dict) -> tuple[dict[int, dict], dict]:
    train = [record for record in records if not record['is_pilot_development']]
    development = [record for record in records if record['is_pilot_development']]
    detector = config['detector']
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=tuple(detector['ngram_range']), max_features=detector['max_features'], min_df=detector['min_df'], sublinear_tf=True, dtype=np.float32)
    train_text = vectorizer.fit_transform([record['detector_text'] for record in train])
    dev_text = vectorizer.transform([record['detector_text'] for record in development])
    train_audio = np.stack([record['target_audio'] for record in train])
    dev_audio = np.stack([record['target_audio'] for record in development])
    scaler = StandardScaler()
    train_audio_scaled = csr_matrix(scaler.fit_transform(train_audio), dtype=np.float32)
    dev_audio_scaled = csr_matrix(scaler.transform(dev_audio), dtype=np.float32)
    train_labels = np.asarray([record['target_label'] for record in train], dtype=np.int64)
    dev_labels = np.asarray([record['target_label'] for record in development], dtype=np.int64)
    common = {'C': detector['c'], 'class_weight': 'balanced', 'solver': 'liblinear', 'max_iter': detector['max_iter'], 'random_state': config['pilot']['seed']}
    text_model = LogisticRegression(**common).fit(train_text, train_labels)
    multimodal_model = LogisticRegression(**common).fit(hstack([train_text, train_audio_scaled], format='csr'), train_labels)
    text_probability = text_model.predict_proba(dev_text)[:, 1]
    multimodal_probability = multimodal_model.predict_proba(hstack([dev_text, dev_audio_scaled], format='csr'))[:, 1]
    threshold = float(detector['threshold'])
    predictions = {record['source_index']: {'text_probability': float(text_value), 'text_label': int(text_value >= threshold), 'multimodal_probability': float(multimodal_value), 'multimodal_label': int(multimodal_value >= threshold)} for record, text_value, multimodal_value in zip(development, text_probability, multimodal_probability, strict=True)}
    metrics = {'schema_version': 1, 'training_samples': len(train), 'development_samples': len(development), 'training_dialogues': len({record['dialogue_id'] for record in train}), 'development_dialogues': len({record['dialogue_id'] for record in development}), 'text_feature_count': len(vectorizer.vocabulary_), 'audio_shape': [128], 'threshold': threshold, 'text_only': detector_metrics(dev_labels, text_probability, threshold), 'text_plus_audio': detector_metrics(dev_labels, multimodal_probability, threshold), 'new_human_annotation': False, 'llm_judge': False}
    return (predictions, metrics)

def ranked_pool(pool: list[dict], target: dict, salt: str) -> list[dict]:
    return sorted(pool, key=lambda row: (abs(len(row['response']) - len(target['response'])), stable_int(f"{target['source_index']}|{row['source_index']}", salt)))

def candidate_set(target: dict, development: list[dict], salt: str) -> list[dict] | None:
    pool = [row for row in development if row['episode_id'] == target['episode_id'] and row['next_speaker'] == target['next_speaker'] and (row['dialogue_id'] != target['dialogue_id']) and (row['response'] != target['response'])]
    same = ranked_pool([row for row in pool if row['target_label'] == target['target_label']], target, salt)
    opposite = ranked_pool([row for row in pool if row['target_label'] != target['target_label']], target, salt)
    desired_third = target['target_label'] if stable_int(f"{target['dialogue_id']}|{target['target_label']}|third", salt) % 2 == 0 else 1 - target['target_label']
    desired = same if desired_third == target['target_label'] else opposite
    if not same or not opposite or len({row['response'] for row in desired}) < 2:
        return None
    chosen = [same[0], opposite[0]]
    third = next((row for row in desired if row['response'] not in {entry['response'] for entry in chosen}), None)
    if third is None:
        return None
    candidates = [{'candidate_id': 'gold', 'text': target['response'], 'is_gold': True, 'source_index': target['source_index'] + 1, 'source_dialogue_id': target['dialogue_id'], 'source_target_label': target['target_label']}]
    candidates.extend(({'candidate_id': f'distractor-{position}', 'text': row['response'], 'is_gold': False, 'source_index': row['source_index'] + 1, 'source_dialogue_id': row['dialogue_id'], 'source_target_label': row['target_label']} for position, row in enumerate([same[0], opposite[0], third], start=1)))
    if len({candidate['text'] for candidate in candidates}) != 4:
        return None
    return sorted(candidates, key=lambda candidate: stable_int(f"{target['source_index']}|{candidate['candidate_id']}|{candidate['text']}", salt))

def fits_token_budget(tokenizer, example: dict, config: dict) -> bool:
    maximum = config['pilot']['max_input_tokens']
    for condition in config['pilot']['conditions']:
        prefix = rendered_prefix(tokenizer, example, condition)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        for candidate in example['candidates']:
            whole = tokenizer.encode(prefix + candidate['text'].strip(), add_special_tokens=False)
            if whole[:len(prefix_ids)] != prefix_ids or len(whole) <= len(prefix_ids) or len(whole) > maximum:
                return False
    return True

def make_example(target: dict, candidates: list[dict], predictions: dict[int, dict]) -> dict:
    prediction = predictions[target['source_index']]
    return {'sample_id': f"masac-{target['source_index']:05d}", 'dialogue_pair_id': target['dialogue_id'], 'source_index': target['source_index'], 'dialogue_id': target['dialogue_id'], 'episode_id': target['episode_id'], 'target_label': target['target_label'], 'context_turns': target['context_turns'], 'target_audio': target['target_audio'].astype(float).tolist(), 'next_speaker': target['next_speaker'], 'candidates': candidates, 'self_predicted_label': prediction['multimodal_label'], 'self_probability': prediction['multimodal_probability'], 'text_predicted_label': prediction['text_label'], 'text_probability': prediction['text_probability']}

def select_examples(records: list[dict], predictions: dict[int, dict], tokenizer, config: dict) -> tuple[list[dict], dict]:
    development = [record for record in records if record['is_pilot_development']]
    by_dialogue_label: dict[tuple[str, int], list[dict]] = {}
    for record in development:
        by_dialogue_label.setdefault((record['dialogue_id'], record['target_label']), []).append(record)
    dialogues = sorted({dialogue for dialogue, label in by_dialogue_label if (dialogue, 1 - label) in by_dialogue_label}, key=lambda value: stable_int(value, config['data']['split_salt']))
    selected = []
    rejected = Counter()
    for dialogue in dialogues:
        pair = []
        for label in config['pilot']['target_labels_per_dialogue']:
            targets = sorted(by_dialogue_label[dialogue, label], key=lambda row: stable_int(str(row['source_index']), config['data']['split_salt']))
            chosen = None
            for target in targets:
                candidates = candidate_set(target, development, config['data']['split_salt'])
                if candidates is None:
                    rejected['candidate_constraints'] += 1
                    continue
                example = make_example(target, candidates, predictions)
                while len(example['context_turns']) > 1 and (not fits_token_budget(tokenizer, example, config)):
                    example['context_turns'] = example['context_turns'][1:]
                if not fits_token_budget(tokenizer, example, config):
                    rejected['token_budget'] += 1
                    continue
                chosen = example
                break
            if chosen is None:
                pair = []
                break
            pair.append(chosen)
        if len(pair) == 2:
            selected.extend(pair)
        if len(selected) == 2 * config['pilot']['dialogue_pairs']:
            break
    if len(selected) != 2 * config['pilot']['dialogue_pairs']:
        raise ValueError(f'Only {len(selected) // 2} complete dialogue pairs satisfy the frozen protocol')
    return (selected, dict(rejected))

def prepare(config_path: Path) -> dict:
    config = load_json(config_path)
    validate_config(config)
    data = config['data']
    dataset_path = Path(data['dataset_path'])
    audio_path = Path(data['audio_features_path'])
    if sha256_file(dataset_path).upper() != data['dataset_sha256'] or sha256_file(audio_path).upper() != data['audio_features_sha256']:
        raise ValueError('Official MaSaC source hash mismatch')
    whole_frame = pd.read_pickle(dataset_path)
    whole_audio = pd.read_pickle(audio_path)
    if whole_frame.shape != (data['official_total_rows'], 7) or len(whole_audio) != data['official_total_rows']:
        raise ValueError('Official MaSaC container shape mismatch')
    expected_columns = ['Speaker', 'text', 'Context', 'Sarcasm', 'Humour', 'Episode_label', 'Audio_Filename']
    if list(whole_frame.columns) != expected_columns:
        raise ValueError('Official MaSaC schema mismatch')
    permitted_end = data['permitted_development_end_exclusive']
    frame = whole_frame.iloc[:permitted_end].copy()
    audio = whole_audio.iloc[:permitted_end].copy()
    del whole_frame, whole_audio
    records = eligible_records(frame, audio, config)
    predictions, detector = fit_detectors(records, config)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['local_path'], local_files_only=True, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples, rejected = select_examples(records, predictions, tokenizer, config)
    prepared_path = Path(data['prepared_path'])
    detector_path = Path(data['detector_metrics_path'])
    manifest_path = Path(data['manifest_path'])
    atomic_json(detector_path, detector)
    write_jsonl(prepared_path, examples)
    max_index = max([example['source_index'] for example in examples] + [candidate['source_index'] for example in examples for candidate in example['candidates']])
    manifest = {'schema_version': 1, 'experiment_family': config['experiment_family'], 'repository': data['repository'], 'revision': data['revision'], 'license_status': data['license_status'], 'dataset_sha256': data['dataset_sha256'].lower(), 'audio_features_sha256': data['audio_features_sha256'].lower(), 'config_sha256': sha256_file(config_path), 'prepared_path': str(prepared_path), 'prepared_sha256': sha256_file(prepared_path), 'detector_metrics_sha256': sha256_file(detector_path), 'permitted_source_range': [0, permitted_end], 'max_serialized_source_index': max_index, 'eligible_permitted_targets': len(records), 'eligible_training_targets': sum((not record['is_pilot_development'] for record in records)), 'eligible_pilot_development_targets': sum((record['is_pilot_development'] for record in records)), 'dialogue_pairs': config['pilot']['dialogue_pairs'], 'response_targets': len(examples), 'target_label_counts': {str(key): value for key, value in sorted(Counter((example['target_label'] for example in examples)).items())}, 'candidate_count_per_target': config['pilot']['candidate_count'], 'condition_count': len(config['pilot']['conditions']), 'score_tasks': len(examples) * config['pilot']['candidate_count'] * len(config['pilot']['conditions']), 'selected_dialogue_ids_sha256': __import__('hashlib').sha256('\n'.join(sorted({example['dialogue_id'] for example in examples})).encode('utf-8')).hexdigest(), 'selection_rejections': rejected, 'source_container_contains_sealed_rows': True, 'container_deserialized_only_to_apply_frozen_positional_seal': True, 'sealed_row_values_logged_or_used': False, 'confirmation_accessed': False, 'official_test_accessed': False, 'new_human_annotation': False, 'llm_judge': False, 'raw_media_downloaded': False, 'claim_scope': 'observed_next_turn_retrieval_not_response_appropriateness'}
    if max_index >= permitted_end:
        raise ValueError('A serialized source index crossed the frozen development boundary')
    atomic_json(manifest_path, manifest)
    return manifest

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config.resolve()), ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
