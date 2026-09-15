from __future__ import annotations
import argparse
import copy
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / 'config' / 'sarcnext_learned.json'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_sarcnext import candidate_set, eligible_records, fits_token_budget
from sarcnext_common import atomic_json, conditional_positions, load_json, read_jsonl, rendered_prefix, sha256_file, split_identity, stable_int, write_jsonl

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def configure_runtime(config: dict) -> None:
    root = Path(config['artifact_root'])
    temporary = root / 'tmp'
    cache = temporary / 'runtime-cache'
    values = {'TMP': str(temporary), 'TEMP': str(temporary), 'HF_HOME': str(cache / 'huggingface'), 'HUGGINGFACE_HUB_CACHE': str(cache / 'huggingface' / 'hub'), 'TORCH_HOME': str(cache / 'torch'), 'TRITON_CACHE_DIR': str(cache / 'triton'), 'CUDA_CACHE_PATH': str(cache / 'cuda'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false', 'CUDA_MODULE_LOADING': 'LAZY'}
    for name, value in values.items():
        os.environ[name] = value
    for value in values.values():
        if value.startswith(str(root)):
            Path(value).mkdir(parents=True, exist_ok=True)

def validate_scope(config: dict, config_path: Path, verify_hashes: bool) -> tuple[list[dict], dict]:
    if config['experiment_family'] != 'SARCNEXT-BRIDGE-1' or config['configuration_id'] != 'learned-selector-v1':
        raise ValueError('Wrong learned-selector configuration')
    policy = config['evaluation_policy']
    for key in ('new_human_annotation', 'llm_judge', 'human_evaluation_claim', 'sealed_confirmation_for_selection', 'sealed_official_test_for_selection', 'paid_services'):
        if policy[key]:
            raise ValueError(f'Forbidden policy enabled: {key}')
    if shutil.disk_usage(config['artifact_root']).free < 10 * 1024 ** 3:
        raise RuntimeError('E: has less than 10 GiB free')
    data = config['data']
    data_manifest = load_json(Path(data['manifest_path']))
    examples = read_jsonl(Path(data['prepared_path']))
    if data_manifest['revision'] != data['revision'] or data_manifest['prepared_sha256'] != sha256_file(Path(data['prepared_path'])):
        raise ValueError('Prepared-data revision or hash mismatch')
    if data_manifest['max_serialized_source_index'] >= data['permitted_development_end_exclusive']:
        raise ValueError('Prepared data cross the frozen development boundary')
    for flag in ('confirmation_accessed', 'official_test_accessed', 'new_human_annotation', 'llm_judge'):
        if data_manifest.get(flag) is not False:
            raise ValueError(f'Prepared-data scope flag failed: {flag}')
    model_manifest = load_json(Path(config['model']['manifest_path']))
    if model_manifest['revision'] != config['model']['revision'] or model_manifest['total_bytes'] != config['model']['expected_download_bytes']:
        raise ValueError('Model manifest mismatch')
    model_root = Path(config['model']['local_path'])
    for entry in model_manifest['files']:
        path = model_root / entry['name']
        if not path.is_file() or path.stat().st_size != entry['size_bytes']:
            raise ValueError(f'Missing or wrong-sized model file: {path}')
        if verify_hashes and sha256_file(path) != entry['sha256']:
            raise ValueError(f'Model hash mismatch: {path}')
    if sha256_file(Path(data['dataset_path'])).upper() != data['dataset_sha256'] or sha256_file(Path(data['audio_features_path'])).upper() != data['audio_features_sha256']:
        raise ValueError('Official MaSaC source hash mismatch')
    return (examples, data_manifest)

class Encoder:

    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        self.torch = torch
        self.device = torch.device('cuda:0')
        torch.manual_seed(config['pilot']['seed'])
        torch.cuda.manual_seed_all(config['pilot']['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['local_path'], local_files_only=True, trust_remote_code=False, use_fast=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        causal = AutoModelForCausalLM.from_pretrained(config['model']['local_path'], local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='sdpa').to(self.device)
        causal.eval()
        causal.config.use_cache = False
        self.model = causal.model
        self.hidden_size = int(causal.config.hidden_size)

def runtime_versions(torch) -> dict:
    import importlib.metadata as metadata
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy', 'scipy', 'pandas', 'scikit-learn'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'created_at': utc_now(), 'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages': packages, 'cuda_available': torch.cuda.is_available(), 'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'gpu_capability': torch.cuda.get_device_capability(0)}

def make_training_example(target: dict, candidates: list[dict]) -> dict:
    return {'sample_id': f"train-{target['source_index']:05d}", 'dialogue_pair_id': target['dialogue_id'], 'source_index': target['source_index'], 'dialogue_id': target['dialogue_id'], 'episode_id': target['episode_id'], 'target_label': target['target_label'], 'context_turns': copy.deepcopy(target['context_turns']), 'target_audio': target['target_audio'].astype(float).tolist(), 'next_speaker': target['next_speaker'], 'candidates': candidates, 'self_predicted_label': target['target_label']}

def select_training_examples(records: list[dict], encoder: Encoder, config: dict, targets_per_label: int, evaluation_examples: list[dict] | None=None) -> list[dict]:
    training = [record for record in records if not record['is_pilot_development']]
    validation_unit = config['selector'].get('head_validation_split_unit', 'dialogue')

    def is_head_validation(record: dict) -> bool:
        return stable_int(split_identity(record['dialogue_id'], validation_unit), 'sarcnext-head-v1') % config['selector']['head_validation_modulus'] == config['selector']['head_validation_bucket']
    selected = []
    for label in (0, 1):
        candidates = sorted([record for record in training if record['target_label'] == label], key=lambda record: stable_int(str(record['source_index']), 'sarcnext-selector-v1'))
        label_examples = []
        for target in candidates:
            target_partition = is_head_validation(target)
            partition_pool = [record for record in training if is_head_validation(record) == target_partition]
            responses = candidate_set(target, partition_pool, config['data']['split_salt'])
            if responses is None:
                continue
            example = make_training_example(target, responses)
            while len(example['context_turns']) > 1 and (not fits_token_budget(encoder.tokenizer, example, config)):
                example['context_turns'] = example['context_turns'][1:]
            if not fits_token_budget(encoder.tokenizer, example, config):
                continue
            label_examples.append(example)
            if len(label_examples) == targets_per_label:
                break
        if len(label_examples) != targets_per_label:
            raise ValueError(f'Only {len(label_examples)} selector-training targets available for label {label}')
        selected.extend(label_examples)
    if any((example['source_index'] >= config['data']['permitted_development_end_exclusive'] for example in selected)):
        raise ValueError('Selector training crossed the permitted development boundary')
    split_unit = config['data'].get('development_split_unit', 'dialogue')
    training_units = {split_identity(example['dialogue_id'], split_unit) for example in selected}
    frozen_evaluation = evaluation_examples if evaluation_examples is not None else read_jsonl(Path(config['data']['prepared_path']))
    evaluation_units = {split_identity(example['dialogue_id'], split_unit) for example in frozen_evaluation}
    if training_units & evaluation_units:
        raise ValueError(f'Selector training and evaluation {split_unit} sets overlap')
    return selected

def status_vector(example: dict, condition: str) -> list[float]:
    if condition == 'direct':
        return [1.0, 0.0, 0.0]
    label = example['self_predicted_label'] if condition == 'self_multimodal' else example['target_label']
    return [0.0, float(label == 0), float(label == 1)]

def feature_tasks(examples: list[dict], conditions: list[str]) -> list[dict]:
    rows = []
    for example in examples:
        for condition in conditions:
            for candidate in example['candidates']:
                rows.append({'sample_id': example['sample_id'], 'dialogue_pair_id': example['dialogue_pair_id'], 'source_index': example['source_index'], 'target_label': example['target_label'], 'condition': condition, 'candidate_id': candidate['candidate_id'], 'candidate_text': candidate['text'], 'is_gold': bool(candidate['is_gold']), 'audio': np.asarray(example['target_audio'], dtype=np.float32), 'status': np.asarray(status_vector(example, condition), dtype=np.float32), 'example': example})
    return rows

def extract_features(encoder: Encoder, tasks: list[dict], config: dict, heartbeat_path: Path, stage: str, progress_offset: int, progress_total: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    qwen_features = []
    audio_features = []
    status_features = []
    labels = []
    metadata = []
    started = time.monotonic()
    batch_size = config['selector']['feature_batch_size']
    for offset in range(0, len(tasks), batch_size):
        batch = tasks[offset:offset + batch_size]
        prefixes = [rendered_prefix(encoder.tokenizer, task['example'], task['condition']) for task in batch]
        encoded, positions, _, _ = conditional_positions(encoder.tokenizer, prefixes, [task['candidate_text'] for task in batch], config['pilot']['max_input_tokens'])
        inputs = {name: tensor.to(encoder.device) for name, tensor in encoded.items()}
        with encoder.torch.inference_mode():
            output = encoder.model(**inputs, use_cache=False, return_dict=True)
            hidden = output.last_hidden_state.float()
            pooled = encoder.torch.stack([hidden[row, candidate_positions].mean(dim=0) for row, candidate_positions in enumerate(positions)])
            pooled = encoder.torch.nn.functional.normalize(pooled, p=2, dim=1)
        qwen_features.append(pooled.cpu().numpy())
        audio_features.extend((task['audio'] for task in batch))
        status_features.extend((task['status'] for task in batch))
        labels.extend((float(task['is_gold']) for task in batch))
        metadata.extend(({'sample_id': task['sample_id'], 'dialogue_pair_id': task['dialogue_pair_id'], 'source_index': task['source_index'], 'target_label': task['target_label'], 'condition': task['condition'], 'candidate_id': task['candidate_id'], 'is_gold': task['is_gold']} for task in batch))
        local_done = offset + len(batch)
        overall_done = progress_offset + local_done
        elapsed = max(time.monotonic() - started, 1e-09)
        throughput = local_done / elapsed
        eta = int((len(tasks) - local_done) / throughput) if throughput > 0 else None
        atomic_json(heartbeat_path, {'stage': stage, 'completed_units': overall_done, 'total_units': progress_total, 'stage_completed_units': local_done, 'stage_total_units': len(tasks), 'recent_throughput_pairs_per_second': throughput, 'stage_eta_seconds': eta, 'last_progress_at': utc_now(), 'gpu_allocated_bytes': int(encoder.torch.cuda.memory_allocated()), 'gpu_reserved_bytes': int(encoder.torch.cuda.memory_reserved()), 'failure_counts': {}})
        if local_done == len(tasks) or local_done % 400 == 0:
            print(f'selector_feature_progress stage={stage} completed={local_done}/{len(tasks)} throughput={throughput:.2f}_pairs_s eta_seconds={eta}', flush=True)
    return (np.concatenate(qwen_features).astype(np.float32), np.stack(audio_features).astype(np.float32), np.stack(status_features).astype(np.float32), np.asarray(labels, dtype=np.float32), metadata)

class SelectorHead:

    def __init__(self, torch, input_size: int, hidden_size: int, dropout: float):
        self.module = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden_size, 1))

def retrieval_accuracy(logits: np.ndarray, labels: np.ndarray, metadata: list[dict]) -> float:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row['sample_id'], row['condition']].append(index)
    correct = 0
    for indices in groups.values():
        best = min(indices, key=lambda index: (-float(logits[index]), metadata[index]['candidate_id']))
        correct += int(labels[best] == 1)
    return correct / len(groups)

def train_head(encoder: Encoder, qwen: np.ndarray, audio: np.ndarray, status: np.ndarray, labels: np.ndarray, metadata: list[dict], config: dict, run_dir: Path) -> tuple[object, np.ndarray, np.ndarray, dict]:
    torch = encoder.torch
    audio_mean = audio.mean(axis=0)
    audio_std = audio.std(axis=0)
    audio_std[audio_std < 1e-06] = 1.0
    features = np.concatenate([qwen, (audio - audio_mean) / audio_std, status], axis=1).astype(np.float32)
    val_mask = np.asarray([stable_int(row['dialogue_pair_id'], 'sarcnext-head-v1') % config['selector']['head_validation_modulus'] == config['selector']['head_validation_bucket'] for row in metadata], dtype=bool)
    if not val_mask.any() or val_mask.all():
        raise ValueError('Selector head train/validation split is degenerate')
    device = encoder.device
    feature_tensor = torch.from_numpy(features)
    label_tensor = torch.from_numpy(labels)
    train_indices = np.flatnonzero(~val_mask)
    val_indices = np.flatnonzero(val_mask)
    generator = torch.Generator().manual_seed(config['pilot']['seed'])
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(feature_tensor[train_indices], label_tensor[train_indices]), batch_size=config['selector']['batch_size'], shuffle=True, generator=generator)
    wrapper = SelectorHead(torch, features.shape[1], config['selector']['hidden_size'], config['selector']['dropout'])
    head = wrapper.module.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['selector']['learning_rate'], weight_decay=config['selector']['weight_decay'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config['selector']['positive_weight']], device=device))
    val_x = feature_tensor[val_indices].to(device)
    val_y = label_tensor[val_indices].to(device)
    history = []
    best_state = None
    best_loss = float('inf')
    best_epoch = None
    for epoch in range(1, config['selector']['epochs'] + 1):
        head.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(batch_x).squeeze(1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_x)
            train_count += len(batch_x)
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_x).squeeze(1)
            val_loss = float(criterion(val_logits, val_y).item())
        entry = {'epoch': epoch, 'train_bce': train_loss / train_count, 'validation_bce': val_loss, 'validation_retrieval_accuracy': retrieval_accuracy(val_logits.cpu().numpy(), labels[val_indices], [metadata[index] for index in val_indices])}
        history.append(entry)
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        print(f"selector_head epoch={epoch}/{config['selector']['epochs']} train_bce={entry['train_bce']:.5f} val_bce={val_loss:.5f} val_retrieval={entry['validation_retrieval_accuracy']:.4f}", flush=True)
    if best_state is None:
        raise RuntimeError('Selector head did not produce a finite checkpoint')
    head.load_state_dict(best_state)
    head.eval()
    checkpoint = {'state_dict': best_state, 'input_size': features.shape[1], 'audio_mean': torch.from_numpy(audio_mean), 'audio_std': torch.from_numpy(audio_std), 'best_epoch': best_epoch, 'best_validation_bce': best_loss, 'config_sha256': sha256_file(DEFAULT_CONFIG)}
    torch.save(checkpoint, run_dir / 'selector_head.pt')
    metrics = {'training_pair_rows': int((~val_mask).sum()), 'validation_pair_rows': int(val_mask.sum()), 'training_dialogues': len({metadata[index]['dialogue_pair_id'] for index in train_indices}), 'validation_dialogues': len({metadata[index]['dialogue_pair_id'] for index in val_indices}), 'feature_size': features.shape[1], 'best_epoch': best_epoch, 'best_validation_bce': best_loss, 'best_validation_retrieval_accuracy': history[best_epoch - 1]['validation_retrieval_accuracy'], 'history': history}
    atomic_json(run_dir / 'selector_training_metrics.json', metrics)
    return (head, audio_mean, audio_std, metrics)

def score_eval(encoder: Encoder, head, qwen: np.ndarray, audio: np.ndarray, status: np.ndarray, metadata: list[dict], audio_mean: np.ndarray, audio_std: np.ndarray) -> list[dict]:
    features = np.concatenate([qwen, (audio - audio_mean) / audio_std, status], axis=1).astype(np.float32)
    with encoder.torch.inference_mode():
        logits = head(encoder.torch.from_numpy(features).to(encoder.device)).squeeze(1).cpu().numpy()
    return [{**row, 'selector_logit': float(logit)} for row, logit in zip(metadata, logits, strict=True)]

def run(config_path: Path, run_id: str, limit_dialogue_pairs: int | None, train_targets_per_label: int | None, verify_hashes: bool) -> dict:
    config = load_json(config_path)
    configure_runtime(config)
    examples, data_manifest = validate_scope(config, config_path, verify_hashes)
    if limit_dialogue_pairs is not None:
        dialogue_order = []
        for example in examples:
            if example['dialogue_pair_id'] not in dialogue_order:
                dialogue_order.append(example['dialogue_pair_id'])
        allowed = set(dialogue_order[:limit_dialogue_pairs])
        examples = [example for example in examples if example['dialogue_pair_id'] in allowed]
    expected_dialogues = limit_dialogue_pairs or config['pilot']['dialogue_pairs']
    if len(examples) != 2 * expected_dialogues:
        raise ValueError('Pilot dialogue-pair completeness mismatch')
    targets_per_label = train_targets_per_label or config['selector']['training_targets_per_label']
    run_dir = Path(config['artifact_root']) / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    completion_path = run_dir / 'completion.json'
    if completion_path.exists():
        completion = load_json(completion_path)
        if completion.get('status') == 'completed':
            return completion
    encoder = Encoder(config)
    atomic_json(run_dir / 'runtime.json', runtime_versions(encoder.torch))
    whole_frame = pd.read_pickle(Path(config['data']['dataset_path']))
    whole_audio = pd.read_pickle(Path(config['data']['audio_features_path']))
    permitted_end = config['data']['permitted_development_end_exclusive']
    frame = whole_frame.iloc[:permitted_end].copy()
    audio = whole_audio.iloc[:permitted_end].copy()
    del whole_frame, whole_audio
    records = eligible_records(frame, audio, config)
    training_examples = select_training_examples(records, encoder, config, targets_per_label)
    train_tasks = feature_tasks(training_examples, config['selector']['training_conditions'])
    eval_tasks = feature_tasks(examples, config['pilot']['conditions'])
    total_feature_tasks = len(train_tasks) + len(eval_tasks)
    run_manifest = {'schema_version': 1, 'run_id': run_id, 'experiment_family': config['experiment_family'], 'configuration_id': config['configuration_id'], 'config_sha256': sha256_file(config_path), 'prepared_sha256': data_manifest['prepared_sha256'], 'data_revision': config['data']['revision'], 'model_revision': config['model']['revision'], 'selector_training_targets_per_label': targets_per_label, 'selector_training_targets': len(training_examples), 'selector_training_feature_tasks': len(train_tasks), 'dialogue_pairs': expected_dialogues, 'response_targets': len(examples), 'candidate_count': config['pilot']['candidate_count'], 'conditions': config['pilot']['conditions'], 'total_score_tasks': len(eval_tasks), 'score_semantics': 'shared_trained_selector_logit', 'same_selector_head_across_conditions': True, 'training_and_pilot_dialogue_overlap': False, 'development_only': True, 'max_source_index': max(max((example['source_index'] for example in examples)), max((example['source_index'] for example in training_examples))), 'confirmation_accessed': False, 'official_test_accessed': False, 'new_human_annotation': False, 'llm_judge': False}
    atomic_json(run_dir / 'run_manifest.json', run_manifest)
    heartbeat_path = run_dir / 'heartbeat.json'
    train_qwen, train_audio, train_status, train_labels, train_meta = extract_features(encoder, train_tasks, config, heartbeat_path, 'selector_training_feature_extraction', 0, total_feature_tasks)
    np.savez_compressed(run_dir / 'selector_training_features.npz', qwen=train_qwen.astype(np.float16), audio=train_audio.astype(np.float32), status=train_status.astype(np.float32), labels=train_labels.astype(np.float32))
    write_jsonl(run_dir / 'selector_training_feature_index.jsonl', train_meta)
    head, audio_mean, audio_std, selector_metrics = train_head(encoder, train_qwen, train_audio, train_status, train_labels, train_meta, config, run_dir)
    eval_qwen, eval_audio, eval_status, _, eval_meta = extract_features(encoder, eval_tasks, config, heartbeat_path, 'pilot_feature_extraction', len(train_tasks), total_feature_tasks)
    score_rows = score_eval(encoder, head, eval_qwen, eval_audio, eval_status, eval_meta, audio_mean, audio_std)
    scores_path = run_dir / 'candidate_scores.jsonl'
    write_jsonl(scores_path, score_rows)
    if len(score_rows) != len(eval_tasks) or len({(row['sample_id'], row['condition'], row['candidate_id']) for row in score_rows}) != len(score_rows):
        raise ValueError('Learned selector score matrix is incomplete or duplicated')
    completion = {'schema_version': 1, 'run_id': run_id, 'status': 'completed', 'completed_at': utc_now(), 'score_rows': len(score_rows), 'unique_score_keys': len(score_rows), 'scores_sha256': sha256_file(scores_path), 'selector_training_metrics_sha256': sha256_file(run_dir / 'selector_training_metrics.json'), 'selector_head_sha256': sha256_file(run_dir / 'selector_head.pt'), 'all_scores_finite': all((math.isfinite(row['selector_logit']) for row in score_rows)), 'confirmation_accessed': False, 'official_test_accessed': False, 'new_human_annotation': False, 'llm_judge': False}
    atomic_json(completion_path, completion)
    return completion

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--limit-dialogue-pairs', type=int)
    parser.add_argument('--train-targets-per-label', type=int)
    parser.add_argument('--verify-model-hashes', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve(), args.run_id, args.limit_dialogue_pairs, args.train_targets_per_label, args.verify_model_hashes), ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
