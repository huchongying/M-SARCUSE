from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
ROOT = Path(__file__).resolve().parents[1]
from paper_protocol import ARTIFACT_ROOT, ROOT, require_preflight
ENCODERS = {'mbert': ARTIFACT_ROOT / 'models' / 'bert-base-multilingual-cased-3f076fdb1ab6', 'bart': ARTIFACT_ROOT / 'models' / 'bart-base-aadd2ab0ae0c', 'roberta': ARTIFACT_ROOT / 'models' / 'roberta-base-e2da8e2f811d'}

def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def encode_batches(model, tokenizer, rows: list[dict], response: bool, maximum: int, batch_size: int, heartbeat: Path) -> np.ndarray:
    features = []
    started = time.monotonic()
    device = next(model.parameters()).device
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset:offset + batch_size]
        if response:
            encoded = tokenizer([x['context_text'] for x in batch], [x['candidate_text'] for x in batch], padding=True, truncation='longest_first', max_length=maximum, return_tensors='pt', return_attention_mask=True)
        else:
            encoded = tokenizer([x['text'] for x in batch], padding=True, truncation=True, max_length=maximum, return_tensors='pt', return_attention_mask=True)
        sequence_ids = None
        if response:
            sequence_ids = [item.sequence_ids for item in encoded.encodings]
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = model(**inputs, return_dict=True).last_hidden_state.float()
        pooled = []
        for index in range(len(batch)):
            if response:
                positions = [j for j, value in enumerate(sequence_ids[index]) if value == 1]
                if not positions:
                    raise ValueError('Candidate tokens disappeared during truncation')
                selected = hidden[index, positions]
            else:
                active = inputs['attention_mask'][index].bool()
                selected = hidden[index, active]
            vector = torch.nn.functional.normalize(selected.mean(dim=0), p=2, dim=0)
            if not torch.isfinite(vector).all():
                raise FloatingPointError('Non-finite encoder feature')
            pooled.append(vector)
        features.append(torch.stack(pooled).cpu().numpy().astype(np.float16))
        completed = offset + len(batch)
        elapsed = max(time.monotonic() - started, 1e-09)
        if completed == len(rows) or completed % 1000 < batch_size:
            atomic_json(heartbeat, {'stage': 'response_features' if response else 'recognition_features', 'completed': completed, 'total': len(rows), 'rows_per_second': completed / elapsed, 'eta_seconds': (len(rows) - completed) / (completed / elapsed)})
            print(f'progress={completed}/{len(rows)} response={response}', flush=True)
    return np.concatenate(features, axis=0)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', choices=sorted(ENCODERS), required=True)
    parser.add_argument('--batch-size', type=int, default=48)
    args = parser.parse_args()
    require_preflight()
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the frozen feature extraction')
    snapshot = ENCODERS[args.encoder]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, token=False, trust_remote_code=False)
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True, token=False)
    positional_limit = int(getattr(model_config, 'max_position_embeddings', 512))
    if getattr(model_config, 'model_type', '') == 'roberta':
        positional_limit -= int(getattr(model_config, 'pad_token_id', 1)) + 1
    maximum = min(576, positional_limit)
    model_kwargs = {'local_files_only': True, 'token': False, 'trust_remote_code': False, 'dtype': torch.bfloat16}
    if getattr(model_config, 'model_type', '') in {'bert', 'roberta'}:
        model_kwargs['add_pooling_layer'] = False
    model = AutoModel.from_pretrained(snapshot, **model_kwargs).to('cuda:0').eval()
    output_root = ARTIFACT_ROOT / 'features' / args.encoder
    heartbeat = output_root / 'heartbeat.json'
    recognition_path = ARTIFACT_ROOT / 'data' / 'recognition_records.jsonl'
    recognition_rows = read_jsonl(recognition_path)
    recognition_features = encode_batches(model, tokenizer, recognition_rows, False, maximum, args.batch_size, heartbeat)
    output_root.mkdir(parents=True, exist_ok=True)
    recognition_output = output_root / 'recognition_features.npz'
    np.savez_compressed(recognition_output, features=recognition_features)
    fold_outputs = []
    for fold in range(4):
        tasks_path = ARTIFACT_ROOT / 'data' / f'fold-{fold}' / 'response_tasks.jsonl'
        tasks = read_jsonl(tasks_path)
        values = encode_batches(model, tokenizer, tasks, True, maximum, args.batch_size, heartbeat)
        path = output_root / f'response_features.fold-{fold}.npz'
        np.savez_compressed(path, features=values)
        fold_outputs.append({'fold': fold, 'path': str(path), 'sha256': sha256(path), 'rows': len(tasks)})
    manifest = {'schema_version': 1, 'status': 'complete', 'science_blind': True, 'encoder': args.encoder, 'snapshot': str(snapshot), 'maximum_tokens': maximum, 'pooling': 'mean L2-normalized candidate tokens for response; mean active tokens for recognition', 'recognition': {'path': str(recognition_output), 'sha256': sha256(recognition_output), 'rows': len(recognition_rows), 'dimension': int(recognition_features.shape[1])}, 'response': fold_outputs, 'confirmation_accessed': False, 'official_test_accessed': False}
    atomic_json(output_root / 'manifest.json', manifest)
    print(json.dumps({'status': 'complete', 'encoder': args.encoder}, ensure_ascii=False))
if __name__ == '__main__':
    main()
