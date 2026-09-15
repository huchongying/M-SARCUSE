from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
from paper_protocol import ARTIFACT_ROOT, ROOT, require_preflight
RUNS_ROOT = ARTIFACT_ROOT / 'runs-reproducible'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sarcnext_common import stable_int
METHOD_ENCODER = {'frozen-early-fusion': 'roberta', 'msnr-msd': 'bart', 'mo-sarcation': 'bart'}

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    tmp.replace(path)

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {'samples': int(len(labels)), 'accuracy': float(accuracy_score(labels, predictions)), 'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)), 'f1': float(f1_score(labels, predictions, zero_division=0))}

class RecognitionAdapter(torch.nn.Module):

    def __init__(self, kind: str, text_size: int, hidden_size: int, dropout: float, multimodal: bool):
        super().__init__()
        self.kind = kind
        self.multimodal = multimodal
        if kind == 'frozen-early-fusion':
            input_size = text_size + (128 if multimodal else 0)
            self.input_projection = torch.nn.Linear(input_size, hidden_size)
            self.norm = torch.nn.LayerNorm(hidden_size)
        else:
            self.text_projection = torch.nn.Linear(text_size, hidden_size)
            self.norm = torch.nn.LayerNorm(hidden_size)
            if multimodal:
                self.audio_projection = torch.nn.Linear(128, hidden_size)
                if kind == 'msnr-msd':
                    self.gate = torch.nn.Linear(hidden_size * 2, hidden_size)
                elif kind == 'mo-sarcation':
                    self.gate = torch.nn.Linear(hidden_size, hidden_size)
                else:
                    raise ValueError(kind)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(hidden_size, 1)

    def representation(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if self.kind == 'frozen-early-fusion':
            values = torch.cat([text, audio], dim=1) if self.multimodal else text
            return self.norm(torch.nn.functional.gelu(self.input_projection(values)))
        text_hidden = self.text_projection(text)
        if not self.multimodal:
            return self.norm(text_hidden)
        audio_hidden = self.audio_projection(audio)
        if self.kind == 'msnr-msd':
            gate = torch.sigmoid(self.gate(torch.cat([text_hidden, audio_hidden], dim=1)))
        else:
            gate = torch.sigmoid(self.gate(audio_hidden))
        return self.norm(text_hidden + gate * audio_hidden)

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.representation(text, audio))).squeeze(1)

class SelectorHead(torch.nn.Module):

    def __init__(self, input_size: int, config: dict):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_size, config['hidden_size']), torch.nn.GELU(), torch.nn.Dropout(config['dropout']), torch.nn.Linear(config['hidden_size'], 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values).squeeze(1)

def fit_recognizer(model: RecognitionAdapter, text: np.ndarray, audio: np.ndarray, labels: np.ndarray, train_mask: np.ndarray, method_config: dict, seed: int, device: torch.device) -> RecognitionAdapter:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    indices = np.flatnonzero(train_mask)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(text[indices]), torch.from_numpy(audio[indices]), torch.from_numpy(labels[indices].astype(np.float32)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=method_config['batch_size'], shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=method_config['learning_rate'], weight_decay=method_config.get('weight_decay', 0.0))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    model.to(device)
    for _ in range(method_config['epochs']):
        model.train()
        for batch_text, batch_audio, batch_labels in loader:
            batch_text = batch_text.to(device)
            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_text, batch_audio), batch_labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def predict(model: RecognitionAdapter, text: np.ndarray, audio: np.ndarray, batch_size: int, device: torch.device):
    probabilities, representations = ([], [])
    with torch.inference_mode():
        for offset in range(0, len(text), batch_size):
            batch_text = torch.from_numpy(text[offset:offset + batch_size]).to(device)
            batch_audio = torch.from_numpy(audio[offset:offset + batch_size]).to(device)
            hidden = model.representation(batch_text, batch_audio)
            logits = model.classifier(hidden).squeeze(1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            representations.append(torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy())
    return (np.concatenate(probabilities), np.concatenate(representations).astype(np.float32))

def response_accuracy(logits: np.ndarray, labels: np.ndarray, metadata: list[dict]) -> float:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[row['sample_id'], row['condition']].append(index)
    return sum((labels[idxs[int(np.argmax(logits[idxs]))]] == 1 for idxs in groups.values())) / len(groups)

def train_selector(features, labels, metadata, selector_config, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    val_mask = np.asarray([stable_int(x['episode_id'], 'sarcnext-head-v1') % 10 == 0 for x in metadata], dtype=bool)
    train_episodes = {x['episode_id'] for x, keep in zip(metadata, ~val_mask, strict=True) if keep}
    val_episodes = {x['episode_id'] for x, keep in zip(metadata, val_mask, strict=True) if keep}
    if not val_mask.any() or train_episodes & val_episodes:
        raise ValueError('Selector validation episode isolation failed')
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels.astype(np.float32))
    train_idx, val_idx = (np.flatnonzero(~val_mask), np.flatnonzero(val_mask))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[train_idx], y[train_idx]), batch_size=selector_config['batch_size'], shuffle=True, generator=generator)
    model = SelectorHead(features.shape[1], selector_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=selector_config['learning_rate'], weight_decay=selector_config['weight_decay'])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(selector_config['positive_weight'], device=device))
    val_x, val_y = (x[val_idx].to(device), y[val_idx].to(device))
    best_loss, best_epoch, best_state, history = (float('inf'), 0, None, [])
    for epoch in range(1, selector_config['epochs'] + 1):
        model.train()
        for bx, by in loader:
            bx, by = (bx.to(device), by.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(val_x)
            loss = float(loss_fn(logits, val_y).item())
        accuracy = response_accuracy(logits.cpu().numpy(), labels[val_idx], [metadata[i] for i in val_idx])
        history.append({'epoch': epoch, 'validation_bce': loss, 'validation_response_accuracy': accuracy})
        if loss < best_loss - 1e-12:
            best_loss, best_epoch = (loss, epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return (model, {'best_epoch': best_epoch, 'best_validation_bce': best_loss, 'training_episodes': len(train_episodes), 'validation_episodes': len(val_episodes), 'training_and_validation_episode_overlap': False, 'history': history})

def run(method_id: str, fold: int) -> None:
    require_preflight()
    config_path = ROOT / 'config' / 'msarcuse_external_baselines.draft.json'
    config = read_json(config_path)
    method_config = next((x for x in config['baselines'] if x['method_id'] == method_id))
    encoder = METHOD_ENCODER[method_id]
    device = torch.device('cuda:0')
    recognition = read_jsonl(ARTIFACT_ROOT / 'data' / 'recognition_records.jsonl')
    text = np.load(ARTIFACT_ROOT / 'features' / encoder / 'recognition_features.npz')['features'].astype(np.float32)
    labels = np.asarray([x['target_label'] for x in recognition], dtype=np.int64)
    eval_mask = np.asarray([x['evaluation_fold'] == fold for x in recognition], dtype=bool)
    train_mask = ~eval_mask
    train_episodes = {x['episode_id'] for x, keep in zip(recognition, train_mask, strict=True) if keep}
    eval_episodes = {x['episode_id'] for x, keep in zip(recognition, eval_mask, strict=True) if keep}
    if train_episodes & eval_episodes:
        raise ValueError('Recognition episode overlap')
    raw_audio = np.asarray([x['target_audio'] for x in recognition], dtype=np.float32)
    scaler = StandardScaler().fit(raw_audio[train_mask])
    audio = scaler.transform(raw_audio).astype(np.float32)
    tasks = read_jsonl(ARTIFACT_ROOT / 'data' / f'fold-{fold}' / 'response_tasks.jsonl')
    task_text = np.load(ARTIFACT_ROOT / 'features' / encoder / f'response_features.fold-{fold}.npz')['features'].astype(np.float32)
    task_training = np.asarray([x['split'] == 'training' for x in tasks], dtype=bool)
    task_audio = scaler.transform(np.asarray([x['target_audio'] for x in tasks], dtype=np.float32)).astype(np.float32)
    run_dir = RUNS_ROOT / method_id / f'fold-{fold}'
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {'method': method_id, 'fold': fold, 'seeds': []}
    config_hash = sha256(config_path)
    for seed in config['seeds']:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text_model = RecognitionAdapter(method_id, text.shape[1], method_config['hidden_size'], method_config['dropout'], False)
        fused_model = RecognitionAdapter(method_id, text.shape[1], method_config['hidden_size'], method_config['dropout'], True)
        text_model = fit_recognizer(text_model, text, audio, labels, train_mask, method_config, seed, device)
        fused_model = fit_recognizer(fused_model, text, audio, labels, train_mask, method_config, seed, device)
        text_p, _ = predict(text_model, text[eval_mask], audio[eval_mask], 256, device)
        fused_p, _ = predict(fused_model, text[eval_mask], audio[eval_mask], 256, device)
        eval_records = [x for x, keep in zip(recognition, eval_mask, strict=True) if keep]
        prediction_rows = []
        for row, pt, pa in zip(eval_records, text_p, fused_p, strict=True):
            prediction_rows.append({'method': method_id, 'source_revision': method_config['source_revision'], 'adapter_config_sha256': config_hash, 'fold': fold, 'seed': seed, 'source_index': row['source_index'], 'dialogue_id': row['dialogue_id'], 'episode_id': row['episode_id'], 'true_label': row['target_label'], 'text_probability': float(pt), 'text_label': int(pt >= 0.5), 'text_audio_probability': float(pa), 'text_audio_label': int(pa >= 0.5)})
        prediction_path = run_dir / f'recognition_predictions.seed-{seed}.jsonl'
        write_jsonl(prediction_path, prediction_rows)
        _, task_text_hidden = predict(text_model, task_text, task_audio, 256, device)
        _, task_fused_hidden = predict(fused_model, task_text, task_audio, 256, device)
        task_text_hidden = np.concatenate([task_text_hidden, np.zeros((len(tasks), 1), dtype=np.float32)], axis=1)
        task_fused_hidden = np.concatenate([task_fused_hidden, np.ones((len(tasks), 1), dtype=np.float32)], axis=1)
        train_features = np.concatenate([task_text_hidden[task_training], task_fused_hidden[task_training]], axis=0)
        eval_features = np.concatenate([task_text_hidden[~task_training], task_fused_hidden[~task_training]], axis=0)
        train_tasks = [x for x, keep in zip(tasks, task_training, strict=True) if keep]
        eval_tasks = [x for x, keep in zip(tasks, ~task_training, strict=True) if keep]
        labels_train = np.asarray([float(x['is_gold']) for x in train_tasks] * 2, dtype=np.float32)
        train_meta, eval_meta = ([], [])
        for condition in ('text', 'text_audio'):
            for x in train_tasks:
                train_meta.append({**{k: x[k] for k in ('sample_id', 'dialogue_pair_id', 'episode_id', 'source_index', 'target_label', 'candidate_id', 'candidate_source_index', 'candidate_source_dialogue_id', 'is_gold')}, 'condition': condition})
            for x in eval_tasks:
                eval_meta.append({**{k: x[k] for k in ('sample_id', 'dialogue_pair_id', 'episode_id', 'source_index', 'target_label', 'candidate_id', 'candidate_source_index', 'candidate_source_dialogue_id', 'is_gold')}, 'condition': condition})
        selector, selector_metrics = train_selector(train_features, labels_train, train_meta, config['selector'], seed, device)
        with torch.inference_mode():
            logits = selector(torch.from_numpy(eval_features).to(device)).cpu().numpy()
        score_rows = [{'method': method_id, 'source_revision': method_config['source_revision'], 'adapter_config_sha256': config_hash, 'fold': fold, 'seed': seed, **row, 'selector_logit': float(logit)} for row, logit in zip(eval_meta, logits, strict=True)]
        score_path = run_dir / f'candidate_scores.seed-{seed}.jsonl'
        write_jsonl(score_path, score_rows)
        torch.save({'text_model': text_model.state_dict(), 'fused_model': fused_model.state_dict()}, run_dir / f'recognizer.seed-{seed}.pt')
        torch.save({'state_dict': selector.state_dict(), 'input_size': train_features.shape[1]}, run_dir / f'selector_head.seed-{seed}.pt')
        summary['seeds'].append({'seed': seed, 'recognition_text': classification_metrics(labels[eval_mask], text_p), 'recognition_text_audio': classification_metrics(labels[eval_mask], fused_p), 'selector_training': selector_metrics, 'recognition_predictions_sha256': sha256(prediction_path), 'candidate_scores_sha256': sha256(score_path), 'candidate_score_rows': len(score_rows)})
        del text_model, fused_model, selector
        torch.cuda.empty_cache()
    summary.update({'status': 'complete', 'attempt': 'A2_seeded_initialization', 'seed_set_before_model_instantiation': True, 'config_sha256': config_hash, 'recognizer_training_episodes': len(train_episodes), 'recognizer_evaluation_episodes': len(eval_episodes), 'recognizer_episode_overlap': False, 'selector_training_targets': int(task_training.sum() // 4), 'selector_evaluation_targets': int((~task_training).sum() // 4), 'confirmation_accessed': False, 'official_test_accessed': False})
    atomic_json(run_dir / 'run_summary.json', summary)
    print(json.dumps({'status': 'complete', 'method': method_id, 'fold': fold}))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=sorted(METHOD_ENCODER), required=True)
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    run(args.method, args.fold)
if __name__ == '__main__':
    main()
