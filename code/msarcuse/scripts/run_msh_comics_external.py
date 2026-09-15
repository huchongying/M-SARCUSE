from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
from paper_protocol import ARTIFACT_ROOT, ROOT, require_preflight
RUNS_ROOT = ARTIFACT_ROOT / 'runs-reproducible'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_external_neural_baseline import classification_metrics, read_json, read_jsonl, sha256, train_selector, write_jsonl, atomic_json
TOKEN_PATTERN = re.compile('\\w+|[^\\w\\s]', re.UNICODE)
MAX_UTTERANCES = 6
MAX_UTTERANCE_TOKENS = 64
MAX_VOCAB = 50000

def utterances(text: str) -> list[str]:
    values = [line.strip() for line in text.splitlines() if line.strip()]
    return values[-MAX_UTTERANCES:]

def tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())[:MAX_UTTERANCE_TOKENS]

def build_vocabulary(texts: list[str]) -> dict[str, int]:
    counts = Counter((token for text in texts for line in utterances(text) for token in tokens(line)))
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:MAX_VOCAB - 2]
    return {'<pad>': 0, '<unk>': 1, **{token: index + 2 for index, (token, _) in enumerate(ordered)}}

def encode_texts(texts: list[str], vocabulary: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.zeros((len(texts), MAX_UTTERANCES, MAX_UTTERANCE_TOKENS), dtype=np.int64)
    token_mask = np.zeros_like(ids, dtype=bool)
    utterance_mask = np.zeros((len(texts), MAX_UTTERANCES), dtype=bool)
    for row, text in enumerate(texts):
        lines = utterances(text)
        start = MAX_UTTERANCES - len(lines)
        for offset, line in enumerate(lines):
            values = tokens(line)
            if not values:
                continue
            column = start + offset
            utterance_mask[row, column] = True
            encoded = [vocabulary.get(value, 1) for value in values]
            ids[row, column, :len(encoded)] = encoded
            token_mask[row, column, :len(encoded)] = True
    return (ids, token_mask, utterance_mask)

class MSHComics(torch.nn.Module):

    def __init__(self, vocabulary_size: int, multimodal: bool):
        super().__init__()
        self.multimodal = multimodal
        self.embedding = torch.nn.Embedding(vocabulary_size, 300, padding_idx=0)
        self.word_conv = torch.nn.Conv1d(300, 128, kernel_size=3, padding=1)
        self.word_attention = torch.nn.Linear(128, 1)
        self.context_lstm = torch.nn.LSTM(128, 128, batch_first=True)
        self.context_conv = torch.nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.context_attention = torch.nn.Linear(128, 1)
        if multimodal:
            self.audio_projection = torch.nn.Linear(128, 128)
            self.filter_gate = torch.nn.Linear(256, 128)
            self.norm = torch.nn.LayerNorm(128)
        self.dropout = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Linear(128, 1)

    def representation(self, ids, token_mask, utterance_mask, audio):
        batch, turns, length = ids.shape
        embedded = self.embedding(ids).reshape(batch * turns, length, 300).transpose(1, 2)
        words = torch.relu(self.word_conv(embedded)).transpose(1, 2)
        word_mask = token_mask.reshape(batch * turns, length)
        scores = self.word_attention(words).squeeze(2).masked_fill(~word_mask, -10000.0)
        weights = torch.softmax(scores, dim=1) * word_mask.float()
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
        utterance = (words * (weights / denominator).unsqueeze(2)).sum(dim=1).reshape(batch, turns, 128)
        contextual, _ = self.context_lstm(utterance)
        contextual = torch.relu(self.context_conv(contextual.transpose(1, 2)).transpose(1, 2))
        scores = self.context_attention(contextual).squeeze(2).masked_fill(~utterance_mask, -10000.0)
        weights = torch.softmax(scores, dim=1) * utterance_mask.float()
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
        text = (contextual * (weights / denominator).unsqueeze(2)).sum(dim=1)
        if not self.multimodal:
            return text
        acoustic = torch.tanh(self.audio_projection(audio))
        gate = torch.sigmoid(self.filter_gate(torch.cat([text, acoustic], dim=1)))
        return self.norm(text + gate * acoustic)

    def forward(self, ids, token_mask, utterance_mask, audio):
        hidden = self.representation(ids, token_mask, utterance_mask, audio)
        return self.classifier(self.dropout(hidden)).squeeze(1)

def fit_model(model, arrays, audio, labels, train_mask, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    indices = np.flatnonzero(train_mask)
    tensors = [torch.from_numpy(value[indices]) for value in arrays]
    dataset = torch.utils.data.TensorDataset(*tensors, torch.from_numpy(audio[indices]), torch.from_numpy(labels[indices].astype(np.float32)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=device))
    for _ in range(15):
        model.train()
        for ids, token_mask, utterance_mask, batch_audio, batch_labels in loader:
            ids, token_mask, utterance_mask = (ids.to(device), token_mask.to(device), utterance_mask.to(device))
            batch_audio, batch_labels = (batch_audio.to(device), batch_labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(ids, token_mask, utterance_mask, batch_audio), batch_labels)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def predict(model, arrays, audio, device, batch_size=256):
    probabilities, features = ([], [])
    with torch.inference_mode():
        for start in range(0, len(audio), batch_size):
            values = [torch.from_numpy(x[start:start + batch_size]).to(device) for x in arrays]
            batch_audio = torch.from_numpy(audio[start:start + batch_size]).to(device)
            hidden = model.representation(*values, batch_audio)
            logits = model.classifier(hidden).squeeze(1)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            features.append(torch.nn.functional.normalize(hidden, p=2, dim=1).cpu().numpy())
    return (np.concatenate(probabilities), np.concatenate(features).astype(np.float32))

def run(fold: int) -> None:
    require_preflight()
    config_path = ROOT / 'config' / 'msarcuse_external_baselines.draft.json'
    config = read_json(config_path)
    method = next((x for x in config['baselines'] if x['method_id'] == 'msh-comics'))
    records = read_jsonl(ARTIFACT_ROOT / 'data' / 'recognition_records.jsonl')
    labels = np.asarray([x['target_label'] for x in records], dtype=np.int64)
    eval_mask = np.asarray([x['evaluation_fold'] == fold for x in records], dtype=bool)
    train_mask = ~eval_mask
    train_episodes = {x['episode_id'] for x, keep in zip(records, train_mask, strict=True) if keep}
    eval_episodes = {x['episode_id'] for x, keep in zip(records, eval_mask, strict=True) if keep}
    if train_episodes & eval_episodes:
        raise ValueError('Recognition episode overlap')
    tasks = read_jsonl(ARTIFACT_ROOT / 'data' / f'fold-{fold}' / 'response_tasks.jsonl')
    task_train = np.asarray([x['split'] == 'training' for x in tasks], dtype=bool)
    vocabulary_texts = [x['text'] for x, keep in zip(records, train_mask, strict=True) if keep]
    vocabulary_texts.extend((x['context_text'] + '\n' + x['candidate_text'] for x, keep in zip(tasks, task_train, strict=True) if keep))
    vocabulary = build_vocabulary(vocabulary_texts)
    record_arrays = encode_texts([x['text'] for x in records], vocabulary)
    task_arrays = encode_texts([x['context_text'] + '\n' + x['candidate_text'] for x in tasks], vocabulary)
    raw_audio = np.asarray([x['target_audio'] for x in records], dtype=np.float32)
    scaler = StandardScaler().fit(raw_audio[train_mask])
    audio = scaler.transform(raw_audio).astype(np.float32)
    task_audio = scaler.transform(np.asarray([x['target_audio'] for x in tasks], dtype=np.float32)).astype(np.float32)
    device = torch.device('cuda:0')
    run_dir = RUNS_ROOT / 'msh-comics' / f'fold-{fold}'
    run_dir.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(config_path)
    summary = {'method': 'msh-comics', 'fold': fold, 'seeds': []}
    for seed in config['seeds']:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        text_model = fit_model(MSHComics(len(vocabulary), False), record_arrays, audio, labels, train_mask, seed, device)
        fused_model = fit_model(MSHComics(len(vocabulary), True), record_arrays, audio, labels, train_mask, seed, device)
        text_p, _ = predict(text_model, tuple((x[eval_mask] for x in record_arrays)), audio[eval_mask], device)
        fused_p, _ = predict(fused_model, tuple((x[eval_mask] for x in record_arrays)), audio[eval_mask], device)
        eval_records = [x for x, keep in zip(records, eval_mask, strict=True) if keep]
        prediction_rows = [{'method': 'msh-comics', 'source_revision': method['source_revision'], 'adapter_config_sha256': config_hash, 'fold': fold, 'seed': seed, 'source_index': row['source_index'], 'dialogue_id': row['dialogue_id'], 'episode_id': row['episode_id'], 'true_label': row['target_label'], 'text_probability': float(pt), 'text_label': int(pt >= 0.5), 'text_audio_probability': float(pa), 'text_audio_label': int(pa >= 0.5)} for row, pt, pa in zip(eval_records, text_p, fused_p, strict=True)]
        prediction_path = run_dir / f'recognition_predictions.seed-{seed}.jsonl'
        write_jsonl(prediction_path, prediction_rows)
        _, text_hidden = predict(text_model, task_arrays, task_audio, device)
        _, fused_hidden = predict(fused_model, task_arrays, task_audio, device)
        text_hidden = np.concatenate([text_hidden, np.zeros((len(tasks), 1), dtype=np.float32)], axis=1)
        fused_hidden = np.concatenate([fused_hidden, np.ones((len(tasks), 1), dtype=np.float32)], axis=1)
        train_features = np.concatenate([text_hidden[task_train], fused_hidden[task_train]], axis=0)
        eval_features = np.concatenate([text_hidden[~task_train], fused_hidden[~task_train]], axis=0)
        train_tasks = [x for x, keep in zip(tasks, task_train, strict=True) if keep]
        eval_tasks = [x for x, keep in zip(tasks, ~task_train, strict=True) if keep]
        train_labels = np.asarray([float(x['is_gold']) for x in train_tasks] * 2, dtype=np.float32)
        fields = ('sample_id', 'dialogue_pair_id', 'episode_id', 'source_index', 'target_label', 'candidate_id', 'candidate_source_index', 'candidate_source_dialogue_id', 'is_gold')
        train_meta, eval_meta = ([], [])
        for condition in ('text', 'text_audio'):
            train_meta.extend(({**{key: x[key] for key in fields}, 'condition': condition} for x in train_tasks))
            eval_meta.extend(({**{key: x[key] for key in fields}, 'condition': condition} for x in eval_tasks))
        selector, selector_metrics = train_selector(train_features, train_labels, train_meta, config['selector'], seed, device)
        with torch.inference_mode():
            logits = selector(torch.from_numpy(eval_features).to(device)).cpu().numpy()
        score_rows = [{'method': 'msh-comics', 'source_revision': method['source_revision'], 'adapter_config_sha256': config_hash, 'fold': fold, 'seed': seed, **row, 'selector_logit': float(logit)} for row, logit in zip(eval_meta, logits, strict=True)]
        score_path = run_dir / f'candidate_scores.seed-{seed}.jsonl'
        write_jsonl(score_path, score_rows)
        torch.save({'vocabulary': vocabulary, 'text_model': text_model.state_dict(), 'fused_model': fused_model.state_dict()}, run_dir / f'recognizer.seed-{seed}.pt')
        torch.save({'state_dict': selector.state_dict(), 'input_size': train_features.shape[1]}, run_dir / f'selector_head.seed-{seed}.pt')
        summary['seeds'].append({'seed': seed, 'recognition_text': classification_metrics(labels[eval_mask], text_p), 'recognition_text_audio': classification_metrics(labels[eval_mask], fused_p), 'selector_training': selector_metrics, 'recognition_predictions_sha256': sha256(prediction_path), 'candidate_scores_sha256': sha256(score_path), 'candidate_score_rows': len(score_rows)})
        del text_model, fused_model, selector
        torch.cuda.empty_cache()
    summary.update({'status': 'complete', 'attempt': 'A2_seeded_initialization', 'seed_set_before_model_instantiation': True, 'config_sha256': config_hash, 'vocabulary_size': len(vocabulary), 'recognizer_training_episodes': len(train_episodes), 'recognizer_evaluation_episodes': len(eval_episodes), 'recognizer_episode_overlap': False, 'selector_training_targets': int(task_train.sum() // 4), 'selector_evaluation_targets': int((~task_train).sum() // 4), 'confirmation_accessed': False, 'official_test_accessed': False})
    atomic_json(run_dir / 'run_summary.json', summary)
    print(json.dumps({'status': 'complete', 'method': 'msh-comics', 'fold': fold}))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    run(args.fold)
if __name__ == '__main__':
    main()
