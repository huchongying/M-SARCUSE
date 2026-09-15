from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from paper_protocol import ARTIFACT_ROOT, ROOT, SEEDS, load_paper_inputs, require_preflight
METHODS = ['msh-comics', 'mbert-acoustic-rf', 'msnr-msd', 'frozen-early-fusion', 'mo-sarcation']

def read_jsonl(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def safe_mean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) else float('nan')

def pooled_values(recognition: pd.DataFrame, response: pd.DataFrame) -> dict:
    labels = recognition.true_label.to_numpy()
    text_label = recognition.text_label.to_numpy()
    audio_label = recognition.text_audio_label.to_numpy()
    text_correct = text_label == labels
    audio_correct = audio_label == labels
    corrected = response.recognition_group == 'corrected'
    delta = response.response_text_audio_correct.astype(int) - response.response_text_correct.astype(int)
    literal = response.target_label == 0
    sarcastic = response.target_label == 1
    helpful = int((delta == 1).sum())
    harmful = int((delta == -1).sum())
    hc = int((delta[corrected] == 1).sum())
    dc = int((delta[corrected] == -1).sum())
    nc = int(corrected.sum())
    hr = int((~text_correct & audio_correct).sum())
    dr = int((text_correct & ~audio_correct).sum())
    if hr != nc or len(recognition) != len(response):
        raise ValueError('Recognition and response must use the same paired observations')
    def ba(predictions):
        if set(labels) != {0, 1}:
            return float('nan')
        return float(balanced_accuracy_score(labels, predictions))
    text_ba, audio_ba = ba(text_label), ba(audio_label)
    dl, ds = safe_mean(delta[literal]), safe_mean(delta[sarcastic])
    return {
        'recognition_text_f1': float(f1_score(labels, text_label, zero_division=0)),
        'recognition_text_ba': text_ba,
        'recognition_text_accuracy': float(accuracy_score(labels, text_label)),
        'recognition_text_audio_f1': float(f1_score(labels, audio_label, zero_division=0)),
        'recognition_text_audio_ba': audio_ba,
        'recognition_text_audio_accuracy': float(accuracy_score(labels, audio_label)),
        'delta_recognition': audio_ba - text_ba,
        'response_text_accuracy': safe_mean(response.response_text_correct),
        'response_text_audio_accuracy': safe_mean(response.response_text_audio_correct),
        'delta_response': safe_mean(delta), 'delta_literal': dl,
        'delta_sarcastic': ds, 'did': ds - dl,
        'delta_corrected': (hc - dc) / nc if nc else float('nan'),
        'observations': int(len(response)), 'corrected_n': nc,
        'helpful': helpful, 'harmful': harmful,
        'helpful_corrected': hc, 'harmful_corrected': dc,
        'unchanged_corrected': nc - hc - dc,
        'recognition_helpful': hr, 'recognition_harmful': dr,
        'recognition_net': hr - dr,
        'help_harm_ratio': helpful / harmful if harmful else None,
        'corrected_help_harm_ratio': hc / dc if dc else None,
        'oracle_route_gain': helpful / len(response),
    }


def metric_values(recognition: pd.DataFrame, response: pd.DataFrame) -> dict:
    by_seed = []
    for seed, rec in recognition.groupby('seed'):
        res = response[response.seed == seed]
        by_seed.append({'seed': int(seed), **pooled_values(rec, res)})
    return {'aggregate': pooled_values(recognition, response), 'by_seed': by_seed}


def assemble(method: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    recognition_rows, response_rows, provenance = ([], [], [])
    inputs = load_paper_inputs()
    expected_targets = {(t['fold'], t['sample_id']): t for t in inputs['targets']}
    config_hash = sha256(ROOT / 'config' / 'msarcuse_external_baselines.draft.json')
    for fold in range(4):
        run_family = 'runs' if method == 'mbert-acoustic-rf' else 'runs-reproducible'
        run_dir = ARTIFACT_ROOT / run_family / method / f'fold-{fold}'
        summary = json.loads((run_dir / 'run_summary.json').read_text(encoding='utf-8'))
        if summary.get('config_sha256') != config_hash:
            raise ValueError(f'Run configuration mismatch: {method} fold {fold}')
        if summary.get('status') != 'complete' or summary.get('recognizer_episode_overlap') is not False:
            raise ValueError(f'Incomplete or unsafe run: {method} fold {fold}')
        if method != 'mbert-acoustic-rf' and summary.get('seed_set_before_model_instantiation') is not True:
            raise ValueError(f'Unseeded neural initialization: {method} fold {fold}')
        if sorted(x['seed'] for x in summary['seeds']) != sorted(SEEDS):
            raise ValueError('Missing or duplicate model-seed runs')
        for seed_entry in summary['seeds']:
            seed = int(seed_entry['seed'])
            prediction_path = run_dir / f'recognition_predictions.seed-{seed}.jsonl'
            score_path = run_dir / f'candidate_scores.seed-{seed}.jsonl'
            if sha256(prediction_path) != seed_entry['recognition_predictions_sha256']:
                raise ValueError('Recognition raw hash mismatch')
            if sha256(score_path) != seed_entry['candidate_scores_sha256']:
                raise ValueError('Candidate-score raw hash mismatch')
            predictions = read_jsonl(prediction_path)
            recognition_map = {int(x['source_index']): x for x in predictions}
            if len(recognition_map) != len(predictions):
                raise ValueError('Duplicate recognition source_index')
            if any(p['fold'] != fold or p['seed'] != seed for p in predictions):
                raise ValueError('Recognition fold/seed mismatch')
            scores = read_jsonl(score_path)
            grouped = defaultdict(list)
            for row in scores:
                if (row['method'] != method or row['fold'] != fold or row['seed'] != seed
                        or row['adapter_config_sha256'] != config_hash):
                    raise ValueError('Candidate-score run identity mismatch')
                grouped[row['sample_id'], row['condition']].append(row)
            selected = {}
            for key, candidates in grouped.items():
                if len(candidates) != 4 or sum((bool(x['is_gold']) for x in candidates)) != 1:
                    raise ValueError(f'Incomplete candidate group {method} {fold} {seed} {key}')
                if key[1] not in {'text', 'text_audio'}:
                    raise ValueError(f'Unexpected condition: {key}')
                expected = expected_targets[fold, key[0]]
                if any(x['source_index'] != expected['source_index']
                       or x['episode_id'] != expected['episode_id']
                       or x['target_label'] != expected['target_label']
                       or type(x['is_gold']) is not bool for x in candidates):
                    raise ValueError(f'Candidate target metadata mismatch: {key}')
                signature = [(x['candidate_id'], x['candidate_source_index'], bool(x['is_gold'])) for x in candidates]
                reference = [(x['candidate_id'], x['source_index'], x['is_gold']) for x in expected['candidates']]
                if signature != reference:
                    raise ValueError(f'Candidate order/content identity mismatch: {key}')
                if not all(np.isfinite(float(x['selector_logit'])) for x in candidates):
                    raise ValueError(f'Nonfinite candidate score: {key}')
                choice = max(candidates, key=lambda x: float(x['selector_logit']))
                selected[key] = int(bool(choice['is_gold']))
            samples = sorted({x['sample_id'] for x in scores})
            if set(samples) != {t['sample_id'] for t in inputs['targets'] if t['fold'] == fold}:
                raise ValueError('Scored targets differ from the final paper target list')
            for sample in samples:
                template = next((x for x in scores if x['sample_id'] == sample))
                source_index = int(template['source_index'])
                rec = recognition_map[source_index]
                if (rec['method'] != method
                        or rec['adapter_config_sha256'] != config_hash):
                    raise ValueError('Recognition run identity mismatch')
                expected = expected_targets[fold, sample]
                if (source_index != expected['source_index']
                        or template['episode_id'] != expected['episode_id']
                        or rec['episode_id'] != expected['episode_id']
                        or rec['true_label'] != expected['target_label']):
                    raise ValueError('Recognition/response target identity mismatch')
                if any(rec[k] not in (0, 1) for k in ('true_label', 'text_label', 'text_audio_label')):
                    raise ValueError('Recognition decisions must be binary')
                recognition_rows.append({**rec, 'sample_id': sample})
                truth = int(rec['true_label'])
                text_rec_correct = int(rec['text_label'] == truth)
                audio_rec_correct = int(rec['text_audio_label'] == truth)
                recognition_group = {(0, 1): 'corrected', (1, 0): 'corrupted', (1, 1): 'stable-correct', (0, 0): 'stable-wrong'}[text_rec_correct, audio_rec_correct]
                text_correct = selected[sample, 'text']
                audio_correct = selected[sample, 'text_audio']
                response_transition = {(0, 1): 'helpful', (1, 0): 'harmful', (1, 1): 'stable-correct', (0, 0): 'stable-wrong'}[text_correct, audio_correct]
                response_rows.append({'method': method, 'fold': fold, 'seed': seed, 'sample_id': sample, 'dialogue_pair_id': template['dialogue_pair_id'], 'episode_id': template['episode_id'], 'source_index': source_index, 'target_label': int(template['target_label']), 'response_text_correct': text_correct, 'response_text_audio_correct': audio_correct, 'recognition_group': recognition_group, 'response_transition': response_transition})
            provenance.append({'fold': fold, 'seed': seed, 'recognition': str(prediction_path), 'recognition_sha256': sha256(prediction_path), 'scores': str(score_path), 'scores_sha256': sha256(score_path)})
    return (pd.DataFrame(recognition_rows), pd.DataFrame(response_rows), provenance)

def transition_table(response: pd.DataFrame) -> list[dict]:
    table = response.groupby(['recognition_group', 'response_transition']).size()
    result = []
    for recognition_group in ('corrected', 'corrupted', 'stable-correct', 'stable-wrong'):
        for response_transition in ('helpful', 'harmful', 'stable-correct', 'stable-wrong'):
            result.append({'recognition_group': recognition_group, 'response_transition': response_transition, 'count': int(table.get((recognition_group, response_transition), 0))})
    return result

def episode_arrays(recognition: pd.DataFrame, response: pd.DataFrame, episodes=None):
    episodes = sorted(set(response.episode_id)) if episodes is None else list(episodes)
    if not set(recognition.episode_id).issubset(episodes):
        raise ValueError('Recognition episode absent from response resampling frame')
    rec_stats, res_stats = [], []
    for episode in episodes:
        rec = recognition[recognition.episode_id == episode]
        res = response[response.episode_id == episode]
        positive, negative = rec.true_label == 1, rec.true_label == 0
        rec_stats.append([
            positive.sum(), negative.sum(),
            ((rec.text_label == rec.true_label) & positive).sum(),
            ((rec.text_label == rec.true_label) & negative).sum(),
            ((rec.text_audio_label == rec.true_label) & positive).sum(),
            ((rec.text_audio_label == rec.true_label) & negative).sum()])
        delta = res.response_text_audio_correct.astype(int) - res.response_text_correct.astype(int)
        literal, sarcastic = res.target_label == 0, res.target_label == 1
        corrected = res.recognition_group == 'corrected'
        res_stats.append([literal.sum(), sarcastic.sum(), corrected.sum(),
                          delta[literal].sum(), delta[sarcastic].sum(), delta[corrected].sum()])
    return np.asarray(rec_stats, dtype=float), np.asarray(res_stats, dtype=float)


def bootstrap_draws(rec_stats, res_stats, weights):
    rec_total, res_total = weights @ rec_stats, weights @ res_stats
    with np.errstate(divide='ignore', invalid='ignore'):
        text_ba = 0.5 * (rec_total[:, 2] / rec_total[:, 0] + rec_total[:, 3] / rec_total[:, 1])
        audio_ba = 0.5 * (rec_total[:, 4] / rec_total[:, 0] + rec_total[:, 5] / rec_total[:, 1])
        dl, ds = res_total[:, 3] / res_total[:, 0], res_total[:, 4] / res_total[:, 1]
        dc = res_total[:, 5] / res_total[:, 2]
        response = (res_total[:, 3] + res_total[:, 4]) / (res_total[:, 0] + res_total[:, 1])
    return {'delta_recognition': audio_ba - text_ba, 'delta_literal': dl,
            'delta_sarcastic': ds, 'delta_response': response, 'did': ds - dl,
            'delta_corrected': dc}


def percentile_intervals(draws):
    result = {}
    for key, values in draws.items():
        finite = np.asarray(values)[np.isfinite(values)]
        if not len(finite):
            result[key] = {'lower': None, 'upper': None, 'valid_resamples': 0,
                           'undefined_resamples': int(len(values))}
        else:
            lower, upper = np.percentile(finite, [2.5, 97.5])
            result[key] = {'lower': float(lower), 'upper': float(upper),
                           'valid_resamples': int(len(finite)),
                           'undefined_resamples': int(len(values) - len(finite))}
    return result


def cluster_bootstrap(recognition: pd.DataFrame, response: pd.DataFrame, resamples=2000, seed=20260831) -> dict:
    if resamples < 1:
        raise ValueError('resamples must be positive')
    rec_stats, res_stats = episode_arrays(recognition, response)
    n = len(rec_stats)
    if n < 2:
        raise ValueError('At least two episode clusters are required')
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1.0 / n), size=resamples)
    return percentile_intervals(bootstrap_draws(rec_stats, res_stats, weights))


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', action='append', choices=METHODS)
    parser.add_argument('--bootstrap-resamples', type=int)
    args = parser.parse_args()
    require_preflight()
    config = json.loads((ROOT / 'config' / 'msarcuse_external_baselines.draft.json').read_text(encoding='utf-8'))
    configured_resamples = int(config['bootstrap']['resamples'])
    if args.bootstrap_resamples is not None and args.bootstrap_resamples != configured_resamples:
        raise ValueError(f'Bootstrap resamples must equal the paper setting: {configured_resamples}')
    bootstrap_resamples = configured_resamples
    selected = args.method or METHODS
    output_root = ARTIFACT_ROOT / 'analysis-reproducible'
    output_root.mkdir(parents=True, exist_ok=True)
    unified = []
    for method in selected:
        recognition, response, provenance = assemble(method)
        values = metric_values(recognition, response)
        ci = cluster_bootstrap(recognition, response, bootstrap_resamples)
        result = {'schema_version': 2, 'units': 'proportion; multiply estimates and CI bounds by 100 for percentage points', 'pooling': 'micro', 'method': method, **values, 'ci95': ci, 'transition_table': transition_table(response), 'raw_provenance': provenance, 'bootstrap': {'resamples': bootstrap_resamples, 'seed': 20260831, 'stratified_by': None, 'clustered_by': 'episode', 'paired_episode_weights': True}}
        path = output_root / f'{method}.metrics.json'
        path.write_text(json.dumps(json_finite(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        response.to_csv(output_root / f'{method}.response_outcomes.csv', index=False, encoding='utf-8')
        row = {'method': method, **values['aggregate']}
        for key, bounds in ci.items():
            row[f'{key}_ci95_lower'] = bounds['lower']
            row[f'{key}_ci95_upper'] = bounds['upper']
        unified.append(row)
    pd.DataFrame(unified).to_csv(output_root / 'unified_comparison.csv', index=False, encoding='utf-8')
    manifest = {'schema_version': 1, 'status': 'complete', 'methods': selected, 'files': [{'path': str(path), 'sha256': sha256(path)} for path in sorted(output_root.glob('*')) if path.is_file() and path.name != 'analysis_manifest.json'], 'confirmation_accessed': False, 'official_test_accessed': False}
    (output_root / 'analysis_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'methods': selected}, ensure_ascii=False))
if __name__ == '__main__':
    main()
