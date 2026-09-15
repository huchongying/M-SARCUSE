from __future__ import annotations
import hashlib
import json
from pathlib import Path
from paper_protocol import ARTIFACT_ROOT, ROOT, require_preflight
METHODS = ['msh-comics', 'mbert-acoustic-rf', 'msnr-msd', 'frozen-early-fusion', 'mo-sarcation']
SEEDS = [20260831, 20260832, 20260833]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def count_jsonl(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return sum((1 for _ in handle))

def main() -> None:
    inputs = require_preflight()
    targets = inputs['fold_counts']
    config_hash = sha256(ROOT / 'config' / 'msarcuse_external_baselines.draft.json')
    checks = []
    for method in METHODS:
        family = 'runs' if method == 'mbert-acoustic-rf' else 'runs-reproducible'
        method_hash = None
        for fold in range(4):
            run_dir = ARTIFACT_ROOT / family / method / f'fold-{fold}'
            summary_path = run_dir / 'run_summary.json'
            summary = read_json(summary_path)
            if summary.get('status') != 'complete' or summary.get('recognizer_episode_overlap') is not False:
                raise ValueError(f'Incomplete or unsafe summary: {method} fold {fold}')
            if method != 'mbert-acoustic-rf' and summary.get('seed_set_before_model_instantiation') is not True:
                raise ValueError(f'Unseeded run: {method} fold {fold}')
            if summary.get('confirmation_accessed') is not False or summary.get('official_test_accessed') is not False:
                raise ValueError(f'Sealed data flag failed: {method} fold {fold}')
            if [x['seed'] for x in summary['seeds']] != SEEDS:
                raise ValueError(f'Seed list changed: {method} fold {fold}')
            if summary['selector_training_targets'] != 1600 or summary['selector_evaluation_targets'] != targets[fold]:
                raise ValueError(f'Target count changed: {method} fold {fold}')
            if method_hash is None:
                method_hash = summary['config_sha256']
            elif method_hash != summary['config_sha256']:
                raise ValueError(f'Within-method config drift: {method}')
            if summary['config_sha256'] != config_hash:
                raise ValueError(f'Run configuration mismatch: {method} fold {fold}')
            for seed in SEEDS:
                rec = run_dir / f'recognition_predictions.seed-{seed}.jsonl'
                scores = run_dir / f'candidate_scores.seed-{seed}.jsonl'
                entry = next((x for x in summary['seeds'] if x['seed'] == seed))
                if sha256(rec) != entry['recognition_predictions_sha256']:
                    raise ValueError(f'Recognition hash mismatch: {rec}')
                if sha256(scores) != entry['candidate_scores_sha256']:
                    raise ValueError(f'Score hash mismatch: {scores}')
                if count_jsonl(scores) != targets[fold] * 4 * 2:
                    raise ValueError(f'Score row count mismatch: {scores}')
            checks.append({'method': method, 'fold': fold, 'run_family': family, 'summary_sha256': sha256(summary_path), 'config_sha256': summary['config_sha256']})
    analysis_root = ARTIFACT_ROOT / 'analysis-reproducible'
    required = [analysis_root / f'{method}.metrics.json' for method in METHODS]
    required += [analysis_root / f'{method}.response_outcomes.csv' for method in METHODS]
    required += [analysis_root / 'unified_comparison.csv', analysis_root / 'analysis_manifest.json']
    if any((not path.is_file() for path in required)):
        raise ValueError('Final analysis artifact is missing')
    manifest = read_json(analysis_root / 'analysis_manifest.json')
    if manifest.get('status') != 'complete' or sorted(manifest['methods']) != sorted(METHODS):
        raise ValueError('Analysis method coverage mismatch')
    listed = {Path(entry['path']).name: entry for entry in manifest['files']}
    if len(listed) != len(manifest['files']):
        raise ValueError('Duplicate analysis manifest entry')
    for path in required:
        if path.name == 'analysis_manifest.json':
            continue
        if path.name not in listed or sha256(path) != listed[path.name]['sha256']:
            raise ValueError(f'Analysis artifact hash mismatch: {path}')
    for method in METHODS:
        result = read_json(analysis_root / f'{method}.metrics.json')
        if (result['method'] != method or result['pooling'] != 'micro'
                or result['bootstrap']['resamples'] != 2000
                or result['bootstrap']['clustered_by'] != 'episode'
                or result['bootstrap']['paired_episode_weights'] is not True):
            raise ValueError(f'Analysis protocol mismatch: {method}')
        for entry in result['raw_provenance']:
            for key in ('recognition', 'scores'):
                if sha256(Path(entry[key])) != entry[f'{key}_sha256']:
                    raise ValueError(f'Analysis raw input changed: {method}')
    completion = {'schema_version': 1, 'status': 'complete', 'methods': METHODS, 'folds_per_method': 4, 'seeds': SEEDS, 'total_response_targets_per_seed_method': sum(targets.values()), 'checks': checks, 'analysis_files': [{'path': str(path), 'sha256': sha256(path)} for path in required], 'confirmation_accessed': False, 'official_test_accessed': False}
    output = ARTIFACT_ROOT / 'pipeline_completion.json'
    output.write_text(json.dumps(completion, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'complete', 'output': str(output)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
