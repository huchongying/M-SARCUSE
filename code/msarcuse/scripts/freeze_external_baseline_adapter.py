from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from paper_protocol import ARTIFACT_ROOT, ROOT, require_preflight

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('--run-family', default='runs')
    parser.add_argument('--freeze-suffix', default='')
    args = parser.parse_args()
    require_preflight()
    run_dir = ARTIFACT_ROOT / args.run_family / args.method / 'fold-0'
    summary_path = run_dir / 'run_summary.json'
    summary = read_json(summary_path)
    if summary.get('status') != 'complete' or summary.get('fold') != 0:
        raise ValueError('A complete fold-0 run is required')
    if summary.get('recognizer_episode_overlap') is not False:
        raise ValueError('Recognizer episode overlap detected')
    if summary.get('confirmation_accessed') is not False:
        raise ValueError('Confirmation access detected')
    if summary.get('official_test_accessed') is not False:
        raise ValueError('Official-test access detected')
    expected = int(summary['selector_evaluation_targets']) * 4 * 2
    if any((int(seed['candidate_score_rows']) != expected for seed in summary['seeds'])):
        raise ValueError('Fold-0 candidate score matrix is incomplete')
    config_path = ROOT / 'config' / 'msarcuse_external_baselines.draft.json'
    freeze_dir = ARTIFACT_ROOT / 'freeze' / f'{args.method}{args.freeze_suffix}'
    freeze_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = freeze_dir / 'adapter_config.snapshot.json'
    runner_snapshot = freeze_dir / args.runner.name
    shutil.copy2(config_path, config_snapshot)
    shutil.copy2(args.runner.resolve(), runner_snapshot)
    raw_files = []
    for path in sorted(run_dir.glob('*')):
        if path.is_file():
            raw_files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    manifest = {'schema_version': 1, 'status': 'frozen_after_complete_fold0_integrity_pass', 'method': args.method, 'scientific_result_used_for_selection': False, 'config_snapshot': {'path': str(config_snapshot), 'sha256': sha256(config_snapshot)}, 'runner_snapshot': {'path': str(runner_snapshot), 'sha256': sha256(runner_snapshot)}, 'fold0_summary': {'path': str(summary_path), 'sha256': sha256(summary_path)}, 'fold0_files': raw_files, 'confirmation_accessed': False, 'official_test_accessed': False, 'no_result_driven_replacement': True}
    manifest_path = freeze_dir / 'freeze_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'frozen', 'manifest': str(manifest_path)}, ensure_ascii=False))
if __name__ == '__main__':
    main()
