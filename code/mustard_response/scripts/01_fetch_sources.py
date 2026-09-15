from __future__ import annotations
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from pipeline_common import FRIENDS_ARCHIVE, FRIENDS_DIR, LOGS, MANIFESTS, MUSTARD_JSON, PACKAGE_ROOT, SOURCE, ensure_output_dirs
MUSTARD_REPO = 'https://github.com/soujanyaporia/MUStARD.git'
FRIENDS_URL = 'https://zissou.infosci.cornell.edu/convokit/datasets/friends-corpus/friends-corpus.zip'

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def git(*args: str, cwd: Path | None=None) -> str:
    result = subprocess.run(['git', *args], cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip()

def main() -> int:
    ensure_output_dirs()
    mustard_dir = SOURCE / 'MUStARD_official'
    if MUSTARD_JSON.is_file():
        mustard_json = MUSTARD_JSON
        existing_manifest = MANIFESTS / 'source_manifest.json'
        if existing_manifest.is_file():
            existing = json.loads(existing_manifest.read_text(encoding='utf-8'))
            mustard_revision = existing.get('sources', {}).get('mustard', {}).get('revision', 'bundled')
        else:
            mustard_revision = 'bundled'
    else:
        if not mustard_dir.exists():
            git('clone', '--depth', '1', MUSTARD_REPO, str(mustard_dir))
        remote = git('remote', 'get-url', 'origin', cwd=mustard_dir)
        if remote.rstrip('/').removesuffix('.git') != MUSTARD_REPO.rstrip('/').removesuffix('.git'):
            raise RuntimeError(f'Unexpected MUStARD remote: {remote}')
        mustard_revision = git('rev-parse', 'HEAD', cwd=mustard_dir)
        source_metadata = mustard_dir / 'data' / 'sarcasm_data.json'
        if not source_metadata.is_file():
            raise FileNotFoundError(source_metadata)
        MUSTARD_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_metadata, MUSTARD_JSON)
        mustard_json = MUSTARD_JSON
    FRIENDS_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not FRIENDS_ARCHIVE.exists():
        with urllib.request.urlopen(FRIENDS_URL, timeout=120) as response, FRIENDS_ARCHIVE.open('wb') as output:
            shutil.copyfileobj(response, output)
    if not FRIENDS_DIR.is_dir():
        shutil.unpack_archive(FRIENDS_ARCHIVE, FRIENDS_ARCHIVE.parent)
    if not FRIENDS_DIR.is_dir():
        raise RuntimeError(f'Archive did not create {FRIENDS_DIR}')
    files = sorted((path for path in FRIENDS_DIR.rglob('*') if path.is_file()))
    relative = lambda path: str(path.relative_to(PACKAGE_ROOT))
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version, 'sources': {'mustard': {'url': MUSTARD_REPO, 'revision': mustard_revision, 'license_file': relative(mustard_dir / 'LICENSE') if (mustard_dir / 'LICENSE').is_file() else '', 'metadata_path': relative(mustard_json), 'metadata_sha256': sha256(mustard_json), 'media_downloaded': False}, 'friends_convokit': {'url': FRIENDS_URL, 'archive_path': relative(FRIENDS_ARCHIVE), 'archive_bytes': FRIENDS_ARCHIVE.stat().st_size, 'archive_sha256': sha256(FRIENDS_ARCHIVE), 'extracted_path': relative(FRIENDS_DIR), 'extracted_file_count': len(files), 'license': 'Apache-2.0 (per ConvoKit Friends Corpus documentation)'}}}
    out = MANIFESTS / 'source_manifest.json'
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    summary = f'MUStARD revision: {mustard_revision}\nMUStARD metadata: {relative(mustard_json)}\nFriends archive: {relative(FRIENDS_ARCHIVE)} ({FRIENDS_ARCHIVE.stat().st_size} bytes)\nFriends extracted files: {len(files)}\n'
    (LOGS / '01_fetch_sources.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
