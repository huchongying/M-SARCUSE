from __future__ import annotations
import csv
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parent.parent
SOURCE = PACKAGE_ROOT / 'source'
ALIGNMENT = PACKAGE_ROOT / 'alignment'
REPORTS = PACKAGE_ROOT / 'reports'
LOGS = PACKAGE_ROOT / 'logs'
MANIFESTS = PACKAGE_ROOT / 'provenance' / 'manifests'
MUSTARD_JSON = SOURCE / 'MUStARD' / 'sarcasm_data.json'
FRIENDS_DIR = SOURCE / 'Friends' / 'friends-corpus'
FRIENDS_ARCHIVE = SOURCE / 'Friends' / 'friends-corpus.zip'
FRIENDS_INDEX = ALIGNMENT / 'friends_index.csv'
UTTERANCE_ID_RE = re.compile('^s(?P<season>\\d{2})_e(?P<episode>\\d{2})_c(?P<scene>\\d{2})_u(?P<order>\\d+)$')

def ensure_output_dirs() -> None:
    for path in (SOURCE, ALIGNMENT, REPORTS, LOGS, MANIFESTS):
        path.mkdir(parents=True, exist_ok=True)

def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise

def write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def load_mustard() -> dict[str, dict]:
    with MUSTARD_JSON.open('r', encoding='utf-8') as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError('MUStARD metadata must be an object keyed by mustard_id')
    return value

def normalize_text(text: object, keep_apostrophe: bool=True) -> str:
    value = unicodedata.normalize('NFKC', '' if text is None else str(text)).lower()
    value = value.translate(str.maketrans({'’': "'", '‘': "'", '`': "'", '“': '"', '”': '"'}))
    pattern = "[^\\w\\s']" if keep_apostrophe else '[^\\w\\s]'
    value = re.sub(pattern, ' ', value, flags=re.UNICODE)
    return ' '.join(value.split())

def normalize_speaker(value: object) -> str:
    return normalize_text(value, keep_apostrophe=False)

def parse_utterance_id(value: str) -> dict[str, int]:
    match = UTTERANCE_ID_RE.fullmatch(value)
    if not match:
        raise ValueError(f'Unexpected Friends utterance id: {value}')
    return {key: int(number) for key, number in match.groupdict().items()}

def is_transcript_note(row: dict[str, str]) -> bool:
    return normalize_speaker(row.get('speaker_raw', '')) == 'transcript_note'

def json_cell(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def parse_json_cell(value: str, default: object) -> object:
    if not value:
        return default
    return json.loads(value)

def pct(part: int | float, whole: int | float) -> float:
    return 100.0 * part / whole if whole else 0.0

def count_markdown(counter: Counter) -> str:
    return '\n'.join((f'- `{key}`: {count}' for key, count in sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))))
