from __future__ import annotations
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from pipeline_common import ALIGNMENT, LOGS, MANIFESTS, PACKAGE_ROOT, REPORTS, SOURCE, ensure_output_dirs
ALIGNMENT_CSV = ALIGNMENT / '03_with_observed_response.csv'
REVIEW_CSV = ALIGNMENT / '04_manual_review.csv'
MEDIA_ZIP = SOURCE / 'MUStARD' / 'mmsd_raw_data.zip'
FFMPEG = Path(shutil.which('ffmpeg') or 'ffmpeg')
AUDIO_DIR = SOURCE / 'MUStARD' / 'audio'
OUTPUT_CSV = ALIGNMENT / '05_media_availability.csv'
VALID_REVIEW_CSV = ALIGNMENT / 'media_valid_manual_review.csv'
REPORT_JSON = REPORTS / '05_media_availability_report.json'
REPORT_MD = REPORTS / '05_media_availability_report.md'
MANIFEST_JSON = MANIFESTS / 'media_source_manifest.json'
LOG_PATH = LOGS / '09_audit_official_media.log'
EXPECTED_INPUT_ROWS = 258
EXPECTED_ZIP_BYTES = 1403183019
EXPECTED_ZIP_SHA256 = 'A2E68C48B7D13C66AF98900DC5119B54B3B0BF962C228D99069CB0930C3880D5'
OFFICIAL_GITHUB_REVISION = 'b34b5134beeec42a36cdac79330637de9cc9894e'
OFFICIAL_MEDIA_REVISION = 'c1f7c88cf6caa9beb67a5b8550436e20a911bace'
OFFICIAL_MEDIA_URL = f'https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/{OFFICIAL_MEDIA_REVISION}/mmsd_raw_data.zip'
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
OUTPUT_FIELDS = ['mustard_id', 'video_found', 'video_path_or_source_reference', 'audio_extractable', 'duration_sec', 'audio_channels', 'sample_rate_before_conversion', 'file_size', 'sha256', 'failure_reason']

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise RuntimeError(f'Missing CSV header: {path}')
        return (list(reader.fieldnames), list(reader))

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)

def probe_video(ffmpeg: Path, video: Path) -> tuple[float | None, int | None, int | None, str]:
    process = subprocess.run([str(ffmpeg), '-hide_banner', '-i', str(video), '-map', '0:a:0', '-f', 'null', '-'], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', check=False)
    diagnostic = process.stderr
    duration_match = re.search('Duration:\\s*(\\d+):(\\d+):(\\d+(?:\\.\\d+)?)', diagnostic)
    audio_lines = [line for line in diagnostic.splitlines() if 'Audio:' in line]
    if not audio_lines:
        return (None, None, None, 'NO_AUDIO_STREAM')
    audio_line = audio_lines[0]
    rate_match = re.search('(?:,\\s*)(\\d+)\\s*Hz(?:,|\\s)', audio_line)
    if not duration_match or not rate_match:
        return (None, None, None, 'PROBE_FAILED')
    hours, minutes, seconds = duration_match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    sample_rate = int(rate_match.group(1))
    channel_layout = audio_line[rate_match.end():].split(',', 1)[0].strip().lower()
    layout_channels = {'mono': 1, 'stereo': 2, '2.1': 3, '3.0': 3, '3.0(back)': 3, '4.0': 4, 'quad': 4, 'quad(side)': 4, '5.0': 5, '5.0(side)': 5, '5.1': 6, '5.1(side)': 6, '6.0': 6, '6.1': 7, '7.0': 7, '7.1': 8, '7.1(wide)': 8}
    channels = layout_channels.get(channel_layout)
    if channels is None:
        channel_match = re.search('\\b(\\d+)\\s+channels?\\b', audio_line, flags=re.IGNORECASE)
        channels = int(channel_match.group(1)) if channel_match else None
    if channels is None:
        return (None, None, None, 'PROBE_FAILED')
    return (duration, channels, sample_rate, '')

def extract_audio(ffmpeg: Path, video: Path, output: Path) -> tuple[bool, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.tmp.wav')
    process = subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(temp)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', check=False)
    if process.returncode != 0 or not temp.exists():
        temp.unlink(missing_ok=True)
        message = process.stderr.strip().replace('\r', ' ').replace('\n', ' ')
        return (False, f'AUDIO_CONVERSION_FAILED:{message[:240]}')
    try:
        with wave.open(str(temp), 'rb') as audio:
            valid = audio.getnchannels() == 1 and audio.getframerate() == 16000 and (audio.getsampwidth() == 2) and (audio.getnframes() > 0)
    except (wave.Error, EOFError):
        valid = False
    if not valid:
        temp.unlink(missing_ok=True)
        return (False, 'OUTPUT_VALIDATION_FAILED')
    temp.replace(output)
    return (True, '')

def validate_inputs(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_INPUT_ROWS:
        raise RuntimeError(f'Expected {EXPECTED_INPUT_ROWS} input rows, found {len(rows)}')
    ids = [row['mustard_id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError('Input mustard_id values are not unique')
    for field in ('alignment_type', 'sarcasm_label', 'episode', 'scene', 'observed_response_text'):
        if any((not (row.get(field) or '').strip() for row in rows)):
            raise RuntimeError(f'Missing required input field: {field}')
    if any((not row['observed_response_text'].strip() for row in rows)):
        raise RuntimeError('Input contains an empty observed response')

def main() -> int:
    ensure_output_dirs()
    for path in (ALIGNMENT_CSV, REVIEW_CSV, MEDIA_ZIP):
        if not path.exists():
            raise FileNotFoundError(path)
    if not FFMPEG.is_file():
        raise FileNotFoundError('ffmpeg')
    if MEDIA_ZIP.stat().st_size != EXPECTED_ZIP_BYTES:
        raise RuntimeError(f'Official media ZIP size mismatch: {MEDIA_ZIP.stat().st_size} != {EXPECTED_ZIP_BYTES}')
    media_zip_sha256 = sha256_file(MEDIA_ZIP)
    if media_zip_sha256 != EXPECTED_ZIP_SHA256:
        raise RuntimeError(f'Official media ZIP SHA-256 mismatch: {media_zip_sha256}')
    _, rows = read_csv(ALIGNMENT_CSV)
    validate_inputs(rows)
    by_id = {row['mustard_id']: row for row in rows}
    with zipfile.ZipFile(MEDIA_ZIP, 'r') as archive:
        video_entries: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in archive.infolist():
            path = Path(info.filename)
            if not info.is_dir() and path.suffix.lower() in VIDEO_EXTENSIONS:
                video_entries[path.stem].append(info)
        results: list[dict[str, object]] = []
        details: dict[str, dict[str, object]] = {}
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='mustard_media_', dir=PACKAGE_ROOT) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            for index, mustard_id in enumerate(by_id, start=1):
                candidates = video_entries.get(mustard_id, [])
                base = {'mustard_id': mustard_id, 'video_found': 'NO', 'video_path_or_source_reference': 'UNKNOWN', 'audio_extractable': 'NO', 'duration_sec': 'UNKNOWN', 'audio_channels': 'UNKNOWN', 'sample_rate_before_conversion': 'UNKNOWN', 'file_size': 'UNKNOWN', 'sha256': 'UNKNOWN', 'failure_reason': 'VIDEO_NOT_FOUND'}
                if len(candidates) > 1:
                    preferred = [info for info in candidates if 'utterances_final' in info.filename]
                    candidates = preferred if len(preferred) == 1 else candidates
                if len(candidates) != 1:
                    if len(candidates) > 1:
                        base['video_found'] = 'YES'
                        base['video_path_or_source_reference'] = ' | '.join((f'zip://{MEDIA_ZIP}!/{info.filename}' for info in candidates))
                        base['failure_reason'] = 'MULTIPLE_VIDEO_CANDIDATES'
                    results.append(base)
                    details[mustard_id] = {**base, 'audio_path': 'UNKNOWN'}
                    continue
                info = candidates[0]
                base['video_found'] = 'YES'
                base['video_path_or_source_reference'] = f'zip://{MEDIA_ZIP}!/{info.filename}'
                base['file_size'] = info.file_size
                temp_video = temp_dir / f'{mustard_id}{Path(info.filename).suffix.lower()}'
                try:
                    video_digest = hashlib.sha256()
                    with archive.open(info, 'r') as source, temp_video.open('wb') as destination:
                        while (chunk := source.read(1024 * 1024)):
                            destination.write(chunk)
                            video_digest.update(chunk)
                    base['sha256'] = video_digest.hexdigest().upper()
                except (OSError, zipfile.BadZipFile) as exc:
                    base['failure_reason'] = f'VIDEO_EXTRACT_FAILED:{type(exc).__name__}'
                    results.append(base)
                    details[mustard_id] = {**base, 'audio_path': 'UNKNOWN'}
                    temp_video.unlink(missing_ok=True)
                    continue
                duration, channels, sample_rate, probe_failure = probe_video(FFMPEG, temp_video)
                if probe_failure:
                    base['failure_reason'] = probe_failure
                    results.append(base)
                    details[mustard_id] = {**base, 'audio_path': 'UNKNOWN'}
                    temp_video.unlink(missing_ok=True)
                    continue
                base['duration_sec'] = f'{duration:.3f}'
                base['audio_channels'] = channels
                base['sample_rate_before_conversion'] = sample_rate
                audio_path = AUDIO_DIR / f'{mustard_id}.wav'
                success, failure = extract_audio(FFMPEG, temp_video, audio_path)
                temp_video.unlink(missing_ok=True)
                if success:
                    base['audio_extractable'] = 'YES'
                    base['failure_reason'] = ''
                    audio_reference = str(audio_path)
                    audio_sha256 = sha256_file(audio_path)
                else:
                    base['failure_reason'] = failure
                    audio_reference = 'UNKNOWN'
                    audio_sha256 = 'UNKNOWN'
                results.append(base)
                details[mustard_id] = {**base, 'audio_path': audio_reference, 'audio_sha256': audio_sha256}
                if index % 25 == 0:
                    print(f'processed={index}/{len(by_id)}', flush=True)
    write_csv(OUTPUT_CSV, OUTPUT_FIELDS, results)
    valid_ids = {row['mustard_id'] for row in results if row['audio_extractable'] == 'YES'}
    video_ids = {row['mustard_id'] for row in results if row['video_found'] == 'YES'}
    alignment_counts: dict[str, dict[str, object]] = {}
    for alignment_type in sorted({row['alignment_type'] for row in rows}):
        group = [row for row in rows if row['alignment_type'] == alignment_type]
        group_video = sum((row['mustard_id'] in video_ids for row in group))
        group_audio = sum((row['mustard_id'] in valid_ids for row in group))
        alignment_counts[alignment_type] = {'total': len(group), 'video_found': group_video, 'video_coverage': group_video / len(group), 'audio_extractable': group_audio, 'audio_coverage': group_audio / len(group)}
    valid_source_rows = [row for row in rows if row['mustard_id'] in valid_ids]
    label_counts = Counter((row['sarcasm_label'] for row in valid_source_rows))
    failure_counts = Counter((row['failure_reason'] or 'NONE' for row in results))
    report = {'status': 'PASS' if len(valid_ids) >= 150 else 'AUDIO_BELOW_150_STOP', 'input_alignment_rows': len(rows), 'valid_video': len(video_ids), 'audio_extractable': len(valid_ids), 'final_alignment_observed_response_audio': len(valid_ids), 'alignment_type_media_coverage': alignment_counts, 'unique_episodes': len({(row['season'], row['episode']) for row in valid_source_rows}), 'unique_scenes': len({(row['season'], row['episode'], row['scene']) for row in valid_source_rows}), 'sarcasm_label_counts': dict(sorted(label_counts.items())), 'failure_reasons': dict(sorted(failure_counts.items())), 'threshold': 150, 'manual_review_policy': 'media_valid_manual_review.csv generated; no human review performed' if len(valid_ids) >= 150 else 'stop subsequent human review; media_valid_manual_review.csv not generated'}
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    coverage_lines = '\n'.join((f"- {name}: video {value['video_found']}/{value['total']} ({value['video_coverage']:.2%}); audio {value['audio_extractable']}/{value['total']} ({value['audio_coverage']:.2%})" for name, value in alignment_counts.items()))
    REPORT_MD.write_text(f"# MUStARD official-media availability audit\n\n- Input aligned targets: {len(rows)}\n- Valid official target videos: {len(video_ids)}\n- Successfully extracted 16 kHz mono PCM16 WAV: {len(valid_ids)}\n- Alignment + observed response + audio: {len(valid_ids)}\n- Unique episodes: {report['unique_episodes']}\n- Sarcasm labels: {json.dumps(dict(sorted(label_counts.items())), ensure_ascii=False)}\n- Decision: {report['status']}\n\n## Coverage by alignment type\n\n{coverage_lines}\n\n## Failure reasons\n\n```json\n{json.dumps(dict(sorted(failure_counts.items())), ensure_ascii=False, indent=2)}\n```\n", encoding='utf-8')
    manifest = {'official_github_repository': 'https://github.com/soujanyaporia/MUStARD.git', 'official_github_revision': OFFICIAL_GITHUB_REVISION, 'official_readme_media_link': 'MichiganNLP/MUStARD mmsd_raw_data.zip', 'official_media_repository': 'https://huggingface.co/datasets/MichiganNLP/MUStARD', 'official_media_revision': OFFICIAL_MEDIA_REVISION, 'official_media_url': OFFICIAL_MEDIA_URL, 'media_zip': str(MEDIA_ZIP), 'media_zip_bytes': MEDIA_ZIP.stat().st_size, 'media_zip_sha256': media_zip_sha256, 'ffmpeg': str(FFMPEG), 'ffmpeg_sha256': sha256_file(FFMPEG), 'conversion': 'ffmpeg -map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le', 'audio_files': [{'mustard_id': mustard_id, 'path': details[mustard_id]['audio_path'], 'sha256': details[mustard_id].get('audio_sha256', 'UNKNOWN')} for mustard_id in sorted(valid_ids)]}
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if len(valid_ids) >= 150:
        review_fields, review_rows = read_csv(REVIEW_CSV)
        valid_review_rows = []
        for row in review_rows:
            if row['mustard_id'] in valid_ids:
                valid_review_rows.append({**row, 'target_media_id': row['mustard_id'], 'target_audio_path': details[row['mustard_id']]['audio_path'], 'target_audio_sha256': details[row['mustard_id']]['audio_sha256']})
        write_csv(VALID_REVIEW_CSV, [*review_fields, 'target_media_id', 'target_audio_path', 'target_audio_sha256'], valid_review_rows)
    elif VALID_REVIEW_CSV.exists():
        raise RuntimeError('Audio count is below 150 but a stale media_valid_manual_review.csv exists; refusing to delete it')
    log = {'status': report['status'], 'input_rows': len(rows), 'output_rows': len(results), 'video_found': len(video_ids), 'audio_extractable': len(valid_ids), 'output_csv_sha256': sha256_file(OUTPUT_CSV), 'report_json_sha256': sha256_file(REPORT_JSON), 'manifest_sha256': sha256_file(MANIFEST_JSON)}
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise
