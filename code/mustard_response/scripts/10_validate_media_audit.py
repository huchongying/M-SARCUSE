from __future__ import annotations
import csv
import hashlib
import json
import wave
from pathlib import Path
from pipeline_common import ALIGNMENT as ALIGNMENT_ROOT, LOGS, MANIFESTS, REPORTS, SOURCE
ALIGNMENT = ALIGNMENT_ROOT / '03_with_observed_response.csv'
AVAILABILITY = ALIGNMENT_ROOT / '05_media_availability.csv'
VALID_REVIEW = ALIGNMENT_ROOT / 'media_valid_manual_review.csv'
MANIFEST = MANIFESTS / 'media_source_manifest.json'
REPORT = REPORTS / '05_media_availability_report.json'
LOG = LOGS / '10_validate_media_audit.json'

def rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()

def main() -> None:
    source = rows(ALIGNMENT)
    availability = rows(AVAILABILITY)
    review = rows(VALID_REVIEW)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    source_ids = {row['mustard_id'] for row in source}
    availability_ids = {row['mustard_id'] for row in availability}
    review_ids = {row['mustard_id'] for row in review}
    assert len(source) == len(source_ids) == 258
    assert len(availability) == len(availability_ids) == 258
    assert len(review) == len(review_ids)
    assert source_ids == availability_ids
    valid = [row for row in availability if row['video_found'] == 'YES'
             and row['audio_extractable'] == 'YES' and not row['failure_reason']]
    valid_ids = {row['mustard_id'] for row in valid}
    assert review_ids == valid_ids
    assert all((float(row['duration_sec']) > 0 for row in valid))
    assert all((int(row['audio_channels']) >= 1 for row in valid))
    assert all((int(row['sample_rate_before_conversion']) > 0 for row in valid))
    assert all((int(row['file_size']) > 0 for row in valid))
    assert all((len(row['sha256']) == 64 for row in valid))
    assert all(('mmsd_raw_data.zip!/utterances_final/' in row['video_path_or_source_reference'] for row in valid))
    manifest_audio = {item['mustard_id']: item for item in manifest['audio_files']}
    assert len(manifest_audio) == len(manifest['audio_files'])
    assert set(manifest_audio) == valid_ids
    total_audio_bytes = 0
    for mustard_id, item in manifest_audio.items():
        audio_path = Path(item['path'])
        assert audio_path == SOURCE / 'MUStARD' / 'audio' / f'{mustard_id}.wav'
        assert audio_path.is_file()
        assert sha256(audio_path) == item['sha256']
        total_audio_bytes += audio_path.stat().st_size
        with wave.open(str(audio_path), 'rb') as audio:
            assert audio.getnchannels() == 1
            assert audio.getframerate() == 16000
            assert audio.getsampwidth() == 2
            assert audio.getnframes() > 0
    for row in review:
        assert row['target_media_id'] == row['mustard_id']
        assert Path(row['target_audio_path']).is_file()
        assert sha256(Path(row['target_audio_path'])) == row['target_audio_sha256']
    assert report['status'] == 'PASS'
    assert report['input_alignment_rows'] == 258
    assert report['valid_video'] == sum(row['video_found'] == 'YES' for row in availability)
    assert report['audio_extractable'] == sum(row['audio_extractable'] == 'YES' for row in availability)
    assert report['final_alignment_observed_response_audio'] == len(valid)
    assert report['unique_episodes'] == len({(row['season'], row['episode']) for row in review})
    assert report['unique_scenes'] == len({(row['season'], row['episode'], row['scene']) for row in review})
    output = {'status': 'PASS', 'validated_rows': len(source), 'validated_wav_files': len(manifest_audio), 'wav_format': {'sample_rate': 16000, 'channels': 1, 'sample_width_bytes': 2}, 'total_audio_bytes': total_audio_bytes, 'availability_csv_sha256': sha256(AVAILABILITY), 'valid_review_csv_sha256': sha256(VALID_REVIEW), 'manifest_sha256': sha256(MANIFEST), 'report_sha256': sha256(REPORT)}
    LOG.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))
if __name__ == '__main__':
    main()
