from __future__ import annotations
import json
import statistics
from collections import Counter
from pipeline_common import FRIENDS_DIR, LOGS, MUSTARD_JSON, REPORTS, ensure_output_dirs, load_mustard, write_json
REQUIRED = ('utterance', 'speaker', 'context', 'context_speakers', 'show', 'sarcasm')

def main() -> int:
    ensure_output_dirs()
    data = load_mustard()
    shows = Counter((str(row.get('show', 'UNKNOWN')) for row in data.values()))
    friends_labels = [name for name in shows if name.casefold() == 'friends']
    if len(friends_labels) != 1:
        raise RuntimeError(f'Expected one observed Friends label, found {friends_labels}')
    friends_label = friends_labels[0]
    rows = [(key, row) for key, row in data.items() if row.get('show') == friends_label]
    labels = Counter(('sarcastic' if row.get('sarcasm') is True else 'literal' if row.get('sarcasm') is False else 'UNKNOWN' for _, row in rows))
    speakers = Counter((str(row.get('speaker', 'UNKNOWN')) for _, row in rows))
    context_lengths = [len(row.get('context', [])) if isinstance(row.get('context'), list) else 0 for _, row in rows]
    missing = {field: sum((field not in row or row[field] is None or row[field] == '' for row in data.values())) for field in REQUIRED}
    schema_counts = Counter((key for row in data.values() for key in row))
    media = {'raw_video_or_audio_files_downloaded': 0, 'trackable_ids': len(data), 'official_audio_feature_file_present': (MUSTARD_JSON.parent / 'audio_features.p').is_file(), 'status': 'TRACKABLE_ID_ONLY', 'note': 'Raw audiovisual clips were not downloaded in this feasibility stage; mustard_id remains the traceable media identifier.'}
    stats = {'mustard_total': len(data), 'show_counts': dict(shows), 'friends_show_label': friends_label, 'friends_targets': len(rows), 'friends_labels': dict(labels), 'friends_speakers': dict(speakers), 'context_length': {'min': min(context_lengths), 'max': max(context_lengths), 'mean': statistics.mean(context_lengths), 'median': statistics.median(context_lengths), 'distribution': dict(Counter(context_lengths))}, 'missing_fields_all_mustard': missing, 'field_presence_all_mustard': dict(schema_counts), 'media': media, 'friends_source_file': str(MUSTARD_JSON), 'friends_corpus_files_present': sorted((path.name for path in FRIENDS_DIR.iterdir())) if FRIENDS_DIR.exists() else []}
    write_json(REPORTS / '01_source_stats.json', stats)
    md = f"# Source inspection\n\n- MUStARD total: **{len(data)}**\n- Observed show values: {dict(shows)}\n- Friends label: **{friends_label}**\n- Friends targets: **{len(rows)}**\n- Friends labels: {dict(labels)}\n- Friends speakers: {dict(speakers)}\n- Context length: min {min(context_lengths)}, median {statistics.median(context_lengths)}, mean {statistics.mean(context_lengths):.2f}, max {max(context_lengths)}\n- Missing fields across all MUStARD records: {missing}\n- Media status: **{media['status']}**; raw media were not downloaded.\n"
    (REPORTS / '01_source_stats.md').write_text(md, encoding='utf-8')
    summary = f'MUStARD={len(data)}; Friends={len(rows)}; labels={dict(labels)}; speakers={dict(speakers)}\n'
    (LOGS / '02_inspect_sources.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
