from __future__ import annotations
import json
from collections import Counter
from pipeline_common import ALIGNMENT, FRIENDS_DIR, LOGS, MANIFESTS, REPORTS, ensure_output_dirs, load_mustard, normalize_speaker, normalize_text, parse_utterance_id, write_csv, write_json
FIELDS = ['friends_utterance_id', 'conversation_id', 'speaker_raw', 'speaker_normalized', 'text_raw', 'text_normalized', 'text_normalized_no_apostrophe', 'reply_to', 'order_in_scene', 'season', 'episode', 'scene']

def main() -> int:
    ensure_output_dirs()
    conversations = json.loads((FRIENDS_DIR / 'conversations.json').read_text(encoding='utf-8'))
    rows = []
    with (FRIENDS_DIR / 'utterances.jsonl').open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            source = json.loads(line)
            parsed = parse_utterance_id(source['id'])
            conversation_id = source['conversation_id']
            if conversation_id not in conversations:
                raise AssertionError(f'Missing conversation metadata at line {line_number}: {conversation_id}')
            meta = conversations[conversation_id]
            expected = (f"s{parsed['season']:02d}", f"e{parsed['episode']:02d}", f"c{parsed['scene']:02d}")
            observed = (meta.get('season'), meta.get('episode'), meta.get('scene'))
            if observed != expected:
                raise AssertionError(f"ID/metadata disagreement for {source['id']}: {expected} != {observed}")
            text = '' if source.get('text') is None else str(source['text'])
            speaker = 'UNKNOWN' if source.get('speaker') is None else str(source['speaker'])
            rows.append({'friends_utterance_id': source['id'], 'conversation_id': conversation_id, 'speaker_raw': speaker, 'speaker_normalized': normalize_speaker(speaker), 'text_raw': text, 'text_normalized': normalize_text(text), 'text_normalized_no_apostrophe': normalize_text(text, keep_apostrophe=False), 'reply_to': source.get('reply-to') or '', 'order_in_scene': parsed['order'], 'season': parsed['season'], 'episode': parsed['episode'], 'scene': parsed['scene']})
    rows.sort(key=lambda r: (int(r['season']), int(r['episode']), int(r['scene']), int(r['order_in_scene'])))
    write_csv(ALIGNMENT / 'friends_index.csv', rows, FIELDS)
    corpus_speakers = Counter((row['speaker_raw'] for row in rows))
    mustard = load_mustard()
    mustard_speakers = Counter((str(row.get('speaker', 'UNKNOWN')) for row in mustard.values() if str(row.get('show', '')).casefold() == 'friends'))
    desired = {'CHANDLER': 'Chandler Bing', 'JOEY': 'Joey Tribbiani', 'MONICA': 'Monica Geller', 'PHOEBE': 'Phoebe Buffay', 'RACHEL': 'Rachel Green', 'ROSS': 'Ross Geller'}
    observed_names = set(corpus_speakers)
    aliases = {source: target for source, target in desired.items() if source in mustard_speakers and target in observed_names}
    unresolved = sorted(set(mustard_speakers) - set(aliases))
    alias_manifest = {'construction': 'Explicit aliases retained only when both the MUStARD name and exact ConvoKit speaker name were observed.', 'aliases': aliases, 'unresolved_mustard_speakers': unresolved, 'mustard_friends_speaker_counts': dict(mustard_speakers), 'convokit_speaker_counts': dict(corpus_speakers)}
    write_json(MANIFESTS / 'speaker_aliases.json', alias_manifest)
    notes = sum((row['speaker_raw'] == 'TRANSCRIPT_NOTE' for row in rows))
    stats = {'utterances': len(rows), 'scenes': len({row['conversation_id'] for row in rows}), 'episodes': len({(row['season'], row['episode']) for row in rows}), 'transcript_notes': notes, 'empty_text': sum((not row['text_normalized'] for row in rows)), 'speakers': len(corpus_speakers), 'unresolved_mustard_speakers': unresolved}
    write_json(REPORTS / '02_friends_index_stats.json', stats)
    summary = f"Friends utterances={len(rows)}; scenes={stats['scenes']}; episodes={stats['episodes']}; notes={notes}; unresolved={unresolved}\n"
    (LOGS / '03_build_friends_index.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
