from __future__ import annotations
from collections import Counter, defaultdict
from pipeline_common import ALIGNMENT, FRIENDS_INDEX, LOGS, ensure_output_dirs, is_transcript_note, read_csv, write_csv
DROP_REASONS = {'SCENE_END', 'NO_DIFFERENT_SPEAKER_RESPONSE', 'TRANSCRIPT_ONLY_AFTER_TARGET', 'AMBIGUOUS_TARGET', 'CONTEXT_MISMATCH', 'SPEAKER_MISMATCH', 'NO_MATCH', 'MEDIA_MISSING', 'OTHER'}
RESPONSE_FIELDS = ['observed_response_id', 'observed_response_text', 'observed_response_speaker', 'observed_response_norm', 'distance_in_utterances', 'skipped_same_speaker_count', 'manual_review_status', 'drop_reason']

def main() -> int:
    ensure_output_dirs()
    alignments = read_csv(ALIGNMENT / '02_alignment_all.csv')
    friends = read_csv(FRIENDS_INDEX)
    by_conversation = defaultdict(list)
    for row in friends:
        by_conversation[row['conversation_id']].append(row)
    for sequence in by_conversation.values():
        sequence.sort(key=lambda r: int(r['order_in_scene']))
    accepted, rejected = ([], [])
    for row in alignments:
        result = dict(row)
        result.update({key: '' for key in RESPONSE_FIELDS})
        alignment_type = row['alignment_type']
        if alignment_type == 'NO_MATCH':
            result['drop_reason'] = 'NO_MATCH'
            rejected.append(result)
            continue
        sequence = by_conversation[row['conversation_id']]
        positions = [i for i, item in enumerate(sequence) if item['friends_utterance_id'] == row['friends_candidate_id']]
        if len(positions) != 1:
            result['drop_reason'] = 'OTHER'
            result['notes'] = (result.get('notes', '') + ';candidate_position_not_unique').strip(';')
            rejected.append(result)
            continue
        position = positions[0]
        skipped_same = 0
        skipped_notes = 0
        response = None
        response_position = None
        for next_position in range(position + 1, len(sequence)):
            item = sequence[next_position]
            if not item['text_normalized'] or is_transcript_note(item):
                skipped_notes += 1
                continue
            if item['speaker_raw'] == row['friends_speaker']:
                skipped_same += 1
                continue
            response = item
            response_position = next_position
            break
        if response is None:
            result['drop_reason'] = 'TRANSCRIPT_ONLY_AFTER_TARGET' if skipped_notes and (not skipped_same) else 'NO_DIFFERENT_SPEAKER_RESPONSE' if skipped_same else 'SCENE_END'
            rejected.append(result)
            continue
        result.update({'observed_response_id': response['friends_utterance_id'], 'observed_response_text': response['text_raw'], 'observed_response_speaker': response['speaker_raw'], 'observed_response_norm': response['text_normalized'], 'distance_in_utterances': response_position - position, 'skipped_same_speaker_count': skipped_same, 'manual_review_status': 'PENDING', 'drop_reason': ''})
        accepted.append(result)
    fields = list(alignments[0].keys()) + RESPONSE_FIELDS
    write_csv(ALIGNMENT / '03_with_observed_response.csv', accepted, fields)
    write_csv(ALIGNMENT / '03_rejected.csv', rejected, fields)
    drops = Counter((row['drop_reason'] for row in rejected))
    unknown = set(drops) - DROP_REASONS
    if unknown:
        raise AssertionError(f'Unknown drop reasons: {unknown}')
    summary = f'Observed responses={len(accepted)}; rejected={len(rejected)}; drops={dict(drops)}\n'
    (LOGS / '05_extract_observed_response.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
