from __future__ import annotations
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pipeline_common import ALIGNMENT, FRIENDS_INDEX, LOGS, MANIFESTS, ensure_output_dirs, json_cell, load_mustard, normalize_text, read_csv, write_csv, write_json
TARGET_HIGH = 0.95
TARGET_REASONABLE = 0.8
CONTEXT1_HIGH = 0.9
CONTEXT2_HIGH = 0.8
BACKWARD_WINDOW = 8
FIELDS = ['mustard_id', 'sarcasm_label', 'mustard_target_raw', 'mustard_target_norm', 'mustard_target_norm_no_apostrophe', 'mustard_speaker', 'mustard_context_raw', 'mustard_context_norm', 'mustard_context_last2', 'mustard_context_speakers_last2', 'target_media_id', 'target_media_path', 'friends_candidate_id', 'friends_utterance_id', 'conversation_id', 'friends_target_raw', 'friends_target_norm', 'friends_speaker', 'season', 'episode', 'scene', 'friends_episode', 'friends_scene', 'order_in_scene', 'friends_prev2_id', 'friends_prev2', 'friends_prev1_id', 'friends_prev1', 'target_similarity', 'context1_similarity', 'context2_similarity', 'speaker_match', 'alignment_type', 'alignment_score', 'num_candidate_matches', 'num_high_confidence_matches', 'auto_status', 'notes']

def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()

def local_context(candidate: dict, context: list[str], context_speakers: list[str], aliases: dict[str, str], by_conversation: dict[str, list[dict]]) -> tuple[float, float, list[dict]]:
    sequence = by_conversation[candidate['conversation_id']]
    position = next((i for i, row in enumerate(sequence) if row['friends_utterance_id'] == candidate['friends_utterance_id']))
    prior = [row for row in sequence[max(0, position - BACKWARD_WINDOW):position] if row['speaker_raw'] != 'TRANSCRIPT_NOTE' and row['text_normalized']]
    immediate = prior[-2:]
    if not context:
        return (0.0, 0.0, immediate)

    def best_before(end: int, raw: str, speaker: str | None) -> tuple[float, int]:
        expected = aliases.get(speaker or '')
        options = []
        for idx, row in enumerate(prior[:end]):
            if expected is not None and row['speaker_raw'] != expected:
                continue
            options.append((similarity(normalize_text(raw), row['text_normalized']), idx))
        return max(options, default=(0.0, -1))
    last_speaker = context_speakers[-1] if len(context_speakers) >= 1 else None
    context1, index1 = best_before(len(prior), context[-1], last_speaker)
    context2 = 0.0
    if len(context) >= 2 and index1 >= 0:
        second_speaker = context_speakers[-2] if len(context_speakers) >= 2 else None
        context2, _ = best_before(index1, context[-2], second_speaker)
    return (context1, context2, immediate)

def main() -> int:
    ensure_output_dirs()
    friends = read_csv(FRIENDS_INDEX)
    alias_manifest = json.loads((MANIFESTS / 'speaker_aliases.json').read_text(encoding='utf-8'))
    aliases = alias_manifest['aliases']
    by_speaker = defaultdict(list)
    by_exact = defaultdict(list)
    by_token = defaultdict(set)
    by_conversation = defaultdict(list)
    id_to_row = {}
    for row in friends:
        by_speaker[row['speaker_raw']].append(row)
        by_exact[row['text_normalized']].append(row)
        for token in set(row['text_normalized'].split()):
            by_token[token].add(row['friends_utterance_id'])
        by_conversation[row['conversation_id']].append(row)
        id_to_row[row['friends_utterance_id']] = row
    for sequence in by_conversation.values():
        sequence.sort(key=lambda r: int(r['order_in_scene']))
    outputs = []
    for mustard_id, source in load_mustard().items():
        if str(source.get('show', '')).casefold() != 'friends':
            continue
        target_raw = str(source.get('utterance', ''))
        target_norm = normalize_text(target_raw)
        target_no_apostrophe = normalize_text(target_raw, keep_apostrophe=False)
        mustard_speaker = str(source.get('speaker', 'UNKNOWN'))
        mapped_speaker = aliases.get(mustard_speaker)
        context = list(source.get('context') or [])
        context_speakers = list(source.get('context_speakers') or [])
        exact = by_exact.get(target_norm, [])
        if exact:
            candidate_pool = exact
        else:
            pool = by_speaker.get(mapped_speaker, friends) if mapped_speaker else friends
            token_ids = set()
            for token in set(target_norm.split()):
                token_ids.update(by_token.get(token, set()))
            candidate_pool = [id_to_row[item] for item in token_ids if not mapped_speaker or id_to_row[item]['speaker_raw'] == mapped_speaker]
            if not candidate_pool and len(target_norm) <= 12:
                candidate_pool = pool
        evaluated = []
        for candidate in candidate_pool:
            target_sim = similarity(target_norm, candidate['text_normalized'])
            if target_sim < TARGET_REASONABLE:
                continue
            speaker_match = mapped_speaker is not None and candidate['speaker_raw'] == mapped_speaker
            context1, context2, immediate = local_context(candidate, context, context_speakers, aliases, by_conversation)
            context_ok = context1 >= CONTEXT1_HIGH and (len(context) < 2 or context2 >= CONTEXT2_HIGH)
            evidence = min([target_sim, context1] + ([context2] if len(context) >= 2 else []))
            evaluated.append((candidate, target_sim, context1, context2, speaker_match, context_ok, evidence, immediate))
        high = [item for item in evaluated if item[1] >= TARGET_HIGH and item[4] and item[5]]
        exact_high = [item for item in high if item[1] == 1.0]
        if len(exact_high) == 1:
            selected = exact_high[0]
            alignment_type = 'LEVEL_A_EXACT'
            auto_status = 'AUTO_CANDIDATE'
        elif len(high) == 1:
            selected = high[0]
            alignment_type = 'LEVEL_B_HIGH_CONFIDENCE'
            auto_status = 'MANUAL_REVIEW_REQUIRED'
        elif evaluated:
            selected = max(evaluated, key=lambda x: (x[6], x[1], x[2], x[3], x[0]['friends_utterance_id']))
            alignment_type = 'LEVEL_C_AMBIGUOUS'
            auto_status = 'MANUAL_REVIEW_REQUIRED'
        else:
            selected = None
            alignment_type = 'NO_MATCH'
            auto_status = 'REJECTED'
        if selected is None:
            candidate = {key: 'UNKNOWN' for key in ('friends_utterance_id', 'conversation_id', 'text_raw', 'text_normalized', 'speaker_raw', 'season', 'episode', 'scene', 'order_in_scene')}
            target_sim = context1 = context2 = evidence = 0.0
            speaker_match_value = 'UNKNOWN' if mapped_speaker is None else 'FALSE'
            immediate = []
        else:
            candidate, target_sim, context1, context2, speaker_match, _, evidence, immediate = selected
            speaker_match_value = 'TRUE' if speaker_match else 'UNKNOWN' if mapped_speaker is None else 'FALSE'
        prev2 = immediate[-2] if len(immediate) >= 2 else None
        prev1 = immediate[-1] if immediate else None
        notes = []
        if mapped_speaker is None:
            notes.append('unresolved_speaker')
        if len(high) > 1:
            notes.append('multiple_high_confidence_candidates')
        if selected is None:
            notes.append('no_candidate_at_or_above_reasonable_threshold')
        else:
            if target_sim < TARGET_HIGH:
                notes.append('target_similarity_below_high_threshold')
            if speaker_match_value != 'TRUE':
                notes.append('speaker_not_verified')
            if context1 < CONTEXT1_HIGH:
                notes.append('context1_below_high_threshold')
            if len(context) >= 2 and context2 < CONTEXT2_HIGH:
                notes.append('context2_below_high_threshold')
        outputs.append({'mustard_id': mustard_id, 'sarcasm_label': 'sarcastic' if source.get('sarcasm') is True else 'literal' if source.get('sarcasm') is False else 'UNKNOWN', 'mustard_target_raw': target_raw, 'mustard_target_norm': target_norm, 'mustard_target_norm_no_apostrophe': target_no_apostrophe, 'mustard_speaker': mustard_speaker, 'mustard_context_raw': json_cell(context), 'mustard_context_norm': json_cell([normalize_text(item) for item in context]), 'mustard_context_last2': json_cell(context[-2:]), 'mustard_context_speakers_last2': json_cell(context_speakers[-2:]), 'target_media_id': mustard_id, 'target_media_path': 'UNKNOWN', 'friends_candidate_id': candidate['friends_utterance_id'], 'friends_utterance_id': candidate['friends_utterance_id'], 'conversation_id': candidate['conversation_id'], 'friends_target_raw': candidate['text_raw'], 'friends_target_norm': candidate['text_normalized'], 'friends_speaker': candidate['speaker_raw'], 'season': candidate['season'], 'episode': candidate['episode'], 'scene': candidate['scene'], 'friends_episode': f"s{int(candidate['season']):02d}_e{int(candidate['episode']):02d}" if candidate['season'] != 'UNKNOWN' else 'UNKNOWN', 'friends_scene': candidate['conversation_id'], 'order_in_scene': candidate['order_in_scene'], 'friends_prev2_id': prev2['friends_utterance_id'] if prev2 else 'UNKNOWN', 'friends_prev2': prev2['text_raw'] if prev2 else 'UNKNOWN', 'friends_prev1_id': prev1['friends_utterance_id'] if prev1 else 'UNKNOWN', 'friends_prev1': prev1['text_raw'] if prev1 else 'UNKNOWN', 'target_similarity': f'{target_sim:.6f}', 'context1_similarity': f'{context1:.6f}', 'context2_similarity': f'{context2:.6f}', 'speaker_match': speaker_match_value, 'alignment_type': alignment_type, 'alignment_score': f'{evidence:.6f}', 'num_candidate_matches': len(evaluated), 'num_high_confidence_matches': len(high), 'auto_status': auto_status, 'notes': ';'.join(notes)})
    outputs.sort(key=lambda row: row['mustard_id'])
    write_csv(ALIGNMENT / '02_alignment_all.csv', outputs, FIELDS)
    counts = Counter((row['alignment_type'] for row in outputs))
    write_json(MANIFESTS / 'alignment_rules.json', {'target_high': TARGET_HIGH, 'target_reasonable': TARGET_REASONABLE, 'context1_high': CONTEXT1_HIGH, 'context2_high': CONTEXT2_HIGH, 'backward_window': BACKWARD_WINDOW, 'rules': {'LEVEL_A_EXACT': 'unique exact target+speaker+context-qualified candidate', 'LEVEL_B_HIGH_CONFIDENCE': 'unique target>=0.95+speaker+context-qualified candidate', 'LEVEL_C_AMBIGUOUS': 'reasonable candidate exists but acceptance conditions are not uniquely satisfied', 'NO_MATCH': 'no candidate with target similarity >=0.80 after blocking'}, 'observed_counts': dict(counts)})
    summary = f'Alignment candidates: {dict(counts)}\n'
    (LOGS / '04_align_mustard_to_friends.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
