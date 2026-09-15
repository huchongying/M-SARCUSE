from __future__ import annotations
import hashlib
import json
from collections import Counter
from pathlib import Path
from pipeline_common import ALIGNMENT, FRIENDS_INDEX, LOGS, MANIFESTS, PACKAGE_ROOT, REPORTS, ROOT, ensure_output_dirs, load_mustard, parse_json_cell, pct, read_csv, write_json
DROP_REASONS = {'SCENE_END', 'NO_DIFFERENT_SPEAKER_RESPONSE', 'TRANSCRIPT_ONLY_AFTER_TARGET', 'AMBIGUOUS_TARGET', 'CONTEXT_MISMATCH', 'SPEAKER_MISMATCH', 'NO_MATCH', 'MEDIA_MISSING', 'OTHER'}

def decide_feasibility(retained: int, episodes: int, max_episode_share: float, max_speaker_share: float, ambiguity_rate: float) -> tuple[str, list[str]]:
    reasons = []
    if retained < 80 or episodes < 10 or max_episode_share > 0.2 or (max_speaker_share > 0.75) or (ambiguity_rate > 0.5):
        return ('NOT_RECOMMENDED', ['minimum size, coverage, concentration, or ambiguity requirement not met'])
    if retained >= 200 and episodes >= 30 and (max_episode_share <= 0.1) and (max_speaker_share <= 0.6):
        return ('STRONG', reasons)
    if retained >= 150 and episodes >= 20 and (max_episode_share <= 0.15) and (max_speaker_share <= 0.7):
        return ('MODERATE', reasons)
    return ('WEAK', reasons)

def main() -> int:
    ensure_output_dirs()
    all_rows = read_csv(ALIGNMENT / '02_alignment_all.csv')
    accepted = read_csv(ALIGNMENT / '03_with_observed_response.csv')
    rejected = read_csv(ALIGNMENT / '03_rejected.csv')
    review = read_csv(ALIGNMENT / '04_manual_review.csv')
    friends = {row['friends_utterance_id']: row for row in read_csv(FRIENDS_INDEX)}
    mustard = load_mustard()
    source = {key: row for key, row in mustard.items() if str(row.get('show', '')).casefold() == 'friends'}
    source_ids = set(source)

    def unique(rows: list[dict], key: str) -> bool:
        values = [row[key] for row in rows]
        return len(values) == len(set(values))
    assert len(all_rows) == len(source_ids), 'Source count and alignment count differ'
    assert unique(all_rows, 'mustard_id'), 'mustard_id is not unique in alignment'
    accepted_ids = {row['mustard_id'] for row in accepted}
    rejected_ids = {row['mustard_id'] for row in rejected}
    assert accepted_ids.isdisjoint(rejected_ids), 'Accepted/rejected overlap'
    assert accepted_ids | rejected_ids == source_ids, 'Accepted+rejected do not close to source'
    assert len(review) == len(accepted), 'Manual review export must contain every retained candidate'
    assert {row['mustard_id'] for row in review} == accepted_ids, 'Manual review IDs differ'
    assert all((row['reviewer_1'] == row['reviewer_2'] == '' for row in review)), 'Reviewer fields must be empty'
    assert all((row['drop_reason'] in DROP_REASONS for row in rejected)), 'Invalid drop_reason'
    assert all((not row['drop_reason'] for row in accepted)), 'Retained row has drop_reason'
    assert all((row['sarcasm_label'] in {'sarcastic', 'literal'} for row in all_rows)), 'Missing sarcasm label'
    assert all((row['mustard_target_raw'] and row['mustard_target_norm'] for row in all_rows)), 'Missing raw/normalized target'
    assert all((row['mustard_context_raw'] != '' and row['mustard_context_norm'] != '' for row in all_rows)), 'Missing raw/normalized context'
    assert all((row['friends_utterance_id'] == row['friends_candidate_id'] for row in all_rows)), 'Candidate/utterance ID mismatch'
    for row in all_rows:
        original = source[row['mustard_id']]
        expected_label = 'sarcastic' if original['sarcasm'] is True else 'literal'
        assert row['sarcasm_label'] == expected_label, 'Sarcasm label changed'
        assert row['mustard_target_raw'] == original['utterance'], 'Raw target text changed'
        assert parse_json_cell(row['mustard_context_raw'], []) == original['context'], 'Raw context changed'
        assert row['target_media_id'] == row['mustard_id'], 'Media trace ID changed'
    assert all((0.0 <= float(row[field]) <= 1.0 for row in all_rows for field in ('target_similarity', 'context1_similarity', 'context2_similarity', 'alignment_score'))), 'Similarity outside [0,1]'
    for row in accepted:
        target = friends[row['friends_candidate_id']]
        response = friends[row['observed_response_id']]
        assert row['observed_response_text'] and row['observed_response_norm'], 'Empty response'
        assert row['friends_target_raw'] == target['text_raw'], 'Friends target raw text changed'
        assert row['observed_response_text'] == response['text_raw'], 'Observed response raw text changed'
        assert target['conversation_id'] == response['conversation_id'] == row['conversation_id'], 'Cross-scene response'
        assert int(response['order_in_scene']) > int(target['order_in_scene']), 'Response precedes target'
        assert response['speaker_raw'] != target['speaker_raw'], 'Response speaker equals target speaker'
        assert response['speaker_raw'] != 'TRANSCRIPT_NOTE', 'Response is transcript note'
        assert row['alignment_type'] in {'LEVEL_A_EXACT', 'LEVEL_B_HIGH_CONFIDENCE', 'LEVEL_C_AMBIGUOUS'}, 'Unexpected retained type'
        assert row['manual_review_status'] == 'PENDING', 'Retained candidate is not pending review'
    alignment_counts = Counter((row['alignment_type'] for row in all_rows))
    retained_alignment_counts = Counter((row['alignment_type'] for row in accepted))
    alignment_reason_tags = Counter((tag for row in all_rows for tag in row.get('notes', '').split(';') if tag))
    label_counts = Counter((row['sarcasm_label'] for row in accepted))
    speaker_counts = Counter((row['mustard_speaker'] for row in accepted))
    drop_counts = Counter((row['drop_reason'] for row in rejected))
    episodes = {(row['season'], row['episode']) for row in accepted}
    scenes = {row['conversation_id'] for row in accepted}
    episode_counts = Counter(((row['season'], row['episode']) for row in accepted))
    duplicate_friend_targets = {key: count for key, count in Counter((row['friends_candidate_id'] for row in accepted)).items() if count > 1}
    unique_friend_targets = len({row['friends_candidate_id'] for row in accepted})
    retained = len(accepted)
    ambiguity_rate = alignment_counts['LEVEL_C_AMBIGUOUS'] / len(all_rows) if all_rows else 0.0
    max_episode_share = max(episode_counts.values(), default=0) / retained if retained else 0.0
    max_speaker_share = max(speaker_counts.values(), default=0) / retained if retained else 0.0
    feasibility, feasibility_notes = decide_feasibility(retained, len(episodes), max_episode_share, max_speaker_share, ambiguity_rate)
    eligible_alignment = len(all_rows) - alignment_counts['NO_MATCH']
    source_manifest = json.loads((MANIFESTS / 'source_manifest.json').read_text(encoding='utf-8'))
    generated_files = [PACKAGE_ROOT / 'README.md', PACKAGE_ROOT / 'code' / 'requirements.txt', ALIGNMENT / 'friends_index.csv', ALIGNMENT / '02_alignment_all.csv', ALIGNMENT / '03_with_observed_response.csv', ALIGNMENT / '03_rejected.csv', ALIGNMENT / '04_manual_review.csv', REPORTS / '01_source_stats.json', REPORTS / '01_source_stats.md', REPORTS / '02_friends_index_stats.json', MANIFESTS / 'source_manifest.json', MANIFESTS / 'speaker_aliases.json', MANIFESTS / 'alignment_rules.json', *sorted((ROOT / 'scripts').glob('*.py')), *sorted(LOGS.glob('*.log'))]
    report = {'source_versions': source_manifest['sources'], 'mustard_total': len(load_mustard()), 'mustard_friends_source_targets': len(all_rows), 'source_labels': dict(Counter((row['sarcasm_label'] for row in all_rows))), 'alignment_counts': dict(alignment_counts), 'retained_alignment_counts': dict(retained_alignment_counts), 'alignment_reason_tags': dict(alignment_reason_tags), 'valid_observed_response': retained, 'dropped_after_response_extraction': eligible_alignment - retained, 'final_candidates_for_manual_review': len(review), 'unique_friends_episodes': len(episodes), 'unique_friends_scenes': len(scenes), 'retained_labels': dict(label_counts), 'retained_speakers': dict(speaker_counts), 'drop_reasons': dict(drop_counts), 'rates_percent': {'exact_alignment': pct(alignment_counts['LEVEL_A_EXACT'], len(all_rows)), 'high_confidence_alignment': pct(alignment_counts['LEVEL_B_HIGH_CONFIDENCE'], len(all_rows)), 'observed_response_recovery_among_candidate_alignments': pct(retained, eligible_alignment), 'episode_coverage_of_236_documented_friends_episodes': pct(len(episodes), 236), 'ambiguity': pct(alignment_counts['LEVEL_C_AMBIGUOUS'], len(all_rows)), 'max_episode_share_retained': 100 * max_episode_share, 'max_speaker_share_retained': 100 * max_speaker_share}, 'duplicate_friends_target_ids_retained': duplicate_friend_targets, 'unique_friends_target_ids_retained': unique_friend_targets, 'validation': {'source_accounting_closed': True, 'accepted_rejected_overlap': 0, 'reviewer_fields_blank': True, 'hard_assertions_passed': True}, 'decision_thresholds': {'STRONG': '>=200 retained, >=30 episodes, max episode <=10%, max speaker <=60%', 'MODERATE': '>=150 retained, >=20 episodes, max episode <=15%, max speaker <=70%', 'WEAK': '80-149 retained or broader criteria above not met without exclusion trigger', 'NOT_RECOMMENDED': '<80 retained, <10 episodes, max episode >20%, max speaker >75%, or ambiguity >50%'}, 'feasibility': feasibility, 'feasibility_notes': feasibility_notes, 'scope': 'Construction feasibility only; no model training and no response-candidate construction.', 'generated_files': [str(path.resolve()) for path in generated_files] + [str((REPORTS / '04_construction_report.json').resolve()), str((REPORTS / '04_construction_report.md').resolve()), str((MANIFESTS / 'artifact_manifest.json').resolve())]}
    write_json(REPORTS / '04_construction_report.json', report)
    md = f"# MUStARD--Friends observed-response construction report\n\n## Sources and alignment\n\n- MUStARD total: **{report['mustard_total']}**\n- MUStARD Friends source targets: **{len(all_rows)}**\n- Source labels: {report['source_labels']}\n- LEVEL_A_EXACT: **{alignment_counts['LEVEL_A_EXACT']}**\n- LEVEL_B_HIGH_CONFIDENCE: **{alignment_counts['LEVEL_B_HIGH_CONFIDENCE']}**\n- LEVEL_C_AMBIGUOUS: **{alignment_counts['LEVEL_C_AMBIGUOUS']}**\n- NO_MATCH: **{alignment_counts['NO_MATCH']}**\n- Alignment reason tags: {dict(alignment_reason_tags)}\n\n## Observed responses\n\n- Valid observed responses: **{retained}**\n- Dropped after response extraction: **{eligible_alignment - retained}**\n- Final candidates for manual review: **{len(review)}**\n- Retained A/B/C: **{dict(retained_alignment_counts)}**\n- Unique Friends target IDs retained: **{unique_friend_targets}** (duplicate mapping records: {duplicate_friend_targets})\n- Unique Friends episodes: **{len(episodes)}**\n- Unique scenes: **{len(scenes)}**\n- Retained labels: {dict(label_counts)}\n- Retained speakers: {dict(speaker_counts)}\n- Drop reasons: {dict(drop_counts)}\n\n## Rates\n\n- Exact alignment rate: **{report['rates_percent']['exact_alignment']:.2f}%**\n- High-confidence alignment rate: **{report['rates_percent']['high_confidence_alignment']:.2f}%**\n- Observed-response recovery among A/B/C candidate alignments: **{report['rates_percent']['observed_response_recovery_among_candidate_alignments']:.2f}%**\n- Episode coverage: **{len(episodes)}/236 ({report['rates_percent']['episode_coverage_of_236_documented_friends_episodes']:.2f}%)**\n- Ambiguity rate: **{report['rates_percent']['ambiguity']:.2f}%**\n\n## Decision\n\n**FEASIBILITY = {feasibility}**\n\nThis is a construction-feasibility result. All A/B/C candidates remain pending human review; LEVEL_C is never auto-accepted. No model experiment or four-choice response task was created.\n"
    (REPORTS / '04_construction_report.md').write_text(md, encoding='utf-8')
    artifact_paths = [path for path in generated_files if path.is_file()] + [REPORTS / '04_construction_report.json', REPORTS / '04_construction_report.md']
    hashes = {}
    for path in artifact_paths:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[str(path.resolve())] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    write_json(MANIFESTS / 'artifact_manifest.json', {'files': hashes})
    summary = f'Validation passed; retained={retained}; episodes={len(episodes)}; scenes={len(scenes)}; FEASIBILITY={feasibility}\n'
    (LOGS / '07_validate_alignment.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
