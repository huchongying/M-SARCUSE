from __future__ import annotations
from pipeline_common import ALIGNMENT, LOGS, ensure_output_dirs, parse_json_cell, read_csv, write_csv
FIELDS = ['mustard_id', 'sarcasm_label', 'mustard_context_prev2', 'mustard_context_prev1', 'mustard_target', 'mustard_speaker', 'friends_prev2', 'friends_prev1', 'friends_target', 'friends_speaker', 'friends_next_response', 'friends_next_speaker', 'season', 'episode', 'scene', 'target_similarity', 'context1_similarity', 'context2_similarity', 'alignment_type', 'auto_status', 'alignment_notes', 'reviewer_1', 'reviewer_2', 'review_notes']

def main() -> int:
    ensure_output_dirs()
    source = read_csv(ALIGNMENT / '03_with_observed_response.csv')
    rows = []
    for row in source:
        context = parse_json_cell(row['mustard_context_last2'], [])
        rows.append({'mustard_id': row['mustard_id'], 'sarcasm_label': row['sarcasm_label'], 'mustard_context_prev2': context[-2] if len(context) >= 2 else 'UNKNOWN', 'mustard_context_prev1': context[-1] if context else 'UNKNOWN', 'mustard_target': row['mustard_target_raw'], 'mustard_speaker': row['mustard_speaker'], 'friends_prev2': row['friends_prev2'], 'friends_prev1': row['friends_prev1'], 'friends_target': row['friends_target_raw'], 'friends_speaker': row['friends_speaker'], 'friends_next_response': row['observed_response_text'], 'friends_next_speaker': row['observed_response_speaker'], 'season': row['season'], 'episode': row['episode'], 'scene': row['scene'], 'target_similarity': row['target_similarity'], 'context1_similarity': row['context1_similarity'], 'context2_similarity': row['context2_similarity'], 'alignment_type': row['alignment_type'], 'auto_status': row['auto_status'], 'alignment_notes': row['notes'], 'reviewer_1': '', 'reviewer_2': '', 'review_notes': ''})
    write_csv(ALIGNMENT / '04_manual_review.csv', rows, FIELDS)
    summary = f'Manual-review rows={len(rows)}; reviewer fields left blank\n'
    (LOGS / '06_export_manual_review.log').write_text(summary, encoding='utf-8')
    print(summary, end='')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
