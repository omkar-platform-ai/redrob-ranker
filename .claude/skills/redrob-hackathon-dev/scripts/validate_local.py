#!/usr/bin/env python3
"""Quick local validator — run before uploading submission.
Usage: python .claude/skills/redrob-hackathon-dev/scripts/validate_local.py \
           --submission submission.csv --candidates data/candidates.jsonl
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

def load_ids(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    ids = set()
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)['candidate_id'])
    return ids

def validate(submission_path, candidates_path):
    errors, warnings = [], []

    with open(submission_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames or []

    expected_cols = ['candidate_id', 'rank', 'score', 'reasoning']
    if list(cols) != expected_cols:
        errors.append(f'Columns wrong. Got {cols}, need {expected_cols}')

    if len(rows) != 100:
        errors.append(f'Row count: {len(rows)} (need 100)')

    try:
        ranks = [int(r['rank']) for r in rows]
        if sorted(ranks) != list(range(1, 101)):
            errors.append('Ranks must be exactly 1-100, each once')
    except (ValueError, KeyError) as e:
        errors.append(f'Invalid rank values: {e}')

    try:
        scores = [float(r['score']) for r in rows]
        for i in range(1, len(scores)):
            if scores[i] > scores[i-1] + 1e-9:
                errors.append(f'Score not non-increasing at rank {i+1}: {scores[i-1]:.4f}→{scores[i]:.4f}')
                break
        if len(set(scores)) == 1:
            errors.append('All scores identical — model not differentiating')
    except (ValueError, KeyError) as e:
        errors.append(f'Invalid score values: {e}')

    ids = [r.get('candidate_id','') for r in rows]
    if len(ids) != len(set(ids)):
        dupes = [x for x in ids if ids.count(x) > 1]
        errors.append(f'Duplicate candidate_ids: {list(set(dupes))[:5]}')

    print(f'Loading valid IDs from {candidates_path}...')
    valid_ids = load_ids(Path(candidates_path))
    invalid = [cid for cid in ids if cid not in valid_ids]
    if invalid:
        errors.append(f'{len(invalid)} IDs not in dataset: {invalid[:3]}')

    empty = [r['rank'] for r in rows if not r.get('reasoning','').strip()]
    if empty:
        errors.append(f'Empty reasoning at ranks: {empty}')

    reasoning_vals = [r.get('reasoning','') for r in rows]
    most_common = max(reasoning_vals.count(v) for v in set(reasoning_vals)) if reasoning_vals else 0
    if most_common > 5:
        warnings.append(f'Most common reasoning repeated {most_common}x — diversify')

    print()
    if errors:
        print(f'FAILED — {len(errors)} error(s):')
        for e in errors:
            print(f'  • {e}')
    else:
        print('PASSED — submission looks good!')
    if warnings:
        print('\nWarnings:')
        for w in warnings:
            print(f'  • {w}')
    print(f'\nStats: {len(rows)} rows | score range: {min(scores):.4f}–{max(scores):.4f}')
    return len(errors) == 0

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--submission', required=True)
    p.add_argument('--candidates', required=True)
    args = p.parse_args()
    ok = validate(Path(args.submission), Path(args.candidates))
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
