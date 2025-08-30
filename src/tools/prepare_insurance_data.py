import json
import os
import random
from typing import List, Dict, Iterable, Tuple
import argparse
from pathlib import Path

# Resolve repo root = two levels up from this file (src/tools/*)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data' / 'processed' / 'training_data'
IN_FILE_DEFAULT = DATA_DIR / 'insurance.csv_qa.json'
TRAIN_OUT = DATA_DIR / 'insurance_qa_train.json'
VAL_OUT = DATA_DIR / 'insurance_qa_validation.json'

SYSTEM_INSTRUCTION = 'Answer the health insurance question accurately and helpfully.'


def load_items(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def iter_all_sources(single_file: Path | None) -> Iterable[Tuple[Path, Dict]]:
    if single_file:
        for rec in load_items(single_file):
            yield (single_file, rec)
        return
    # aggregate all *_qa.json files
    for p in DATA_DIR.glob('*_qa.json'):
        # skip already prepared splits
        if p.name in {TRAIN_OUT.name, VAL_OUT.name}:
            continue
        try:
            for rec in load_items(p):
                yield (p, rec)
        except Exception:
            continue


def clean_item(item: Dict) -> Dict | None:
    q = (item.get('question') or '').strip()
    a = (item.get('answer') or '').strip()
    # Allow any source; previously filtered to 'tabular' only
    # src = (item.get('data_source') or '').strip().lower()

    # Non-empty Q/A only
    if not q or not a:
        return None

    # Light content heuristics to drop meta/schema-y items
    bad_substrings = ['dataset', 'records with', 'attributes', 'columns', 'sample size']
    lowq = q.lower()
    if any(bs in lowq for bs in bad_substrings):
        return None
    # Normalize phrasing
    q = q.replace('None insurance', 'health insurance')
    a = a.replace('None insurance', 'health insurance')
    return {
        'instruction': SYSTEM_INSTRUCTION,
        'input': q,
        'output': a,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default=None, help='Optional single input file')
    args = parser.parse_args()

    single = Path(args.in_file).resolve() if args.in_file else None

    cleaned: List[Dict] = []
    seen: set[tuple[str, str]] = set()
    for src_path, it in iter_all_sources(single):
        ci = clean_item(it)
        if not ci:
            continue
        key = (ci['input'], ci['output'])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(ci)

    if not cleaned:
        raise RuntimeError(f'No cleaned samples produced. Check input files in {DATA_DIR}.')

    random.seed(42)
    random.shuffle(cleaned)

    n = len(cleaned)
    val_n = max(1, int(0.15 * n))
    val = cleaned[:val_n]
    train = cleaned[val_n:]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_OUT.open('w', encoding='utf-8') as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with VAL_OUT.open('w', encoding='utf-8') as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f'Wrote {len(train)} train and {len(val)} validation samples to {DATA_DIR}')


if __name__ == '__main__':
    main()
