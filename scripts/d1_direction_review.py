"""C10：D1 60 段方向复核（作者指定由助手执行，非人工）。

prepare：从暂留 233 段中固定抽样并导出盲包与私有映射。
score：读取复核标签，与预定目标、两模型标签比较并写结果。
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from antler.memory.d1_review import SALT, SPLIT_QUOTA, make_packet, score_review, select_review_sample, validate_review_labels

ROOT = Path(__file__).resolve().parents[1]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def load_rows(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))['selected']


def prepare(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.selection)
    chosen = select_review_sample(rows)
    public, mapping = make_packet(rows, chosen)
    (out / 'blind.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in public),
                                     encoding='utf-8', newline='\n')
    write_json(out / 'private_mapping.json', mapping)
    write_json(out / 'run_meta.json', {
        'task': 'C10', 'stage': 'prepare', 'time': now(), 'git_commit': git_head(), 'salt': SALT,
        'split_quota': SPLIT_QUOTA, 'n': len(public), 'reviewer_kind': 'assistant', 'human_review_complete': False,
        'input_sha256': {str(args.selection): sha256(args.selection)},
        'artifact_sha256': {n: sha256(out / n) for n in ('blind.jsonl', 'private_mapping.json')}})
    print(f'已导出 {len(public)} 段盲包：{out / "blind.jsonl"}')


def score(args):
    out = Path(args.out)
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    for name, digest in meta['artifact_sha256'].items():
        if sha256(out / name) != digest:
            raise ValueError(f'{name} 哈希与准备阶段不符')
    if sha256(args.selection) != meta['input_sha256'][str(args.selection)]:
        raise ValueError('选择台账已改变')
    public = [json.loads(l) for l in (out / 'blind.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    bundle = json.loads(Path(args.labels).read_text(encoding='utf-8'))
    if bundle.get('input_sha256') != meta['artifact_sha256']['blind.jsonl']:
        raise ValueError('复核标签未绑定当前盲包')
    review = validate_review_labels(bundle['records'], public)
    mapping = json.loads((out / 'private_mapping.json').read_text(encoding='utf-8'))
    result = score_review(load_rows(args.selection), mapping, review)
    result.update({'time': now(), 'git_commit': git_head(), 'reviewer': bundle.get('reviewer'),
                   'labels_sha256': sha256(args.labels)})
    write_json(out / 'review_results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare', 'score'])
    ap.add_argument('--selection', default='outputs/analysis/d1_replacement_selection_20261004.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--labels')
    args = ap.parse_args()
    prepare(args) if args.stage == 'prepare' else score(args)


if __name__ == '__main__':
    main()
