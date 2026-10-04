"""C10：准备D1盲标文件或合并真实独立标签；不调用模型自动伪造标签。"""
from pathlib import Path
import json
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from _common import parse_args, setup
from antler.memory.episode_data import BANNED, blind_packet, merge_and_split, validate_split_plan
from antler.traits.data import load_jsonl
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


def extra(ap):
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--prepared', help='已准备的盲标目录；合并时必须给出')
    ap.add_argument('--labels-a')
    ap.add_argument('--labels-b')
    ap.add_argument('--split-plan', help='生成前冻结的D1情境家族切分计划')


def jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8', newline='\n')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'd1_builder' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C10', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'data_hashes': {}, 'model_loaded': False,
            'judge_called': False, 'formal_ready': False, 'config_path': str(Path(args.config).resolve())}
    try:
        inputs = [Path(args.candidates), Path(args.config), Path(__file__),
                  Path('antler/memory/episode_data.py'), Path('docs/D1_DATA_PIPELINE_2026-10-04.md')]
        rows = load_jsonl(args.candidates)
        plan_path = getattr(args, 'split_plan', None)
        plan = json.loads(Path(plan_path).read_text(encoding='utf-8')) if plan_path else None
        if plan is not None:
            validate_split_plan(rows, plan)
            inputs.append(Path(plan_path))
        public, mapping = blind_packet(rows, allow_grouped=plan is not None)
        jsonl(out / 'blind.jsonl', public)
        write_json(out / 'private_mapping.json', mapping)
        write_json(out / 'banned_words.json', sorted(BANNED))
        if any((args.prepared, args.labels_a, args.labels_b)):
            if not all((args.prepared, args.labels_a, args.labels_b)):
                raise ValueError('合并必须同时提供prepared和两份独立标签')
            prepared = Path(args.prepared)
            if sha256(prepared / 'blind.jsonl') != sha256(out / 'blind.jsonl'):
                raise ValueError('候选与已发送盲标文本不符')
            inputs += [prepared / 'blind.jsonl', prepared / 'private_mapping.json', Path(args.labels_a), Path(args.labels_b)]
            read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
            result = merge_and_split(rows, load_jsonl(prepared / 'blind.jsonl'), read(prepared / 'private_mapping.json'),
                                     read(args.labels_a), read(args.labels_b), sha256(prepared / 'blind.jsonl'), split_plan=plan)
            jsonl(out / 'accepted.jsonl', result.pop('accepted'))
            jsonl(out / 'rejected.jsonl', result.pop('rejected'))
            write_json(out / 'report.json', result)
            meta['quota_complete'] = result['quota_complete']
        meta['data_hashes'] = {str(p.resolve()): sha256(p) for p in inputs}
        meta['status'] = 'completed'
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})
    print(f'[D1] 处理完成：{out}；正式就绪仍需独立复核')
    return out


if __name__ == '__main__':
    run(parse_args(extra))
