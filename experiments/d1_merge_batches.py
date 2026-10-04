"""C10：核验多个已完成的独立盲标批次并汇总，不重评已有标签。"""
import json
from pathlib import Path
import uuid

from _common import parse_args, setup
from antler.memory.episode_batches import consolidate
from antler.traits.data import load_jsonl
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


def extra(ap):
    ap.add_argument('--manifest', required=True, help='批次列表：id、candidates、prepared、labels_a、labels_b')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'd1_consolidated' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(task='C10', status='running', start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), formal_ready=False)
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    try:
        paths = [Path(args.manifest), Path(args.config), Path(__file__),
                 Path('antler/memory/episode_batches.py'), Path('antler/memory/episode_data.py'),
                 Path('antler/memory/d1_plan.py')]
        batches = []
        for b in read(args.manifest):
            prepared = Path(b['prepared'])
            candidates = Path(b['candidates'])
            public = prepared / 'blind.jsonl'; mapping = prepared / 'private_mapping.json'
            paths += [candidates, public, mapping, Path(b['labels_a']), Path(b['labels_b'])]
            # 标签文件必须属于完成的归档，并核验所有请求/响应来源。
            for key in ('labels_a', 'labels_b'):
                root = Path(b[key]).resolve().parent
                run_meta = root / 'run_meta.json'; manifest = root / 'artifact_hashes.json'
                if read(run_meta)['status'] != 'completed':
                    raise ValueError('独立标注尚未完成')
                paths += [run_meta, manifest]
                for name, digest in read(manifest).items():
                    p = (root / name).resolve()
                    if not p.is_relative_to(root) or sha256(p) != digest:
                        raise ValueError('独立标注产物哈希不符或路径越界')
                    paths.append(p)
            batches.append(dict(id=b['id'], rows=load_jsonl(candidates), public=load_jsonl(public),
                                mapping=read(mapping), a=read(b['labels_a']), b=read(b['labels_b']),
                                packet_sha256=sha256(public)))
        hashes = {str(p.resolve()): sha256(p) for p in paths}
        report = consolidate(batches)
        for key in ('label_accepted', 'label_rejected'):
            rows = report.pop(key)
            (out / (key + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')
        if any(sha256(p) != h for p, h in hashes.items()):
            raise ValueError('汇总期间输入改变')
        write_json(out / 'report.json', report)
        meta.update(status='completed', data_hashes=hashes)
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now(); write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {p.name: sha256(p) for p in out.iterdir()
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
    print(f'[D1合并] {out}；语义审核与作者盲审仍需完成')
    return out


if __name__ == '__main__':
    run(parse_args(extra))
