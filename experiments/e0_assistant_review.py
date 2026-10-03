"""C9：助手评分的公开封存与揭盲比较，不加载模型或调用评委。"""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval import assistant_review
from antler.eval.blind import write_json
from antler.eval.human import sha256
from e0_extract_coherence import _redacted


def add_arguments(ap):
    ap.add_argument('--action', choices=['seal', 'compare'], required=True)
    ap.add_argument('--bundle', required=True, help='包含公开blind.csv的D5目录')
    ap.add_argument('--ratings', help='已获作者量尺批准的助手评分JSON')
    ap.add_argument('--sealed-dir', help='seal步骤生成的sealed目录')


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    stamp = now()
    out = Path(cfg['out_path']) / 'assistant_d5' / (stamp.replace(':', '').replace('+', '_')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C9', 'action': args.action, 'rating_source': 'assistant',
            'status': 'running', 'start_time': stamp, 'end_time': None,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'argv': sys.argv[1:], 'seed': cfg.get('seed', 0), 'python': sys.version,
            'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'gpu': None, 'judge': None, 'model_loaded': False, 'judge_called': False,
            'input_hashes': {}, 'data_hashes': {}, 'formal_e0_pass': False}
    write_json(out / 'run_meta.json', meta)
    try:
        # 比较入口也不提前读取私钥；完整性检查由compare按顺序执行。
        inputs = [Path(args.config), Path(args.bundle) / 'blind.csv']
        if args.action == 'seal':
            if not args.ratings or args.sealed_dir:
                raise ValueError('封存需要--ratings且不能同时提供--sealed-dir')
            inputs.append(Path(args.ratings))
            inputs.append(Path(args.bundle) / 'assistant_rubric.md')
        elif args.action == 'compare':
            if not args.sealed_dir or args.ratings:
                raise ValueError('比较需要--sealed-dir且不能同时提供--ratings')
            inputs.extend(Path(args.sealed_dir) / name for name in assistant_review.SEALED_FILES)
        else:
            raise ValueError('未知操作')
        for path in inputs:
            meta['input_hashes'][str(path.resolve())] = sha256(path)
        meta['data_hashes'] = dict(meta['input_hashes'])
        if args.action == 'seal':
            result = assistant_review.seal(args.bundle, args.ratings, out / 'sealed')
        else:
            result = assistant_review.compare(args.bundle, args.sealed_dir)
        assistant_review._unchanged(meta['input_hashes'])
        meta['input_hashes'].update(result['input_hashes'])
        meta['data_hashes'] = dict(meta['input_hashes'])
        write_json(out / 'result.json', result)
        meta['status'] = 'completed'
    except BaseException as exc:
        meta['status'] = 'failed'
        meta['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        meta['data_hashes'] = dict(meta['input_hashes'])
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
    print(f'[助手复核结果目录] {out}')
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
