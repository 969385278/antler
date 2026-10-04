"""C12：离线审计已完成的结果闸门，不加载模型、不调用评委。"""
from pathlib import Path
import sys
import uuid

import torch
import transformers
from transformers import AutoTokenizer

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.twin_gate_audit import audit, digest
from e0_diagnostics import now, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


def extra(ap):
    ap.add_argument('--run', required=True, help='已经完成且包含全部原文的固定闸门目录')
    ap.add_argument('--model-dir', required=True, help='只重查文件指纹并加载本地分词器，不加载模型')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'e1_twin_gate_audit' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__).resolve(), ROOT / 'antler/eval/twin_gate_audit.py', Path(args.config).resolve()]
    meta = {'task': 'C12', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()), 'config': _redacted(cfg),
            'argv': _redacted(sys.argv[1:]), 'judge': None, 'gpu': None, 'model_loaded': False,
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'seed': 0, 'input_run': str(Path(args.run).resolve()),
            'model_forward_calls': 0, 'judge_calls': 0, 'formal_e1_pass': False,
            'data_hashes': {str(p): digest(p) for p in sources}}
    write_json(out / 'run_meta.json', meta)
    try:
        for name in ('artifact_hashes.json', 'run_meta.json'):
            path = Path(args.run).resolve() / name
            meta['data_hashes'][str(path)] = digest(path)
        write_json(out / 'run_meta.json', meta)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        result = audit(args.run, tokenizer, args.model_dir)
        if any(digest(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('审计代码或配置发生变化')
        write_json(out / 'integrity.json', result)
        meta['status'] = 'completed'
        print(f'[独立审计] 2520回答/7560评分、来源和模型指纹核验通过：{out}', flush=True)
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {p.name: digest(p) for p in out.iterdir()
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
