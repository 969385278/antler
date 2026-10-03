"""C5：对锁定的注入条件运行MMLU，独立保存抽样索引、基线和结果。"""
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pyarrow.parquet as pq
import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.eval.capability import stratified_sample, mmlu_accuracy, capability_summary
from antler.traits.data import DIMS, BIG_FIVE
from antler.traits.vectors import steer_vector, steer_vector_poles
from e0_diagnostics import now, sha256, write_json
from e0_calibrate import _redacted


def add_arguments(ap):
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--z', nargs=5, type=float, required=True, help='锁定的O/C/E/A/N坐标')
    ap.add_argument('--data', default='data/hf/mmlu/all', help='含test-*.parquet的目录')
    ap.add_argument('--n-questions', type=int, default=1400)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--axis', choices=['bipolar', 'pole'], default='bipolar')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    files = sorted((ROOT / args.data).glob('test-*.parquet'))
    if not files:
        raise ValueError('未找到MMLU test parquet，不自动下载')
    rows = []
    for file in files:
        rows.extend(pq.read_table(file).to_pylist())
    sample = stratified_sample(rows, args.n_questions, seed=0)
    path = cfg['out_path'] / 'directions.pt'
    direction = torch.load(path, map_location='cpu', weights_only=True)
    z = torch.tensor(args.z)
    if (direction.get('dims') != DIMS or not 0 <= args.layer < len(direction['U'])
            or not torch.isfinite(z).all()):
        raise ValueError('层、维度或坐标无效')
    vector_files = []
    if args.axis == 'pole':
        poles = {}
        for k in DIMS:
            for name in BIG_FIVE[k]:
                p = cfg['out_path'] / 'vectors' / f'{name}.pt'
                value = torch.load(p, map_location='cpu', weights_only=True)
                if value.shape != (direction['U'].shape[0], direction['U'].shape[2]):
                    raise ValueError('单极向量形状不匹配')
                poles[name] = value[args.layer]
                vector_files.append(p)
        vector = steer_vector_poles(poles, float(direction['s'][args.layer]), z)
    else:
        vector = steer_vector(direction['U'][args.layer], float(direction['s'][args.layer]), z)
    if not torch.isfinite(vector).all():
        raise ValueError('注入向量非有限')
    out = cfg['out_path'] / 'capability' / (now().replace(':', '').replace('+','_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    write_json(out / 'sample_indices.json', [{'source_index': r['source_index'], 'subject': r['subject']} for r in sample])
    meta = {'task': 'C5', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'config_path': args.config, 'config': _redacted({k:v for k,v in cfg.items() if k != 'out_path'}),
            'argv': sys.argv[1:], 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'judge': None,
            'layer': args.layer, 'z': args.z, 'axis': args.axis, 'seed': 0, 'n_questions': len(sample), 'model': cfg['model'],
            'batch_size': args.batch_size, 'data_hashes': {str(p): sha256(p) for p in [*files, path, *vector_files]},
            'scoring': 'zero_shot_chat_ABCD_logits'}
    write_json(out / 'run_meta.json', meta)
    try:
        model, tok = load_model(cfg['model'],cfg.get('dtype','bfloat16'),cfg.get('device','auto'))
        if (model.config.num_hidden_layers, model.config.hidden_size) != (direction['U'].shape[0], direction['U'].shape[2]):
            raise ValueError('模型与方向不匹配')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        baseline = mmlu_accuracy(model,tok,sample,batch_size=args.batch_size)
        write_json(out / 'baseline.json', baseline)
        injected = mmlu_accuracy(model,tok,sample,steer={args.layer:vector},batch_size=args.batch_size)
        write_json(out / 'injected.json', injected)
        summary = capability_summary(baseline, injected)
        loss = summary['loss_percentage_points']
        write_json(out / 'summary.json', summary)
        if any(sha256(Path(p)) != h for p,h in meta['data_hashes'].items()):
            raise RuntimeError('实验期间输入发生变化')
        meta['status'] = 'completed'
        print(f'[MMLU] 基线{baseline["accuracy"]:.4f}，注入{injected["accuracy"]:.4f}，下降{loss:.2f}个百分点',flush=True)
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = now()
        meta['artifact_hashes'] = {p.name:sha256(p) for p in out.glob('*.json') if p.name != 'run_meta.json'}
        write_json(out / 'run_meta.json',meta)
        print(f'[结果目录] {out}',flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
