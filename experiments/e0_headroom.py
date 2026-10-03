"""C21：从相反人格起点检验单一方向的可用余量，仅用于 calib。"""
import csv
import hashlib
import math
import subprocess
import sys
import uuid
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model, n_layers
from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT, Judge
from antler.eval.trait_score import trait_score
from antler.traits.data import BIG_FIVE, DIMS, KIT
from antler.traits.headroom_prompts import HEADROOM_PROMPTS
from antler.traits.vectors import steer_vector, steer_vector_poles
from e0_calibrate import _redacted
from e0_diagnostics import now, sha256, write_json


def add_arguments(ap):
    ap.add_argument('--dim', choices=['C', 'N'], required=True)
    ap.add_argument('--prompt-id', choices=list(HEADROOM_PROMPTS), required=True)
    ap.add_argument('--coefs', nargs='+', type=float, required=True)
    ap.add_argument('--layer', type=int, default=16)
    ap.add_argument('--n-samples', type=int, default=3)
    ap.add_argument('--axis', choices=['bipolar', 'pole'], default='bipolar')


def validate(args, cfg):
    if HEADROOM_PROMPTS[args.prompt_id]['dim'] != args.dim:
        raise ValueError('起点提示与检验维度不匹配')
    if (0 not in args.coefs or len(set(args.coefs)) != len(args.coefs)
            or any(not math.isfinite(c) or (c < 0 if args.dim == 'C' else c > 0) for c in args.coefs)):
        raise ValueError('系数必须唯一、有限，包含零基线，C非负、N非正')
    if args.n_samples < 1 or cfg.get('calib_questions', 10) != 10 or not cfg.get('use_judge', True):
        raise ValueError('需要正采样数、10道calib题及原评委')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    validate(args, cfg)
    source = cfg['out_path']
    direction_path = source / 'directions.pt'
    direction = torch.load(direction_path, map_location='cpu', weights_only=True)
    U, s = direction['U'], direction['s']
    if (direction.get('dims') != DIMS or not 0 <= args.layer < len(U)
            or not torch.isfinite(U).all() or not torch.isfinite(s).all() or s[args.layer] <= 0):
        raise ValueError('方向、层号或尺度无效')
    selected_poles = ['high'] if args.dim == 'C' else ['high', 'low']
    files = [direction_path, Path(args.config).resolve(), Path(__file__).resolve(),
             ROOT / 'antler/traits/headroom_prompts.py']
    files += [KIT / 'trait_eval' / f'{BIG_FIVE[args.dim][0 if p == "high" else 1]}.json' for p in selected_poles]
    pole_vectors = {}
    if args.axis == 'pole':
        for p in BIG_FIVE[args.dim]:
            path = source / 'vectors' / f'{p}.pt'
            vector = torch.load(path, map_location='cpu', weights_only=True)
            if vector.shape != (U.shape[0], U.shape[2]):
                raise ValueError('单极向量形状不匹配')
            pole_vectors[p] = vector[args.layer]
            files.append(path)
    out = source / 'headroom' / (now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    system = HEADROOM_PROMPTS[args.prompt_id]['text']
    meta = {'task': 'C21', 'status': 'running', 'start_time': now(), 'end_time': None,
            'scope': 'calib_exploratory', 'task_score_validation': 'unvalidated',
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'config_path': args.config, 'argv': sys.argv[1:], 'seed': 0,
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'prompt_id': args.prompt_id, 'system_prompt': system, 'dim': args.dim,
            'layer': args.layer, 'axis': args.axis, 'coefs': args.coefs, 'n_samples': args.n_samples,
            'max_new_tokens': 320, 'data_hashes': {str(p): sha256(p) for p in files},
            'prompt_hashes': {name: hashlib.sha256(text.encode()).hexdigest()
                              for name, text in [('system', system), ('coherence', COHERENCE_PROMPT), ('task', TASK_PROMPT)]}}
    write_json(out / 'run_meta.json', meta)
    try:
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if n_layers(model) != len(U) or model.config.hidden_size != U.shape[2]:
            raise ValueError('模型与方向不匹配')
        judge = Judge(cache_path=str(out / 'judge_cache.jsonl'))
        meta['judge'] = _redacted({k: getattr(judge, k, None) for k in ('model', 'base_url', 'max_tokens', 'extra_body')})
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'run_meta.json', meta)
        import json
        with (out / 'samples.jsonl').open('w', encoding='utf-8', newline='\n') as sf, (out / 'summary.csv').open('w', encoding='utf-8', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['dim', 'pole', 'coef', 'trait', 'coherence', 'task'])
            for pole in selected_poles:
                for coefficient in args.coefs:
                    z = torch.zeros(5); z[DIMS.index(args.dim)] = coefficient
                    vector = (steer_vector_poles(pole_vectors, float(s[args.layer]), z) if args.axis == 'pole'
                              else steer_vector(U[args.layer], float(s[args.layer]), z))
                    result = trait_score(model, tok, judge, args.dim, pole=pole, system=system,
                                         steer=None if coefficient == 0 else {args.layer: vector},
                                         n_questions=10, n_samples=args.n_samples, max_new_tokens=320, with_task=True)
                    sf.write(json.dumps({'coef': coefficient, 'pole': pole, 'result': result}, ensure_ascii=False) + '\n'); sf.flush()
                    writer.writerow([args.dim, pole, coefficient, result['trait'], result['coherence'], result['task']]); cf.flush()
                    print(f"[测量] {args.prompt_id} {pole} {coefficient:+.2f}: {result['trait']}/{result['coherence']}/{result['task']}", flush=True)
        if any(sha256(Path(p)) != h for p, h in meta['data_hashes'].items()):
            raise RuntimeError('实验期间输入发生变化')
        meta['status'] = 'completed'
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = now()
        meta['artifact_hashes'] = {p.name: sha256(p) for p in (out / 'samples.jsonl', out / 'summary.csv') if p.exists()}
        write_json(out / 'run_meta.json', meta)
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
