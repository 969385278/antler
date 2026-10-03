"""C19：独立目录运行双极轴与单极对照，不覆盖主坐标系或既有结果。"""
import shutil
import subprocess
import sys
import uuid
import zipfile
from pathlib import Path

import torch
import yaml

from _common import parse_args, setup
from antler.config import ROOT
from antler.traits.data import BIG_FIVE, DIMS
from antler.traits.vectors import build_directions
from e0_diagnostics import now, read_json, sha256, write_json


def commands(configs):
    """两个轴模式使用相同题目、采样数、系数和长度。"""
    return {name: [sys.executable, "-u", "experiments/e0_calibrate.py", "--config", str(path),
                   "--axis", name.split('_')[0], "--dims", name.split('_')[1],
                   "--layers", "16", "--coefs", "0", *(['-0.2', '-0.25'] if name.endswith('_C') else ['0.2', '0.25']),
                   "--pole", "both", "--n-samples", "3", "--max-new-tokens", "320",
                   "--save-samples", "--reuse-baseline", "--with-task"]
            for name, path in configs.items()}


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    if cfg.get('calib_questions', 10) != 10 or not cfg.get('use_judge', True):
        raise ValueError('本轮固定10道calib题且必须启用原评委')
    source = cfg['out_path']
    paths = [source / 'directions.pt'] + [source / 'vectors' / f'{p}.pt' for k in DIMS for p in BIG_FIVE[k]]
    hashes = {str(p): sha256(p) for p in paths}
    direction = torch.load(paths[0], map_location='cpu', weights_only=True)
    poles = {p.stem: torch.load(p, map_location='cpu', weights_only=True) for p in paths[1:]}
    rebuilt = build_directions(poles)
    if (direction.get('dims') != DIMS or rebuilt.shape != direction['U'].shape
            or not torch.allclose(rebuilt, direction['U'], atol=1e-5, rtol=1e-4)):
        raise ValueError('原始极点不能重建当前双极轴，停止不一致来源的比较')
    if rebuilt.shape[0] <= 16 or not torch.isfinite(rebuilt).all():
        raise ValueError('缺少第16层有效方向')
    out = source / 'pole_diagnostics' / (now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    meta = {'task': 'C19', 'status': 'running', 'start_time': now(), 'end_time': None,
            'scope': 'calib_exploratory', 'task_score_validation': 'unvalidated',
            'source_hashes': hashes, 'steps': {}, 'expected_records': 24, 'expected_answers': 720}
    # D-0：高极与反向低极的余弦，及双极轴与首个主成分的余弦。
    _, _, vh = torch.linalg.svd(direction['U'][16].float(), full_matrices=False)
    geometry = {}
    for i, k in enumerate(DIMS):
        hi, lo = [poles[p][16].float() for p in BIG_FIVE[k]]
        geometry[k] = {'high_vs_negative_low': float(torch.nn.functional.cosine_similarity(hi, -lo, dim=0)),
                       'axis_vs_pc1': float(direction['U'][16].float()[i] @ vh[0])}
    write_json(out / 'geometry.json', geometry)
    configs = {}
    for axis in ('bipolar', 'pole'):
        for dim in ('C', 'N'):
            name = f'{axis}_{dim}'
            folder = out / name
            (folder / 'vectors').mkdir(parents=True)
            shutil.copy2(paths[0], folder / 'directions.pt')
            for p in BIG_FIVE[dim]:
                shutil.copy2(source / 'vectors' / f'{p}.pt', folder / 'vectors' / f'{p}.pt')
            if (source / 'judge_cache.jsonl').exists():
                shutil.copy2(source / 'judge_cache.jsonl', folder / 'judge_cache.jsonl')
            config = {k: v for k, v in cfg.items() if k != 'out_path'}
            config['out_dir'] = str(folder)
            configs[name] = folder / 'config.yaml'
            configs[name].write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8', newline='\n')
    plan = commands(configs)
    meta['commands'] = plan
    write_json(out / 'run_meta.json', meta)
    try:
        for name, command in plan.items():
            print(f'[开始] {name}', flush=True)
            meta['steps'][name] = {'status': 'running'}
            write_json(out / 'run_meta.json', meta)
            with (out / name / 'run.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            child = read_json(out / name / 'run_meta.json')
            if child['status'] != 'completed':
                raise RuntimeError(f'{name}: 子实验未完成')
            meta['steps'][name] = {'status': 'completed', 'artifacts': {
                p.name: sha256(p) for p in (out / name).iterdir()
                if p.name in ('run_meta.json', 'calibration_judge.csv', 'calibration_judge_samples.jsonl')}}
            write_json(out / 'run_meta.json', meta)
            print(f'[完成] {name}', flush=True)
        if any(sha256(Path(p)) != h for p, h in hashes.items()):
            raise RuntimeError('运行期间原始向量被更改')
        meta['status'] = 'completed'
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
        with zipfile.ZipFile(out / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in out.rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.jsonl', '.csv', '.log') and p.name != 'judge_cache.jsonl':
                    archive.write(p, p.relative_to(out))
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args())
