"""C11：复用归档激活的随机子空间诊断，不加载模型。"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


DIRECTION_HASH = '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18'
ACTIVATION_HASH = '938a4c95079c6382c7333517699204de3216ddd6e4bccfc7f58437123f76b167'
LAYER = 16
N_RANDOM = 1000
SEED = 0


def add_arguments(ap):
    ap.add_argument('--directions', required=True)
    ap.add_argument('--activations', required=True)


def basis(u):
    """非正交人格方向先正交化；秩不足不能当五维子空间。"""
    u = u.detach().double().cpu()
    if u.ndim != 2 or u.shape[0] != 5 or u.shape[1] < 5 or not torch.isfinite(u).all():
        raise ValueError('方向须为有限的5×d矩阵')
    if int(torch.linalg.matrix_rank(u)) != 5:
        raise ValueError('人格方向秩不足5')
    return torch.linalg.qr(u.T, mode='reduced').Q


def fractions(x, q):
    energy = x.square().sum(1)
    valid = energy > 0
    if not valid.any():
        raise ValueError('所有经历激活差均为零')
    values = (x @ q).square().sum(1)[valid] / energy[valid]
    return [float(v) if ok else None for v, ok in zip(
        (x @ q).square().sum(1) / energy.clamp_min(torch.finfo(x.dtype).tiny), valid)], float(values.mean())


def analyze(x, u, n_random=N_RANDOM, seed=SEED):
    x = x.detach().double().cpu()
    q = basis(u)
    if x.ndim != 2 or x.shape[1] != q.shape[0] or x.shape[0] < 2 or not torch.isfinite(x).all():
        raise ValueError('经历激活差形状无效或包含非有限值')
    if n_random < 1:
        raise ValueError('随机子空间数量须为正')
    variants = {'raw': x, 'centered': x - x.mean(0)}
    result = {}
    for name, value in variants.items():
        per_episode, observed = fractions(value, q)
        result[name] = {'per_episode': per_episode, 'macro_mean': observed,
                        'undefined_count': per_episode.count(None), 'random_macro_means': []}
    generator = torch.Generator(device='cpu').manual_seed(seed)
    for _ in range(n_random):
        random_q = torch.linalg.qr(torch.randn(q.shape, dtype=torch.float64, generator=generator), mode='reduced').Q
        for name, value in variants.items():
            result[name]['random_macro_means'].append(fractions(value, random_q)[1])
    for group in result.values():
        random = torch.tensor(group['random_macro_means'], dtype=torch.float64)
        group.update(random_mean=float(random.mean()),
                     random_quantiles=torch.quantile(random, torch.tensor([.025, .975], dtype=torch.float64)).tolist(),
                     random_tail_fraction=float((1 + (random >= group['macro_mean']).sum()) / (n_random + 1)))
    return {'variants': result, 'n_random': n_random, 'seed': seed,
            'formal_e1_pass': False, 'metric': 'activation_energy_fraction_not_KL'}


def load_inputs(directions, activations):
    if sha256(directions) != DIRECTION_HASH or sha256(activations) != ACTIVATION_HASH:
        raise ValueError('归档输入哈希不符合登记')
    d = torch.load(directions, map_location='cpu', weights_only=True)
    a = torch.load(activations, map_location='cpu', weights_only=True)
    if d['dims'] != ['O', 'C', 'E', 'A', 'N'] or a['layer'] != LAYER or a['episode_ids'] != [f'p{i:02}' for i in range(1, 13)]:
        raise ValueError('维度、层号或经历顺序不符合登记')
    x, u = a['mean_differences'], d['U'][LAYER]
    if x.shape != (12, u.shape[1]):
        raise ValueError('必须保留全部12段经历')
    return x, u, a['episode_ids']


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    out = Path(cfg['out_path']) / 'e1_subspace_pilot' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(args.directions).resolve(), Path(args.activations).resolve(),
               Path(__file__).resolve(), Path('docs/E1_SUBSPACE_PILOT_2026-10-04.md').resolve(),
               Path('docs/RESEARCH_SPEC.md').resolve(), Path(args.config).resolve()]
    meta = {'task': 'C11', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()), 'config': _redacted({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}),
            'argv': _redacted(sys.argv[1:]), 'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'judge': None, 'model_loaded': False,
            'seed': SEED, 'n_random': N_RANDOM, 'layer': LAYER, 'formal_e1_pass': False, 'data_hashes': {}}
    write_json(out / 'run_meta.json', meta)
    try:
        meta['data_hashes'] = {str(p): sha256(p) for p in sources}
        x, u, ids = load_inputs(args.directions, args.activations)
        result = analyze(x, u)
        result['episode_ids'] = ids
        if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('运行期间输入发生变化')
        write_json(out / 'results.json', result)
        meta.update(status='completed', results_sha256=sha256(out / 'results.json'))
    except Exception as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
    print(f'[表示层] 固定随机子空间对照完成：{out}')
    return out


if __name__ == '__main__':
    torch.set_num_threads(1)
    run(parse_args(add_arguments))
