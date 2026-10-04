"""C11：固定四包坐标与十段 dev 行为的 CPU 离线审计。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.e1_coordinate_audit import BUNDLES, analyze
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


LAYER = 16
DIRECTIONS_SHA256 = '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18'
INPUT_SHA256 = {
    'A_reference': '6a6c1cb5253c3e11b3840e800c410f8afff6b4b5b821b5df428efadeada660dc',
    'A_repeat': 'a10d28faadb54a14b6c9f7488f852f703c85b01576ecd711326d7b9e60ca7fa5',
    'B_reference': '56adfdf6414bca55fef21bc38d3b2c7514467846ab316d2c3f914eab07ab1dbb',
    'B_repeat': '3e17f03fb8d8d061b6730c2d3b2efd0e5151598744c5967530853f25c525aa94',
    'behavior': 'd0b33d41666169afe154b5981e135c2517619ca7677ffd5243d9302b33bb7b3d',
}
EXPECTED_IDS = (
    'd1_context08_Ominus', 'd1_context08_Oplus', 'd1_context06_Cminus', 'd1_context10_Cplus',
    'd1_context08_Eminus', 'd1_context08_Eplus', 'd1_context10_Aminus', 'd1_context08_Aplus',
    'd1_context08_Nminus', 'd1_context07_Nplus',
)


def add_arguments(ap):
    ap.add_argument('--directions', required=True, help='固定原始 directions.pt')
    ap.add_argument('--coordinate-dir', required=True, help='含 A/B reference/repeat 四个 JSON 的归档目录')
    ap.add_argument('--behavior', required=True, help='固定 context_coordinate_comparison.json')


def _read(path):
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('JSON 中出现重复键')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=reject_duplicate_keys)


def input_paths(args):
    folder = Path(args.coordinate_dir).resolve()
    return {**{name: folder / f'{name}.json' for name in BUNDLES},
            'behavior': Path(args.behavior).resolve(), 'directions': Path(args.directions).resolve()}


def load_inputs(paths, hashes):
    """归档哈希必须匹配登记值，不接受另选经历或新一轮结果。"""
    expected = {**INPUT_SHA256, 'directions': DIRECTIONS_SHA256}
    if set(paths) != set(expected):
        raise ValueError('审计输入集合不完整')
    for name, path in paths.items():
        if hashes.get(str(path)) != expected[name] or sha256(path) != expected[name]:
            raise ValueError(f'输入 {name} 哈希不符合固定归档')
    bundles = {name: _read(paths[name]) for name in BUNDLES}
    behavior = _read(paths['behavior'])
    directions = torch.load(paths['directions'], map_location='cpu', weights_only=True)
    return bundles, behavior, directions


def run(args):
    config_path = Path(args.config).resolve()
    config_sha = sha256(config_path)
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'e1_coordinate_behavior_audit' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = {
        'task': 'C11', 'status': 'running', 'stage': 'validation',
        'start_time': now(), 'end_time': None,
        'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
        'config_path': str(config_path), 'config': _redacted(cfg), 'argv': _redacted(sys.argv[1:]),
        'torch': str(torch.__version__), 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': None, 'judge': None, 'seed': None,
        'device': 'cpu', 'dtype': 'float64', 'model_loaded': False,
        'model_forward_calls': 0, 'generation_calls': 0, 'judge_calls': 0,
        'formal_e1_pass': False, 'behavior_validated': False,
        'data_hashes': {str(config_path): config_sha}, 'n_episodes': 10, 'n_bundles': 4,
        'input_scope': '只读已归档的固定四包坐标和上下文行为，不读取 trait_eval 题目或模型文件',
    }
    save = lambda: write_json(out / 'run_meta.json', meta)
    save()
    try:
        paths = input_paths(args)
        sources = [Path(__file__).resolve(), ROOT / 'experiments/_common.py',
                   ROOT / 'experiments/e0_diagnostics.py', ROOT / 'experiments/e0_extract_coherence.py',
                   ROOT / 'experiments/e0_refilter.py', ROOT / 'antler/config.py',
                   ROOT / 'antler/eval/e1_coordinate_audit.py', ROOT / 'antler/traits/data.py',
                   ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_STORAGE_READOUT_REVISION_2026-10-05.md']
        meta['data_hashes'].update({str(path): sha256(path) for path in [*paths.values(), *sources]})
        meta['inputs'] = {name: {'path': str(path), 'sha256': meta['data_hashes'][str(path)]}
                          for name, path in paths.items()}
        save()

        def unchanged():
            if any(sha256(path) != digest for path, digest in meta['data_hashes'].items()):
                raise ValueError('离线分析期间输入、配置、代码或登记发生变化')

        unchanged()
        bundles, behavior, directions = load_inputs(paths, meta['data_hashes'])
        unchanged()
        meta['stage'] = 'analysis'; save()
        result = analyze(bundles, behavior, directions, layer=LAYER,
                         basis_sha256=DIRECTIONS_SHA256, expected_ids=EXPECTED_IDS)
        unchanged()
        write_json(out / 'analysis.json', result)
        unchanged()
        meta.update(status='completed', stage='completed', n_groups=result['n_groups'])
        for name, bundle in result['bundles'].items():
            counts = bundle['sign_counts']
            print(f'[离线描述] {name}：对偶 {counts["dual_raw"]["correct"]}/10，'
                  f'中心化对偶 {counts["dual_centered"]["correct"]}/10，'
                  f'轴投影 {counts["axis_raw"]["correct"]}/10，'
                  f'中心化轴投影 {counts["axis_centered"]["correct"]}/10', flush=True)
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__,
                    error='固定离线审计失败；不丢弃失败经历，不自动改输入重试。')
        raise
    finally:
        meta['end_time'] = now(); save()
        write_json(out / 'artifact_hashes.json', {path.name: sha256(path) for path in out.iterdir()
                   if path.is_file() and path.name != 'artifact_hashes.json'})
        print(f'[离线审计目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
