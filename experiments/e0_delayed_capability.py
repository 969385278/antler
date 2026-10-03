"""C5：固定换行的一次decode能力诊断；不改canonical MMLU，不调用评委。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import uuid
import zipfile

import pyarrow.parquet as pq
import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.eval.delayed_capability import (
    POSITION_NOTE, delayed_mmlu, existing_sample, paired_comparison, vector_identity,
)
from antler.traits.data import BIG_FIVE, DIMS
from antler.traits.vectors import steer_vector_poles
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


LAYER, BATCH_SIZE = 21, 16
COEFS = {'C': -.175, 'N': .165}
SOURCE_SHA256 = {
    'sample_indices': '8af9e5913d145d889ac7455dee4c4d402612027abb1a93891e231b9317240ed9',
    'reference_baseline': 'bacc5d519ab1eac28a683eb09d5d06bc4b5021581327866b17c16a77afc4d490',
    'parquet': '74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7',
}
VECTOR_SHA256 = {
    'directions.pt': '5e99e6e40abbdcd6f5c3d3833f0d0c24c4d41ef676fc0b0c923afabc4db7cfaa',
    'vectors/careless.pt': 'cc6c27f273e0e78d3ed069ab51ea360f00ea1e76d8ec0bfcf749d003c4e6b539',
    'vectors/nervous.pt': '4caa81abb73a3ee01b6bde595dab19a7f879057d7190d7325468d955b4f9f507',
}


def add_arguments(ap):
    ap.add_argument('--vectors-dir', required=True, help='固定完整paired向量目录')
    ap.add_argument('--sample-indices', required=True, help='已登记的1400道原题号文件')
    ap.add_argument('--reference-baseline', required=True, help='同一历史运行的baseline.json')
    ap.add_argument('--data', default='data/hf/mmlu/all', help='本地已登记MMLU parquet目录')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fixed_inputs(args, cfg):
    """以预登记内容哈希和历史记录双重核验来源；不下载或重新抽样。"""
    indices_path = Path(args.sample_indices).resolve()
    baseline_path = Path(args.reference_baseline).resolve()
    if indices_path.parent != baseline_path.parent:
        raise ValueError('题号与历史基线必须来自同一运行目录')
    meta_path = baseline_path.parent / 'run_meta.json'
    files = sorted((ROOT / args.data).glob('test-*.parquet'))
    if len(files) != 1:
        raise ValueError('必须使用已登记的单个本地MMLU parquet，不自动下载')
    expected = [(indices_path, SOURCE_SHA256['sample_indices']),
                (baseline_path, SOURCE_SHA256['reference_baseline']), (files[0], SOURCE_SHA256['parquet'])]
    if any(sha256(p) != digest for p, digest in expected):
        raise ValueError('固定题号、基线或题库内容哈希不符')
    old = read_json(meta_path)
    if (old.get('status') != 'completed' or old.get('scoring') != 'zero_shot_chat_ABCD_logits'
            or old.get('option_rotation', 0) != 0 or old.get('n_questions') != 1400
            or old.get('model', old.get('config', {}).get('model')) != cfg['model']):
        raise ValueError('历史能力来源未完成、不是原顺序或模型身份不符')
    recorded = old.get('artifact_hashes', {})
    if (recorded.get(indices_path.name) != SOURCE_SHA256['sample_indices']
            or recorded.get(baseline_path.name) != SOURCE_SHA256['reference_baseline']):
        raise ValueError('历史元数据未验证题号和基线文件')
    data_hashes = {name.replace('\\', '/').rsplit('/', 1)[-1]: digest
                   for name, digest in old.get('data_hashes', {}).items() if name.endswith('.parquet')}
    if data_hashes != {files[0].name: SOURCE_SHA256['parquet']}:
        raise ValueError('历史题库哈希与当前固定来源不一致')
    data = pq.read_table(files[0]).to_pylist()
    baseline = read_json(baseline_path)
    rows = existing_sample(data, read_json(indices_path), baseline,
                           total=14042, n=1400, subjects=57)
    return rows, baseline, [indices_path, baseline_path, meta_path, *files]


def load_vectors(directory):
    """固定第21层的两个单极方向；记录同目录全部十个极点的身份。"""
    source = Path(directory).resolve()
    path = source / 'directions.pt'
    directions = torch.load(path, map_location='cpu', weights_only=True)
    u, scale = directions['U'], directions['s']
    if (directions.get('dims') != DIMS or u.ndim != 3 or u.shape[1] != 5
            or u.shape[0] <= LAYER or scale.shape != (u.shape[0],)
            or not torch.isfinite(u).all() or not torch.isfinite(scale).all() or scale[LAYER] <= 0):
        raise ValueError('方向维度、层号或尺度无效')
    poles, files = {}, [path]
    for names in BIG_FIVE.values():
        for pole in names:
            file = source / 'vectors' / f'{pole}.pt'
            tensor = torch.load(file, map_location='cpu', weights_only=True)
            if tensor.shape != (u.shape[0], u.shape[2]) or not torch.isfinite(tensor).all():
                raise ValueError('完整单极向量集的形状或数值无效')
            poles[pole] = tensor[LAYER]
            files.append(file)
    vectors = {}
    for dim, coefficient in COEFS.items():
        z = torch.zeros(5)
        z[DIMS.index(dim)] = coefficient
        vector = steer_vector_poles(poles, float(scale[LAYER]), z)
        if not torch.isfinite(vector).all() or vector.norm() <= 0:
            raise ValueError('候选干预向量无效或为零')
        vectors[dim] = {LAYER: vector}
    return vectors, tuple(u.shape), files


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = cfg['out_path'] / 'delayed_capability' / (now().replace(':', '').replace('+', '_')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C5', 'status': 'running', 'stage': 'input_validation', 'start_time': now(),
            'end_time': None, 'git_commit': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')), 'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'argv': sys.argv[1:], 'model': cfg['model'], 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'gpu': None, 'judge': None, 'generation_performed': False, 'layer': LAYER,
            'coefs': COEFS, 'conditions': ['zero', 'C', 'N'], 'batch_size': BATCH_SIZE,
            'prefix_text': '\n', 'prefix_tokens': 1, 'expected_questions': 1400,
            'expected_predictions': 4200, 'expected_example_forwards': 8400,
            'formal_gate_eligible': False, 'replaces_canonical_gate': False,
            'position_note': POSITION_NOTE, 'fixed_source_sha256': SOURCE_SHA256,
            'fixed_vector_sha256': VECTOR_SHA256,
            'vectors_dir': str(Path(args.vectors_dir).resolve()), 'data_hashes': {},
            'completed_predictions': 0}
    samples = []

    def save():
        write_json(out / 'run_meta.json', meta)
        temporary = out / 'samples.tmp'
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in samples),
                             encoding='utf-8', newline='\n')
        temporary.replace(out / 'samples.jsonl')

    save()
    try:
        rows, historical, input_files = fixed_inputs(args, cfg)
        if any(sha256(Path(args.vectors_dir) / name) != digest for name, digest in VECTOR_SHA256.items()):
            raise ValueError('候选向量与预登记paired来源不一致')
        vectors, shape, vector_files = load_vectors(args.vectors_dir)
        model_dir = ROOT / cfg['model']
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise ValueError('固定本地模型目录或config.json缺失，不自动下载')
        input_files += vector_files + [Path(args.config).resolve(), Path(__file__).resolve()]
        input_files += list((ROOT / 'antler').rglob('*.py'))
        input_files += [ROOT / 'experiments' / name for name in
                        ('_common.py', 'e0_diagnostics.py', 'e0_extract_coherence.py', 'e0_refilter.py')]
        input_files += sorted(model_dir.glob('*.json'))
        meta['data_hashes'] = {str(p): sha256(p) for p in input_files}
        meta['vector_identities'] = {dim: vector_identity(v) for dim, v in vectors.items()}
        write_json(out / 'sample_indices.json', [{'source_index': r['source_index'], 'subject': r['subject'],
                                                  'answer': r['answer']} for r in rows])
        meta['stage'] = 'model_load'
        save()
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if (model.config.num_hidden_layers, model.config.hidden_size) != (shape[0], shape[2]):
            raise ValueError('模型与向量形状不一致')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'generation_settings.json', model.generation_config.to_dict())
        results = {}
        for condition in ('zero', 'C', 'N'):
            meta.update(stage='measurement', current_condition=condition)
            save()

            def on_batch(batch, condition=condition):
                samples.extend({'condition': condition, **row} for row in batch)
                meta['completed_predictions'] = len(samples)
                save()

            results[condition] = delayed_mmlu(model, tok, rows, steer=vectors.get(condition),
                batch_size=BATCH_SIZE, reference_first=None if condition == 'zero' else results['zero']['samples'],
                on_batch=on_batch)
            write_json(out / f'{condition}.json', results[condition])
            print(f'[延迟选项] {condition} 完成1400题', flush=True)
        if len(samples) != 4200 or any(result['n'] != 1400 for result in results.values()):
            raise ValueError('最终样本数不符合固定矩阵')
        if any(sha256(p) != digest for p, digest in meta['data_hashes'].items()):
            raise ValueError('实验期间输入内容发生变化')
        zero = results['zero']['samples']
        report = {'formal_gate_eligible': False, 'replaces_canonical_gate': False,
                  'measurement': '强制单换行后的一次decode条件选项预测，不覆盖自由长生成或推理',
                  'position_note': POSITION_NOTE,
                  'historical_to_current_prefill': paired_comparison(historical['samples'], zero,
                                                                   left_step=None, right_step='prefill'),
                  'zero_prefix_effect': paired_comparison(zero, zero, left_step='prefill', right_step='delayed'),
                  'decode_effects': {condition: paired_comparison(zero, results[condition]['samples'])
                                     for condition in COEFS},
                  'n_questions': 1400, 'n_conditions': 3, 'n_delayed_predictions': 4200}
        write_json(out / 'assessment.json', report)
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__,
                    error='诊断已停止；保留阶段及已完成批次，异常原文不写入产物以避免认证信息泄漏。')
        raise
    finally:
        meta['end_time'] = now()
        save()
        names = ['run_meta.json', 'samples.jsonl', 'sample_indices.json', 'generation_settings.json',
                 'zero.json', 'C.json', 'N.json', 'assessment.json']
        manifest = {name: sha256(out / name) for name in names if (out / name).exists()}
        write_json(out / 'artifact_hashes.json', manifest)
        with zipfile.ZipFile(out / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in [*manifest, 'artifact_hashes.json']:
                archive.write(out / name, name)
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
