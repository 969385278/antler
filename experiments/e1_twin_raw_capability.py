"""C12：八个原幅度状态的固定能力检查；默认只离线预检。"""
from pathlib import Path
import hashlib
import math
import sys
import uuid

import pyarrow.parquet as pq
import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.eval.capability import stratified_sample, mmlu_accuracy, capability_summary
from antler.eval.twin_gate_audit import read, digest, verify_manifest
from antler.memory.twin_state import analyze, CELLS
from antler.traits.data import DIMS
from e0_diagnostics import now, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model
from e1_twin_context_gate import seal_sources


STATE_RUN = '03e4e1c7f1d94fe0a31c87e623abf356'
STATE_SHA = '007e42132ed6f2679efb50372523bb36d4712431794685c9f0eef0879810abab'
MMLU_SHA = '74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7'
INDICES_SHA = '8af9e5913d145d889ac7455dee4c4d402612027abb1a93891e231b9317240ed9'
PROTOCOL = {'layer': 16, 'phase': 'all', 'factor': 1., 'n_vectors': 8,
            'n_questions': 1400, 'batch_size': 16, 'seed': 0,
            'max_question_conditions': 12600, 'generated': 0, 'judge_calls': 0}


def extra(ap):
    ap.add_argument('--state-run', required=True)
    ap.add_argument('--canonical-indices', required=True)
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--data', default='data/hf/mmlu/all')
    ap.add_argument('--execute', action='store_true', help='执行固定一次能力检查，失败不自动重跑')


def fixed_vectors(summary, activations, basis):
    """从完整激活独立重建单段状态，不把反馈差值或单位向量用于注入。"""
    if (summary.get('layer') != 16 or summary.get('dims') != DIMS
            or basis.get('dims') != DIMS or not math.isfinite(summary['scale'])
            or summary['scale'] <= 0 or summary['scale'] != float(basis['s'])):
        raise ValueError('状态层、单位或维度改变')
    rebuilt = analyze(activations, basis['U'], summary['scale'])
    result = []
    for a, r in CELLS:
        cell = f'a{a}_r{r}'
        for fmt in ('full_state', 'reconstructed_state'):
            state = torch.tensor(summary['groups']['all60'][cell][fmt], dtype=torch.float64)
            expected = torch.tensor(rebuilt['groups']['all60'][cell][fmt], dtype=torch.float64)
            if state.shape != expected.shape or not torch.isfinite(state).all() or not torch.allclose(
                    state, expected, atol=1e-10, rtol=1e-8):
                raise ValueError('单段状态与封存激活不一致')
            # 先按s单位恢复物理幅度，再统一存为实际传给Steer的float32。
            physical = (state * summary['scale']).float().contiguous()
            result.append({'id': f'{cell}_{fmt}', 'cell': cell, 'format': fmt,
                           'action_sign': a, 'outcome_sign': r, 'vector': physical,
                           'vector_sha256': hashlib.sha256(physical.numpy().tobytes()).hexdigest(),
                           'norm_physical': float(physical.norm())})
    return result


def fixed_sample(rows, canonical, *, total=1400):
    """必须逐项等于旧canonical身份顺序，重新抽样仅用于核验。"""
    sample = stratified_sample(rows, total=total, seed=0)
    identities = [{'source_index': r['source_index'], 'subject': r['subject']} for r in sample]
    if identities != canonical:
        raise ValueError('能力题目身份或顺序与canonical不符')
    return sample


def validate(args):
    folder = Path(args.state_run).resolve()
    verify_manifest(folder, read(folder / 'artifact_hashes.json'))
    meta = read(folder / 'run_meta.json')
    if (folder.name != STATE_RUN or meta.get('status') != 'completed'
            or digest(folder / 'summary.json') != STATE_SHA
            or meta.get('conditions_completed') != 9 or meta.get('prompt_sequences_completed') != 540):
        raise ValueError('激活来源不是固定完成批次')
    vectors = fixed_vectors(read(folder / 'summary.json'),
                            torch.load(folder / 'activations.pt', weights_only=True, map_location='cpu'),
                            torch.load(folder / 'basis.pt', weights_only=True, map_location='cpu'))
    files = sorted((ROOT / args.data).glob('test-*.parquet'))
    if len(files) != 1 or digest(files[0]) != MMLU_SHA or digest(args.canonical_indices) != INDICES_SHA:
        raise ValueError('能力题库或canonical索引哈希改变')
    sample = fixed_sample(pq.read_table(files[0]).to_pylist(), read(args.canonical_indices))
    return vectors, sample, files


def collect(model, tok, sample, vectors, *, layer=16, progress=None):
    """共同零注入与八向量各一次；失败候选完整保留，不择优。"""
    if len(vectors) != 8 or len({v['id'] for v in vectors}) != 8:
        raise ValueError('固定候选必须恰为八个且身份唯一')
    if not all(not p.requires_grad for p in model.parameters()):
        raise ValueError('模型权重未冻结')
    baseline = mmlu_accuracy(model, tok, sample, batch_size=16)
    if progress:
        progress('baseline', baseline, None)
    summaries = {}
    for item in vectors:
        injected = mmlu_accuracy(model, tok, sample, steer={layer: item['vector']}, batch_size=16)
        report = capability_summary(baseline, injected)
        report.update(vector_sha256=item['vector_sha256'],
                      status='capability_pass' if report['within_2pp'] else 'capability_failure',
                      formal_e1_pass=False, behavior_validated=False)
        summaries[item['id']] = report
        if progress:
            progress(item['id'], injected, report)
    return summaries


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    vectors, sample, files = validate(args)
    if not args.execute:
        print('[能力预检] 八向量/1400固定题，未加载模型、未调用评委', flush=True)
        return {'protocol': PROTOCOL, 'model_loaded': False, 'judge_loaded': False}
    hashes = verify_model((ROOT / cfg['model']).resolve(), args.model_fingerprint)
    root = Path(cfg['out_path']) / 'e1_twin_raw_capability'
    root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (STATE_RUN + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = {'task': 'C12', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()), 'config': _redacted(cfg),
            'argv': _redacted(sys.argv[1:]), 'protocol': PROTOCOL, 'model_hashes': hashes,
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'seed': 0, 'judge': None,
            'conditions_completed': 0, 'question_conditions_completed': 0,
            'generated': 0, 'judge_calls': 0, 'formal_e1_pass': False, 'behavior_validated': False}
    write_json(out / 'run_meta.json', meta)
    try:
        sources = [Path(args.config), Path(args.model_fingerprint), Path(args.canonical_indices), *files,
                   ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_TWIN_RAW_BEHAVIOR_2026-10-05.md']
        sources += [Path(args.state_run) / name for name in
                    ('summary.json', 'basis.pt', 'activations.pt', 'stories.json', 'run_meta.json', 'artifact_hashes.json')]
        sources += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        meta['data_hashes'] = {str(p.resolve()): digest(p) for p in sources}
        seal_sources(out, meta['data_hashes'])
        write_json(out / 'sample.json', sample)
        torch.save({v['id']: v['vector'] for v in vectors}, out / 'vectors.pt')
        write_json(out / 'vectors.json', [{k: value for k, value in v.items() if k != 'vector'} for v in vectors])
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if model.config.hidden_size != vectors[0]['vector'].numel() or model.config.num_hidden_layers <= 16:
            raise ValueError('模型形状不符')
        meta['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        def save(condition, predictions, report):
            write_json(out / (condition + '.json'), predictions)
            if report is not None:
                write_json(out / (condition + '_summary.json'), report)
            if any(digest(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('能力检查期间来源改变')
            meta['conditions_completed'] += 1
            meta['question_conditions_completed'] += predictions['n']
            write_json(out / 'run_meta.json', meta)
            print(f'[固定能力] {meta["conditions_completed"]}/9，{condition}，'
                  f'正确{sum(r["correct"] for r in predictions["samples"])}/1400', flush=True)
        summary = collect(model, tok, sample, vectors, progress=save)
        write_json(out / 'summary.json', summary)
        if meta['question_conditions_completed'] != 12600:
            raise ValueError('完成预算不符')
        meta['status'] = 'completed'
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {p.relative_to(out).as_posix(): digest(p)
                   for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
