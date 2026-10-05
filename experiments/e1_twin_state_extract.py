"""C12：只对已建立的E情境读取激活，默认无模型预检。"""
from pathlib import Path
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.activations import last_prompt_acts
from antler.core.model import chat_prompt, load_model
from antler.eval.twin_gate_audit import read, digest, verify_manifest
from antler.eval.twin_context_gate import expand
from antler.memory.twin_state import analyze, select_probes, CELLS
from antler.memory.write import MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from antler.traits.data import BIG_FIVE, DIMS, KIT, load_trait
from e0_diagnostics import now, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model
from e1_twin_context_gate import read_materials, seal_sources


RUN_ID = '9c008133e6e94c7d91b0ff2d5a0af44f'
SUMMARY_SHA = '8d2b85c769f6d9aa3c51b4590c8da24645d66d706e5e64f1681047b7c0e359a9'
DIRECTION_SHA = '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18'
PROTOCOL = {'scenario': 'twin04_ideas', 'layer': 16, 'n_probes': 60, 'probe_seed': 0,
            'probe_groups': [30, 30], 'batch_size': 8, 'prompt_sequences': 540,
            'max_batch_forwards': 72, 'generated': 0, 'judge_calls': 0, 'steer': None}


def extra(ap):
    ap.add_argument('--layer21-pilot', action='store_true', help='作者批准的独立第21层小样本；旧默认不变')
    ap.add_argument('--reference-run', required=True)
    ap.add_argument('--audit-run', required=True)
    ap.add_argument('--directions', default='outputs/analysis/e1_projection_20261004/directions.pt')
    ap.add_argument('--materials', default='data/diagnostics/twin_outcome_dev_v1.json')
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--execute', action='store_true', help='执行固定一次激活诊断；失败不自动重跑')


def validate(args):
    reference, audit = Path(args.reference_run).resolve(), Path(args.audit_run).resolve()
    for folder in (reference, audit):
        verify_manifest(folder, read(folder / 'artifact_hashes.json'))
        if read(folder / 'run_meta.json')['status'] != 'completed':
            raise ValueError('上游或独立审计未完成')
    integrity = read(audit / 'integrity.json')
    if (reference.name != RUN_ID or integrity['status'] != 'passed'
            or integrity['run_id'] != RUN_ID or integrity['summary_sha256'] != SUMMARY_SHA
            or digest(reference / 'summary.json') != SUMMARY_SHA
            or digest(reference / 'run_meta.json') != integrity['run_meta_sha256']):
        raise ValueError('上游不是已审计固定批次')
    episodes = [e for e in expand(read_materials(args.materials)) if e['scenario_id'] == PROTOCOL['scenario']]
    expected = [e for e in read(reference / 'stories.json') if e['scenario_id'] == PROTOCOL['scenario']]
    report = next(r for r in read(reference / 'summary.json')['reports'] if r['scenario_id'] == PROTOCOL['scenario'])
    if episodes != expected or report['gate'] != 'established':
        raise ValueError('四故事改变或该情境未建立效应')
    if digest(args.directions) != DIRECTION_SHA:
        raise ValueError('人格基不是固定来源')
    directions = torch.load(args.directions, map_location='cpu', weights_only=True)
    layer = 21 if getattr(args, 'layer21_pilot', False) else 16
    basis, scale = directions['U'][layer], float(directions['s'][layer])
    if basis.ndim != 2 or basis.shape[0] != 5 or not torch.allclose(basis.norm(dim=1), torch.ones(5), atol=1e-5):
        raise ValueError('人格方向形状或单位范数不符')
    pool = [{'dim': d, 'pole': BIG_FIVE[d][0], 'index': i, 'question': q}
            for d in DIMS for i, q in enumerate(load_trait(BIG_FIVE[d][0], 'extract')['questions'])]
    probes = select_probes(pool)
    analyze(torch.zeros(9, 60, basis.shape[1]), basis, scale)
    return episodes, probes, basis, scale


def collect(model, tok, episodes, probes, *, layer=16, progress=None):
    """固定九条件读取，保存每条提示token；不生成、不干预。"""
    if ([(e['action_sign'], e['outcome_sign']) for e in episodes] != list(CELLS)
            or len(probes) != 60 or len(NEUTRAL_MEMORIES) != 5):
        raise ValueError('条件顺序或探针预算不符')
    memories = [e['text'] for e in episodes] + list(NEUTRAL_MEMORIES)
    groups, prompts_saved = [], []
    for i, memory in enumerate(memories):
        prompts = [chat_prompt(tok, p['question'], system=MEMORY_TEMPLATE.format(experience=memory)) for p in probes]
        records = [{'condition_index': i, 'probe_index': j, 'prompt': p,
                    'input_ids': tok(p, add_special_tokens=False)['input_ids']} for j, p in enumerate(prompts)]
        value = last_prompt_acts(model, tok, prompts, layers=[layer], batch_size=8)
        if value.shape != (60, 1, model.config.hidden_size) or not torch.isfinite(value).all():
            raise ValueError('逐探针激活缺失或非有限')
        groups.append(value[:, 0]); prompts_saved.extend(records)
        if progress:
            progress(i + 1, groups, prompts_saved)
    return torch.stack(groups), prompts_saved


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    episodes, probes, basis, scale = validate(args)
    layer = 21 if getattr(args, 'layer21_pilot', False) else 16
    protocol = {**PROTOCOL, 'layer': layer}
    print(f'[激活诊断] 四故事/60探针/540提示，执行={args.execute}', flush=True)
    if not args.execute:
        return {'protocol': protocol, 'model_loaded': False, 'judge_loaded': False}
    model_hashes = verify_model((ROOT / cfg['model']).resolve(), args.model_fingerprint)
    root = Path(cfg['out_path']) / ('e1_twin_state_extract_l21' if layer == 21 else 'e1_twin_state_extract'); root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (RUN_ID + '_twin04_ideas.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = {'task': 'C12', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config': _redacted(cfg), 'config_path': str(Path(args.config).resolve()),
            'argv': _redacted(sys.argv[1:]), 'protocol': protocol, 'model_hashes': model_hashes,
            'torch': str(torch.__version__), 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'seed': 0, 'judge': None,
            'conditions_completed': 0, 'prompt_sequences_completed': 0, 'batch_forwards_completed': 0,
            'generated': 0, 'judge_calls': 0, 'formal_e1_pass': False, 'behavior_validated': False}
    write_json(out / 'run_meta.json', meta)
    try:
        sources = [Path(args.config), Path(args.materials), Path(args.directions), Path(args.model_fingerprint),
                   Path(args.reference_run) / 'summary.json', Path(args.reference_run) / 'stories.json',
                   Path(args.audit_run) / 'integrity.json', ROOT / 'docs/RESEARCH_SPEC.md',
                   ROOT / 'docs/E1_TWIN_STATE_EXTRACTION_2026-10-05.md']
        sources += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        if layer == 21:
            sources.append(ROOT / 'docs/E1_L21_SMALL_PILOT_2026-10-05.md')
        sources += [KIT / 'trait_extract' / (BIG_FIVE[d][0] + '.json') for d in DIMS]
        meta['data_hashes'] = {str(p.resolve()): digest(p) for p in sources}
        seal_sources(out, meta['data_hashes'])
        write_json(out / 'stories.json', episodes); write_json(out / 'probes.json', probes)
        torch.save({'U': basis, 's': scale, 'dims': DIMS}, out / 'basis.pt')
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        meta['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        if not meta['all_parameters_frozen']:
            raise ValueError('模型权重未冻结')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        def count_forward(_, positional, keyword):
            ids = keyword['input_ids']
            meta['batch_forwards_completed'] += 1
            meta['prompt_sequences_completed'] += int(ids.shape[0])
            if meta['batch_forwards_completed'] > 72 or meta['prompt_sequences_completed'] > 540:
                raise ValueError('实际前向超过固定预算')
        def save_progress(n, groups, prompts):
            if any(digest(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('固定来源在执行期间改变')
            torch.save(torch.stack(groups), out / 'activations.pt')
            write_json(out / 'prompts.json', prompts)
            meta.update(conditions_completed=n)
            write_json(out / 'run_meta.json', meta)
            print(f'[激活读取] {n}/9条件，{n * 60}/540提示', flush=True)
        counter = model.register_forward_pre_hook(count_forward, with_kwargs=True)
        try:
            activations, _ = collect(model, tok, episodes, probes, layer=layer, progress=save_progress)
        finally:
            counter.remove()
        if (meta['batch_forwards_completed'], meta['prompt_sequences_completed']) != (72, 540):
            raise ValueError('实际前向不符合固定预算')
        write_json(out / 'summary.json', analyze(activations, basis, scale, layer=layer))
        meta['status'] = 'completed'
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now(); write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {p.relative_to(out).as_posix(): digest(p)
                   for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    print(f'[激活诊断完成] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(extra))
