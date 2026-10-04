"""C11：完整dev网格之后的固定探针复测；默认只核验，不启动模型。"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.eval.e1_stability import compare_probe_replicates
from antler.memory.e1_dev_summary import summarize_grid
from antler.memory.e1_probe_repeat import probe_repeat_plan
from antler.memory.kl_validation import evaluate_kl, fit_and_evaluate
from antler.memory.write import projection_coordinates
from antler.traits.data import BIG_FIVE, DIMS, KIT, load_trait
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_kl_dev import load_inputs
from e1_kl_pilot import prepare_corpus, fit_activation_difference, serializable
from e1_projection_diagnostic import LAYER


PLAN_SHA = 'ead7eed31ae79613207439c896355b4416998bc1a48f40d8be62dc01d153e857'
FINGERPRINT_SHA = 'bc1de2777cf71a729570fefafcaea5b40cf5e03d2e8acea1bc39ab25d68eb1ae'


def add_arguments(ap):
    ap.add_argument('--reference-run', required=True, help='已完整结束且保留全部产物的90条件目录')
    ap.add_argument('--directions', required=True)
    ap.add_argument('--experiences', required=True)
    ap.add_argument('--probe-plan', required=True)
    ap.add_argument('--model-fingerprint', required=True, help='原dev运行期间封存的模型文件指纹')
    ap.add_argument('--execute', action='store_true', help='核验后实际复测；默认不加载模型')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_archive(folder):
    """要求终态和完整产物，不从局部结果或声明的计数推断完成。"""
    folder = Path(folder).resolve()
    meta = read(folder / 'run_meta.json')
    if (meta.get('status') != 'completed' or meta.get('preflight') is not False
            or meta.get('episodes_completed') != 10):
        raise ValueError('参考dev尚未完整结束，禁止提前选参复测')
    manifest = read(folder / 'artifact_hashes.json')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*')
              if p.is_file() and p.name != 'artifact_hashes.json'}
    # 兼容清单在Linux生成、在Windows审计的路径分隔符。
    actual = {s.replace('\\', '/') for s in actual}
    if set(manifest) != actual:
        raise ValueError('参考产物清单缺失或有未登记文件')
    for name, digest in manifest.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or sha256(path) != digest:
            raise ValueError('参考产物哈希错误或路径越界')
    return meta


def validate_reference(args):
    folder = Path(args.reference_run).resolve()
    meta = verify_archive(folder)
    if sha256(args.probe_plan) != PLAN_SHA:
        raise ValueError('复测探针不是封存版本')
    plan = read(args.probe_plan)
    if folder.name != plan['reference_run']:
        raise ValueError('参考运行不是登记的90条件实验')
    for name, digest in plan['source_hashes'].items():
        basename = name.replace('\\', '/').rsplit('/', 1)[-1]
        path = ROOT / 'antler/memory/e1_probe_repeat.py' if basename == 'e1_probe_repeat.py' else KIT / 'trait_extract' / basename
        if sha256(path) != digest:
            raise ValueError('探针选择器或提取集来源改变')
    u, scales, episodes = load_inputs(args.directions, args.experiences)
    if sha256(args.experiences) != plan['episode_manifest_sha256']:
        raise ValueError('经历清单改变')
    reference_probes = read(folder / 'probes.json')
    if sha256(folder / 'probes.json') != plan['reference_probes_sha256']:
        raise ValueError('参考探针改变')
    pool = [q for d in DIMS for q in load_trait(BIG_FIVE[d][0], 'extract')['questions']]
    expected = probe_repeat_plan(pool, reference_probes)
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('新探针与固定选择规则不一致')
    for k, v in {'layer': LAYER, 'steps': 60, 'seed': 0, 'lam': .001,
                 'probes_per_step': 4, 'max_new_tokens': 64, 'steer_phase': 'all'}.items():
        if meta['plan'].get(k) != v:
            raise ValueError('原拟合配置不符合复测登记')
    if not meta.get('all_parameters_frozen'):
        raise ValueError('参考运行未记录权重冻结')
    results = read(folder / 'results.json')
    summary = summarize_grid(results, read(args.experiences), hidden=u.shape[2])
    for row in results['episodes']:
        if read(folder / row['id'] / 'results.json') != row:
            raise ValueError('逐段结果与汇总不一致')
        if len(row['delta_A']) != 5 or not all(math.isfinite(v) for v in row['delta_A']):
            raise ValueError('参考A坐标缺维或非有限')
        for name, arm in row['arms'].items():
            size = u.shape[2] if name.startswith('full_') else 5
            if len(arm['coordinates']) != size or not all(math.isfinite(v) for v in arm['coordinates']):
                raise ValueError('参考拟合坐标缺维或非有限')
            if arm['fit_probes'] != reference_probes['fit'] or arm['evaluation_probes'] != reference_probes['evaluation']:
                raise ValueError('不同经历的参考探针不一致')
    bases = torch.load(folder / 'bases.pt', map_location='cpu', weights_only=True)
    if (bases['layer'] != LAYER or not torch.equal(bases['personality'], u[LAYER])
            or float(bases['s']) != float(scales[LAYER])):
        raise ValueError('参考坐标基或单位改变')
    return meta, plan, u, scales, episodes, results, summary


def verify_model(model_dir, fingerprint):
    """只证明与原dev中途指纹一致，不声称上游真实性或原运行前已验证。"""
    if sha256(fingerprint) != FINGERPRINT_SHA:
        raise ValueError('模型指纹不是原封存版本')
    report = read(fingerprint)
    hashes = {}
    for record in report['files']:
        p = (model_dir / record['file']).resolve()
        if (not p.is_relative_to(model_dir.resolve()) or p.stat().st_size != record['bytes']
                or sha256(p) != record['sha256']):
            raise ValueError('模型文件与原dev中途指纹不一致')
        hashes[str(p)] = record['sha256']
    return hashes


def heldout_differences(original, repeat, probes):
    """同留出题的老师prompt、token及文本均须一致，差异原样保存。"""
    first = {r['probe']: r for r in original}
    second = {r['probe']: r for r in repeat}
    if len(first) != len(original) or len(second) != len(repeat):
        raise ValueError('老师回答有重复探针')
    keys = ('teacher_prompt', 'teacher_prompt_ids', 'neutral_prompts', 'text',
            'token_ids', 'n_generated_tokens', 'finish_reason')
    differences = []
    for q in probes:
        a, b = first.get(q, {}), second.get(q, {})
        changed = [k for k in keys if k not in a or k not in b or a[k] != b[k]]
        if changed:
            differences.append({'probe': q, 'different_fields': changed})
    return differences


def bundle(method, episodes, deltas, probes, basis_sha, lr):
    settings = {'projection': 'dual', 'neutral_count': 5, 'activation_average': 'all_fit_probes', 'steer_phase': 'all'}
    if method == 'B':
        settings.update(lr=lr, steps=60, seed=0, lam=.001, probes_per_step=4,
                        initialization='own_probe_A', optimizer='Adam', max_new_tokens=64)
    return {'method': method, 'layer': LAYER, 'basis_sha256': basis_sha, 'units': 's_layer',
            'fit_settings': settings, 'probes': probes,
            'episodes': [{'id': r['id'], 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest(),
                          'group_id': r['group_id'], 'intended': r['intended'], 'delta': deltas[r['id']]}
                         for r in episodes]}


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    reference_meta, plan, u, scales, episodes, reference, selection = validate_reference(args)
    choice = selection['spaces']['personality']
    print(f'[探针复测] 完整90条件已核验，固定学习率{choice["selected_lr"]}，执行={args.execute}', flush=True)
    if not args.execute:
        return None
    model_dir = (ROOT / cfg['model']).resolve()
    if cfg.get('dtype', 'bfloat16') != reference_meta['config'].get('dtype', 'bfloat16'):
        raise ValueError('复测模型精度与参考运行不一致')
    model_hashes = verify_model(model_dir, args.model_fingerprint)
    root = Path(cfg['out_path']) / 'e1_probe_repeat'; root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (PLAN_SHA + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = dict(task='C11', status='running', stage='load', start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), git_dirty=bool(_git('status', '--porcelain')),
                config_path=str(Path(args.config).resolve()), config=_redacted(cfg), argv=_redacted(sys.argv[1:]),
                torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
                gpu=None, judge=None, seed=0,
                reference_run=str(Path(args.reference_run).resolve()), selected_arm=choice['selected_arm'],
                selected_lr=choice['selected_lr'], episodes_completed=0, formal_e1_pass=False,
                behavior_validated=False, data_hashes={}, model_fingerprint_scope='matches_mid_run_reference_only')
    save = lambda: write_json(out / 'run_meta.json', meta)
    save()
    try:
        paths = [Path(args.config), Path(args.directions), Path(args.experiences), Path(args.probe_plan),
                 Path(args.model_fingerprint), Path(args.reference_run) / 'artifact_hashes.json',
                 ROOT / 'docs/E1_PROBE_REPEAT_2026-10-04.md', ROOT / 'docs/RESEARCH_SPEC.md']
        paths += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        paths += [KIT / 'trait_extract' / f'{BIG_FIVE[d][0]}.json' for d in DIMS]
        meta['data_hashes'] = {str(p.resolve()): sha256(p) for p in paths}
        meta['model_hashes'] = model_hashes; save()
        def check():
            if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('复测期间输入或源码改变')
        write_json(out / 'selection.json', selection)
        write_json(out / 'probe_plan.json', plan)
        torch.manual_seed(0)
        model, tok = load_model(str(model_dir), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        model.eval()
        if (model.config.num_hidden_layers, model.config.hidden_size) != (u.shape[0], u.shape[2]):
            raise ValueError('模型形状与方向不一致')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型权重未冻结')
        meta.update(all_parameters_frozen=True, parameter_count=sum(p.numel() for p in model.parameters()),
                    model_shape={'num_hidden_layers': model.config.num_hidden_layers,
                                 'hidden_size': model.config.hidden_size},
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        save()
        original = {r['id']: r for r in reference['episodes']}
        new_a, new_b = {}, {}
        for episode in episodes:
            check()
            folder = out / episode['id']; folder.mkdir()
            meta.update(stage='teacher_corpus', episode=episode['id']); save()
            items, answers = prepare_corpus(model, tok, episode['text'], plan['fit'] + plan['evaluation'], 64)
            write_json(folder / 'teacher_answers.json', answers)
            torch.save([asdict(i) for i in items], folder / 'teacher_corpus.pt')
            prior_answers = read(Path(args.reference_run) / episode['id'] / 'teacher_answers.json')
            differences = heldout_differences(prior_answers, answers, plan['evaluation'])
            write_json(folder / 'heldout_teacher_comparison.json', {'differences': differences})
            if differences:
                raise ValueError('共享留出题老师回答变化；已保存差异，停止纯探针比较')
            difference = fit_activation_difference(model, tok, episode['text'], plan['fit'])
            torch.save(difference, folder / 'fit_activation_difference.pt')
            scale = float(scales[LAYER]); a = projection_coordinates(difference, u[LAYER], scale)
            meta['stage'] = 'fit'; save()
            result = fit_and_evaluate(model, items[:20], items[20:], basis=u[LAYER], scale=scale,
                                      layer=LAYER, steps=60, lr=choice['selected_lr'], lam=.001,
                                      probes_per_step=4, seed=0, init=a, diagnostics=True)
            torch.save(result, folder / 'personality.pt')
            new_a[episode['id']] = a.tolist(); new_b[episode['id']] = result['coordinates'].tolist()
            write_json(folder / 'results.json', {'id': episode['id'], 'delta_A': a.tolist(),
                       'A_eval': evaluate_kl(model, items[20:], layer=LAYER, vector=scale*(a @ u[LAYER])),
                       'B': serializable(result), 'formal_e1_pass': False})
            meta['episodes_completed'] += 1; save()
            print(f'[探针复测] {episode["id"]}完成，{meta["episodes_completed"]}/10', flush=True)
            del items
        for method, current in [('A', new_a), ('B', new_b)]:
            previous = {r['id']: original[r['id']]['delta_A'] if method == 'A' else
                        original[r['id']]['arms'][choice['selected_arm']]['coordinates'] for r in episodes}
            before = bundle(method, episodes, previous, plan['reference_fit'], sha256(args.directions), choice['selected_lr'])
            after = bundle(method, episodes, current, plan['fit'], sha256(args.directions), choice['selected_lr'])
            write_json(out / f'{method}_reference.json', before)
            write_json(out / f'{method}_repeat.json', after)
            write_json(out / f'{method}_stability.json', compare_probe_replicates(before, after))
        check()
        if any(sha256(p) != h for p, h in model_hashes.items()):
            raise ValueError('复测期间模型文件改变')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        meta['end_time'] = now(); save()
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
