"""C11：固定10段dev的三空间学习率诊断，不运行正式test或行为测量。"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.memory.write import probe_questions, projection_coordinates
from antler.memory.kl_validation import evaluate_kl, fit_and_evaluate
from antler.memory.e1_dev_design import learning_rates
from antler.memory.episode_data import validate_candidates, validate_split_plan
from antler.memory.d1_plan import make_plan
from antler.traits.data import KIT, BIG_FIVE, DIMS
from e1_kl_pilot import prepare_corpus, fit_activation_difference, random_basis, serializable, settings as pilot_settings
from e1_projection_diagnostic import LAYER
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


def add_arguments(ap):
    ap.add_argument('--directions', required=True)
    ap.add_argument('--experiences', required=True, help='请求前封存的10段dev清单JSON')
    ap.add_argument('--preflight', action='store_true', help='只检查首段真实梯度与归档，不用于选参')


def settings(preflight):
    plan = pilot_settings(preflight)
    plan['n_episodes'] = 1 if preflight else 10
    plan.pop('lr')
    plan['diagnostics'] = True
    return plan


def load_inputs(directions, experiences):
    if sha256(experiences) != 'c73dfaf15f4627b253f77f2ffc8bf0777ff8171a40077bf55902890e0cbc405c':
        raise ValueError('dev经历清单不是请求前封存版本')
    if sha256(directions) != '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18':
        raise ValueError('方向不是冻结的原第16层双极来源')
    payload = torch.load(directions, map_location='cpu', weights_only=True)
    manifest = json.loads(Path(experiences).read_text(encoding='utf-8'))
    episodes = manifest['episodes']
    validate_candidates(episodes, allow_grouped=True)
    validate_split_plan(episodes, make_plan())
    if (manifest.get('scope') != 'dev_optimization_only' or len(episodes) != 10
            or any(r['split'] != 'dev' for r in episodes)
            or [r['intended'] for r in episodes] != [{d:s} for d in DIMS for s in (-1,1)]):
        raise ValueError('只接受固定顺序、正负平衡的10段dev')
    if manifest['text_sha256'] != {r['id']:hashlib.sha256(r['text'].encode('utf-8')).hexdigest() for r in episodes}:
        raise ValueError('dev清单原文哈希不符')
    return payload['U'], payload['s'], episodes


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    plan = settings(args.preflight)
    out = Path(cfg['out_path']) / 'e1_kl_dev' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(task='C11', status='running', stage='validation', start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), git_dirty=bool(_git('status', '--porcelain')),
                config_path=str(Path(args.config).resolve()), config=_redacted(cfg), argv=_redacted(sys.argv[1:]),
                torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
                gpu=None, judge=None, preflight=args.preflight, plan=plan, formal_e1_pass=False,
                behavior_validated=False, data_hashes={}, episodes_completed=0)
    def save():
        write_json(out / 'run_meta.json', meta)
    def check():
        if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('运行期间输入改变')
    save()
    try:
        u, scales, episodes = load_inputs(args.directions, args.experiences)
        plan['learning_rates'] = learning_rates(u.shape[2])
        model_dir = (ROOT / cfg['model']).resolve()
        if not (model_dir / 'config.json').is_file():
            raise ValueError('只允许使用已下载本地模型')
        paths = [Path(args.config), Path(args.directions), Path(args.experiences),
                 ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_DEV_OPTIMIZATION_2026-10-04.md']
        paths += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        paths += [KIT / 'trait_extract' / f'{BIG_FIVE[k][0]}.json' for k in DIMS]
        paths += [p for p in model_dir.iterdir() if p.suffix in ('.json', '.txt', '.model', '.jinja')]
        meta['data_hashes'] = {str(p.resolve()): sha256(p) for p in paths}
        probes = probe_questions(30, seed=0)
        fit = probes[:plan['n_fit']]
        # 预检也从完整方案的留出区域取题，不改变最终20/10划分。
        evaluation = probes[20:20 + plan['n_eval']]
        if len(probes) != 30 or len(set(' '.join(p.casefold().split()) for p in probes)) != 30:
            raise ValueError('固定30道提取探针缺失或重复')
        write_json(out / 'probes.json', {'fit': fit, 'evaluation': evaluation, 'source': 'trait_extract'})
        torch.manual_seed(0)
        model, tok = load_model(str(model_dir), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if (model.config.num_hidden_layers, model.config.hidden_size) != (u.shape[0], u.shape[2]):
            raise ValueError('模型与方向形状不符')
        model.eval()
        meta['model_shape'] = {'num_hidden_layers': model.config.num_hidden_layers,
                               'hidden_size': model.config.hidden_size}
        meta['model_weights_sha256_verified'] = False
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        meta['parameter_count'] = sum(p.numel() for p in model.parameters())
        meta['all_parameters_frozen'] = all(not p.requires_grad for p in model.parameters())
        if not meta['all_parameters_frozen']:
            raise ValueError('模型权重未冻结')
        q = random_basis(u.shape[2])
        torch.save({'personality': u[LAYER], 'random': q, 's': scales[LAYER], 'layer': LAYER}, out / 'bases.pt')
        results = []
        for episode in episodes[:plan['n_episodes']]:
            check()
            folder = out / episode['id']
            folder.mkdir()
            meta.update(stage='teacher_corpus', episode=episode['id']); save()
            items, records = prepare_corpus(model, tok, episode['text'], fit + evaluation, plan['max_new_tokens'])
            write_json(folder / 'teacher_answers.json', records)
            torch.save([asdict(i) for i in items], folder / 'teacher_corpus.pt')
            fit_items, eval_items = items[:len(fit)], items[len(fit):]
            difference = fit_activation_difference(model, tok, episode['text'], fit)
            scale = float(scales[LAYER])
            d_a = projection_coordinates(difference, u[LAYER], scale)
            row = {'id': episode['id'], 'intended': episode['intended'], 'delta_A': d_a.tolist(),
                   'A_eval': evaluate_kl(model, eval_items, layer=LAYER, vector=scale * (d_a @ u[LAYER])), 'arms': {}}
            torch.save(difference, folder / 'fit_activation_difference.pt')
            for name, basis, rate in [(f'{space}_lr{index}', basis, rate)
                                      for space, basis in [('personality', u[LAYER]), ('random', q), ('full', None)]
                                      for index, rate in enumerate(plan['learning_rates'][space])]:
                check()
                meta.update(stage='fit', arm=name); save()
                init = difference / scale if basis is None else projection_coordinates(difference, basis, scale)
                result = fit_and_evaluate(model, fit_items, eval_items, basis=basis, scale=scale, layer=LAYER,
                                          steps=plan['steps'], lr=rate, lam=plan['lam'],
                                          probes_per_step=plan['probes_per_step'], seed=0, init=init, diagnostics=True)
                torch.save(result, folder / f'{name}.pt')
                row['arms'][name] = serializable(result)
                write_json(folder / 'results.json', row)
                print(f'[E1 KL] {episode["id"]}/{name} 留出KL {result["baseline"]["kl"]:.6f} → {result["final"]["kl"]:.6f}', flush=True)
            results.append(row)
            write_json(out / 'results.json', {'episodes': results, 'formal_e1_pass': False, 'preflight': args.preflight})
            meta['episodes_completed'] += 1; save()
            del items, fit_items, eval_items
        check()
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error=str(exc))
        raise
    finally:
        meta['end_time'] = now(); save()
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
