"""C11：固定pilot的共享老师token蒸馏与留出KL，不调用人格评委。"""
from dataclasses import asdict
from pathlib import Path
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.activations import last_prompt_acts
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model, model_device
from antler.memory.write import MEMORY_TEMPLATE, NEUTRAL_MEMORIES, _seq_logprobs, probe_questions, projection_coordinates
from antler.memory.kl_validation import KLItem, validate_items, evaluate_kl, fit_and_evaluate
from antler.traits.data import BIG_FIVE, DIMS, KIT
from e1_projection_diagnostic import load_inputs, LAYER
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git


def add_arguments(ap):
    ap.add_argument('--directions', required=True)
    ap.add_argument('--experiences', default='data/pilot_experiences.jsonl')
    ap.add_argument('--preflight', action='store_true', help='独立流程预检，不用于研究结论')


def settings(preflight):
    return dict(n_episodes=1 if preflight else 12, n_fit=2 if preflight else 20,
                n_eval=2 if preflight else 10, max_new_tokens=16 if preflight else 64,
                steps=2 if preflight else 60, probes_per_step=2 if preflight else 4,
                lr=.02, lam=.001, seed=0, layer=LAYER, steer_phase='all')


@torch.no_grad()
def prepare_corpus(model, tok, experience, probes, max_tokens):
    """只生成一次老师回答，直接保存实际token，包括EOS；不重分词回答。"""
    teacher = [chat_prompt(tok, q, system=MEMORY_TEMPLATE.format(experience=experience)) for q in probes]
    neutral = [[chat_prompt(tok, q, system=MEMORY_TEMPLATE.format(experience=m)) for m in NEUTRAL_MEMORIES]
               for q in probes]
    generated = generate(model, tok, teacher, max_new_tokens=max_tokens, temperature=0.,
                         batch_size=1, seed=0, return_details=True, return_token_ids=True)
    if len(generated) != len(probes):
        raise ValueError('老师回答数量不完整')
    items, records = [], []
    device = model_device(model)
    encode = lambda text: torch.tensor(tok(text, add_special_tokens=False)['input_ids'], dtype=torch.long)
    for q, prompt, controls, answer in zip(probes, teacher, neutral, generated):
        ids = torch.tensor(answer['token_ids'], dtype=torch.long)
        if len(ids) != answer['n_generated_tokens'] or not 0 < len(ids) <= max_tokens:
            raise ValueError('老师token为空或长度不一致')
        if tok.batch_decode([ids.tolist()], skip_special_tokens=True)[0] != answer['text']:
            raise ValueError('老师token与文本不一致')
        prompt_ids = encode(prompt)
        distribution = _seq_logprobs(model, prompt_ids.to(device), ids.to(device)).detach().cpu()
        items.append(KLItem(q, ids, distribution, tuple(encode(p) for p in controls)))
        records.append(dict(probe=q, teacher_prompt=prompt, teacher_prompt_ids=prompt_ids.tolist(),
                            neutral_prompts=controls, **answer))
    validate_items(items)
    return items, records


@torch.no_grad()
def fit_activation_difference(model, tok, experience, probes):
    render = lambda memory: [chat_prompt(tok, q, system=MEMORY_TEMPLATE.format(experience=memory)) for q in probes]
    teacher = last_prompt_acts(model, tok, render(experience), layers=[LAYER], batch_size=1)[:, 0]
    baseline = torch.stack([last_prompt_acts(model, tok, render(m), layers=[LAYER], batch_size=1)[:, 0]
                            for m in NEUTRAL_MEMORIES]).mean(0)
    difference = (teacher - baseline).mean(0).float().cpu()
    if not torch.isfinite(difference).all():
        raise ValueError('拟合探针激活差非有限')
    return difference


def random_basis(hidden):
    g = torch.Generator(device='cpu').manual_seed(0)
    return torch.linalg.qr(torch.randn(hidden, 5, generator=g, dtype=torch.float64), mode='reduced').Q.T.float()


def serializable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    plan = settings(args.preflight)
    out = Path(cfg['out_path']) / 'e1_kl_pilot' / uuid.uuid4().hex
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
        model_dir = (ROOT / cfg['model']).resolve()
        if not (model_dir / 'config.json').is_file():
            raise ValueError('只允许使用已下载本地模型')
        paths = [Path(args.config), Path(args.directions), Path(args.experiences),
                 ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_KL_PILOT_2026-10-04.md']
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
            for name, basis in [('personality', u[LAYER]), ('random', q), ('full', None)]:
                check()
                meta.update(stage='fit', arm=name); save()
                init = difference / scale if basis is None else projection_coordinates(difference, basis, scale)
                result = fit_and_evaluate(model, fit_items, eval_items, basis=basis, scale=scale, layer=LAYER,
                                          steps=plan['steps'], lr=plan['lr'], lam=plan['lam'],
                                          probes_per_step=plan['probes_per_step'], seed=0, init=init)
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
