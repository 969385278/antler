"""C11：已授权的固定dev/calib无注入诊断，默认只预检，无自动重试。"""
import hashlib
import json
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.judge import Judge
from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from antler.traits.data import BIG_FIVE, DIMS, KIT, load_trait
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model

EPISODES_SHA = 'c73dfaf15f4627b253f77f2ffc8bf0777ff8171a40077bf55902890e0cbc405c'
PLAN_SHA = 'b0426daf6e5136d8047d2c8e7a62118dd82696fd003aa331e6dbb5388f79d3cf'


def extra(ap):
    ap.add_argument('--experiences', required=True)
    ap.add_argument('--plan', required=True, help='原封存提案，授权见规格与登记')
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--execute', action='store_true')


class SingleAttemptJudge(Judge):
    def _call(self, prompt, retries=1):
        return super()._call(prompt, retries=1)


def jobs_for(episodes):
    if (len(episodes) != 10 or [e['intended'] for e in episodes] != [{d: s} for d in DIMS for s in (-1, 1)]
            or any(e['split'] != 'dev' for e in episodes) or len({e['id'] for e in episodes}) != 10):
        raise ValueError('只能使用固定十段正负dev经历')
    jobs = []
    for dim in DIMS:
        for pole in ('high', 'low'):
            data = load_trait(BIG_FIVE[dim][pole == 'low'], 'eval')
            if len(data['questions']) != 20:
                raise ValueError('评测题数不符固定calib/test切分')
            arms = [('none', None, None, BASE_SYSTEM)]
            arms += [('neutral', i, None, MEMORY_TEMPLATE.format(experience=m)) for i, m in enumerate(NEUTRAL_MEMORIES)]
            arms += [('context', None, e['id'], MEMORY_TEMPLATE.format(experience=e['text']))
                     for e in episodes if dim in e['intended']]
            for arm, memory, episode, system in arms:
                for seed in (0, 1, 2):
                    for q in range(10):
                        jobs.append(dict(id=len(jobs), dim=dim, pole=pole, split='calib', arm=arm,
                                         memory_index=memory, episode_id=episode, seed=seed,
                                         sample_index=seed, question_index=q, q=data['questions'][q], system=system))
    if len(jobs) != 2400:
        raise ValueError('超出固定2400回答预算')
    return jobs


def summarize(rows, episodes):
    expected = jobs_for(episodes)
    if len(rows) != len(expected) or any(any(r.get(k) != v for k, v in job.items()) for r, job in zip(rows, expected)):
        raise ValueError('回答身份、提示或完整性与固定计划不符')
    indexed = {(r['dim'], r['pole'], r['arm'], r['memory_index'], r['episode_id'], r['question_index'], r['seed']): r for r in rows}
    reports = []
    indices = np.random.default_rng(0).integers(0, 10, size=(10000, 10))
    for e in episodes:
        dim, intended = next(iter(e['intended'].items()))
        for pole in ('high', 'low'):
            def group(arm, memory=None, episode=None):
                return [indexed[dim, pole, arm, memory, episode, q, s] for q in range(10) for s in (0, 1, 2)]
            context = group('context', episode=e['id'])
            neutral = [group('neutral', i) for i in range(5)]
            none = group('none')
            selected = context + none + [r for g in neutral for r in g]
            missing = sum(any(type(r.get(m)) not in (int, float) or not np.isfinite(r[m]) or not 0 <= r[m] <= 100
                              for m in ('trait', 'coherence', 'task')) for r in selected)
            record = {'id': e['id'], 'dim': dim, 'pole': pole, 'missing_or_invalid_rows': missing}
            if missing:
                record['status'] = 'incomplete_scores'
            else:
                metrics = {}
                for metric in ('trait', 'coherence', 'task'):
                    mean_questions = lambda rs: np.array([r[metric] for r in rs], dtype=float).reshape(10, 3).mean(1)
                    c, b, n = mean_questions(context), np.mean([mean_questions(g) for g in neutral], axis=0), mean_questions(none)
                    diff = c - b
                    metrics[metric] = {'context': float(c.mean()), 'neutral': float(b.mean()), 'none': float(n.mean()),
                                       'context_minus_neutral': float(diff.mean()),
                                       'question_bootstrap_ci': np.quantile(diff[indices].mean(1), [.025, .975]).tolist()}
                record.update(status='descriptive', metrics=metrics,
                              context_matches_intended=metrics['trait']['context_minus_neutral'] * intended * (1 if pole == 'high' else -1) > 0)
            reports.append(record)
    return {'scope': 'dev_context_only_calib_diagnostic', 'reports': reports, 'formal_e1_pass': False,
            'behavior_validated': False, 'limitation': '十段仅四家族；题目区间不是跨家族区间；无注入，不证明迁移'}


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    if sha256(args.experiences) != EPISODES_SHA or sha256(args.plan) != PLAN_SHA:
        raise ValueError('经历或计划不是原封存版本')
    episodes = json.loads(Path(args.experiences).read_text(encoding='utf-8'))['episodes']
    jobs = jobs_for(episodes)
    print(f'[上下文诊断] 固定{len(jobs)}回答，无注入，执行={args.execute}', flush=True)
    if not args.execute:
        return None
    model_hashes = verify_model((ROOT / cfg['model']).resolve(), args.model_fingerprint)
    root = Path(cfg['out_path']) / 'e1_context_control'; root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (PLAN_SHA + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = dict(task='C11', status='running', stage='load', start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), config=_redacted(cfg), config_path=str(Path(args.config).resolve()),
                authorization='作者明确回复都批准；规格5.3有界calib例外',
                generated=0, scored=0, formal_e1_pass=False, behavior_validated=False,
                generation={'max_new_tokens': 320, 'temperature': 1., 'batch_size': 8, 'seeds': [0, 1, 2], 'steer': None},
                model_hashes=model_hashes, data_hashes={}, judge=None)
    save = lambda: write_json(out / 'run_meta.json', meta)
    save()
    try:
        judge = SingleAttemptJudge(cache_path=str(out/'judge_cache.jsonl'), response_audit_path=str(out/'judge_responses.jsonl'), max_workers=1)
        prior_path = ROOT/'outputs/main/prompt_persona/d09a117a119541a2985f54a77c8e4391/run_meta.json'
        prior = json.loads(prior_path.read_text())['judge']
        actual_judge = {'model': judge.model, 'base_url': judge.base_url, 'max_tokens': judge.max_tokens, 'extra_body': judge.extra_body}
        if actual_judge != prior:
            raise ValueError('评委设置与封存主评委不一致')
        paths = [Path(args.config), Path(args.experiences), Path(args.plan), Path(args.model_fingerprint), prior_path,
                 ROOT/'docs/RESEARCH_SPEC.md', ROOT/'docs/E1_CONTEXT_CONTROL_PROPOSAL_2026-10-04.md']
        paths += list((ROOT/'antler').rglob('*.py')) + list((ROOT/'experiments').glob('*.py'))
        paths += [KIT/'trait_eval'/f'{name}.json' for pair in BIG_FIVE.values() for name in pair]
        meta.update(judge=actual_judge, data_hashes={str(p.resolve()): sha256(p) for p in paths})
        save(); write_json(out/'jobs.json', jobs)
        def check():
            if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('来源或源码运行期间改变')
        model, tok = load_model(str(ROOT/cfg['model']), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型必须冻结且eval')
        meta.update(gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    all_parameters_frozen=True, stage='generate'); save()
        rows = []
        for offset in range(0, len(jobs), 10):
            check()
            batch = jobs[offset:offset+10]
            prompts = [chat_prompt(tok, r['q'], system=r['system']) for r in batch]
            answers = generate(model, tok, prompts, max_new_tokens=320, temperature=1., seed=batch[0]['seed'],
                               batch_size=8, return_details=True, return_token_ids=True)
            if len(answers) != len(batch):
                raise ValueError('生成返回数不符')
            with (out/'answers.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                for r, prompt, a in zip(batch, prompts, answers):
                    row = {**r, 'prompt': prompt, 'a': a['text'], **{k: a[k] for k in ('token_ids', 'n_generated_tokens', 'finish_reason')}}
                    f.write(json.dumps(row, ensure_ascii=False)+'\n'); rows.append(row)
            meta['generated'] = len(rows); save()
            if len(rows) % 100 == 0:
                print(f'[上下文生成] {len(rows)}/2400', flush=True)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        meta['stage'] = 'score'; save()
        for dim in DIMS:
            for pole in ('high', 'low'):
                check()
                group = [r for r in rows if r['dim'] == dim and r['pole'] == pole]
                random.Random(0).shuffle(group)
                write_json(out/f'{dim}_{pole}_score_order.json', [r['id'] for r in group])
                qs, answers = [r['q'] for r in group], [r['a'] for r in group]
                data = load_trait(BIG_FIVE[dim][pole == 'low'], 'eval')
                for metric, call in [('trait', lambda: judge.trait_scores(data['eval_prompt'], qs, answers)),
                                     ('coherence', lambda: judge.coherence_scores(qs, answers)),
                                     ('task', lambda: judge.task_scores(qs, answers))]:
                    scores = call()
                    if len(scores) != len(group): raise ValueError('评分数量不符')
                    for rank, (row, value) in enumerate(zip(group, scores)):
                        row[metric] = value; row['judge_order_index'] = rank
                    meta['scored'] += len(scores)
                    write_json(out/'scored_answers.json', rows); save()
                print(f'[上下文评分] {dim}/{pole}完成', flush=True)
        write_json(out/'summary.json', summarize(rows, episodes))
        check()
        if any(sha256(p) != h for p, h in model_hashes.items()): raise ValueError('模型文件改变')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        meta['end_time'] = now(); save()
        write_json(out/'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
