"""C12：固定六情境、24故事的无注入结果闸门，默认只预检。"""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.judge import Judge, parse_score
from antler.eval.twin_context_gate import DIMS_TESTED, MAX_TOKENS, N_ANSWERS, expand, jobs_for, summarize
from antler.traits.data import BIG_FIVE, KIT, load_trait
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model


DATA_SHA = '04a160c4c41c0ea9ebb7e97d61b7071136ab51a9b9c3b2c0ac7dcd17bea4e5cb'
PRIOR_JUDGE = 'outputs/main/prompt_persona/d09a117a119541a2985f54a77c8e4391/run_meta.json'
PROTOCOL = {'n_scenarios': 6, 'n_stories': 24, 'n_answers': N_ANSWERS, 'n_scores_max': 7560,
            'dims': list(DIMS_TESTED), 'split': 'calib', 'question_indices': list(range(10)),
            'max_new_tokens': MAX_TOKENS, 'temperature': 1., 'top_p': 1., 'batch_size': 8,
            'seeds': [0, 1, 2], 'score_shuffle_seed': 0, 'bootstrap_seed': 0,
            'bootstrap_samples': 10000, 'steer': None, 'retry': False}


class SequentialJudge(Judge):
    """接口故障立即停；不预先排队其他请求，不改变评委或提示。"""
    def _call(self, prompt, retries=1):
        if not hasattr(self, '_attempt_outputs'):
            self._attempt_outputs = {}
        key = self._key(prompt)
        if key in self._attempt_outputs:
            return self._attempt_outputs[key]
        output = super()._call(prompt, retries=1)
        # 无法解析的响应也保留到本批内存，重复输入不能再次请求。
        self._attempt_outputs[key] = output
        return output

    def score_many(self, prompts):
        scores = {p: parse_score(self._call(p)) for p in dict.fromkeys(prompts)}
        return [scores[p] for p in prompts]


def extra(ap):
    ap.add_argument('--materials', default='data/diagnostics/twin_outcome_dev_v1.json')
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--execute', action='store_true', help='固定预检之后执行，失败不自动重启')


def read_materials(path):
    if sha256(path) != DATA_SHA:
        raise ValueError('材料不是运行前封存版本')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    expand(data)
    return data


def seal_sources(out, hashes):
    """开始生成前保存源码和输入原文；不复制模型权重或密钥环境文件。"""
    for name, digest in hashes.items():
        path = Path(name).resolve()
        relative = path.relative_to(ROOT.resolve()) if path.is_relative_to(ROOT.resolve()) else (
            Path('external') / (digest[:12] + '_' + path.name))
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256(target) != digest:
            raise ValueError('来源快照哈希不一致')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    data = read_materials(args.materials)
    jobs = jobs_for(data)
    print(f'[结果闸门] 6情境/24故事/{len(jobs)}回答，执行={args.execute}', flush=True)
    if not args.execute:
        return {'protocol': PROTOCOL, 'n_jobs': len(jobs), 'model_loaded': False, 'judge_loaded': False}
    model_hashes = verify_model((ROOT / cfg['model']).resolve(), args.model_fingerprint)
    root = Path(cfg['out_path']) / 'e1_twin_context_gate'
    root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (DATA_SHA + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = {'task': 'C12', 'status': 'running', 'stage': 'validation', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()), 'config': _redacted(cfg),
            'argv': _redacted(sys.argv[1:]), 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'gpu': None, 'judge': None, 'seed': 0, 'protocol': PROTOCOL,
            'authorization': '作者同意新目标并提出六情境无注入上下文闸门；SPEC9.9有界calib用途',
            'generated': 0, 'scored': 0, 'model_hashes': model_hashes, 'data_hashes': {},
            'formal_e1_pass': False, 'behavior_validated': False, 'injection_performed': False}
    save = lambda: write_json(out / 'run_meta.json', meta)
    rows = []
    save()
    print(f'[结果闸门目录] {out}', flush=True)
    try:
        judge = SequentialJudge(cache_path=str(out / 'judge_cache.jsonl'), max_workers=1,
                                response_audit_path=str(out / 'judge_responses.jsonl'))
        prior_path = ROOT / PRIOR_JUDGE
        prior = json.loads(prior_path.read_text(encoding='utf-8'))['judge']
        actual = {'model': judge.model, 'base_url': judge.base_url,
                  'max_tokens': judge.max_tokens, 'extra_body': judge.extra_body}
        if actual != prior:
            raise ValueError('主评委设置与既有封存配置不一致')
        paths = [Path(args.config).resolve(), Path(args.materials).resolve(),
                 Path(args.model_fingerprint).resolve(), prior_path,
                 ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_TWIN_CONTEXT_GATE_2026-10-05.md']
        paths += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        paths += [KIT / 'trait_eval' / f'{name}.json' for d in DIMS_TESTED for name in BIG_FIVE[d]]
        meta.update(judge=actual, data_hashes={str(p.resolve()): sha256(p) for p in paths})
        save()
        write_json(out / 'jobs.json', jobs)
        write_json(out / 'stories.json', expand(data))
        seal_sources(out, meta['data_hashes'])
        def check():
            if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('固定材料、配置、题目或源码发生变化')
        check()
        model, tok = load_model(str(ROOT / cfg['model']), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型须冻结且eval')
        gc = getattr(model, 'generation_config', model.config)
        meta.update(stage='generate', all_parameters_frozen=True,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    stop_config={'model_eos_token_id': gc.eos_token_id,
                                 'tokenizer_eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id})
        save()
        for offset in range(0, len(jobs), 10):
            check()
            batch = jobs[offset:offset + 10]
            prompts = [chat_prompt(tok, row['q'], system=row['system']) for row in batch]
            answers = generate(model, tok, prompts, max_new_tokens=MAX_TOKENS, temperature=1., top_p=1.,
                               seed=batch[0]['seed'], batch_size=8, return_details=True, return_token_ids=True)
            if len(answers) != 10:
                raise ValueError('生成返回数量与工作清单不符')
            with (out / 'answers.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                for row, prompt, answer in zip(batch, prompts, answers):
                    n = answer['n_generated_tokens']
                    if (type(n) is not int or n != len(answer['token_ids']) or not 0 < n <= MAX_TOKENS
                            or answer['finish_reason'] not in ('eos', 'length')):
                        raise ValueError('原始token计数或停止状态异常')
                    record = {**row, 'prompt': prompt, 'a': answer['text'], **{k: answer[k] for k in
                              ('token_ids', 'n_generated_tokens', 'finish_reason')}}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    rows.append(record)
            meta['generated'] = len(rows)
            save()
            if len(rows) % 100 == 0 or len(rows) == len(jobs):
                print(f'[双胞胎生成] {len(rows)}/{N_ANSWERS}', flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        meta['stage'] = 'score'
        save()
        for dim in DIMS_TESTED:
            for pole in ('high', 'low'):
                check()
                group = [row for row in rows if row['dim'] == dim and row['pole'] == pole]
                random.Random(0).shuffle(group)
                write_json(out / f'{dim}_{pole}_score_order.json', [row['id'] for row in group])
                trait_prompt = load_trait(BIG_FIVE[dim][pole == 'low'], 'eval')['eval_prompt']
                for metric in ('trait', 'coherence', 'task'):
                    for rank, row in enumerate(group):
                        check() if rank % 50 == 0 else None
                        q, answer = [row['q']], [row['a']]
                        value = (judge.trait_scores(trait_prompt, q, answer) if metric == 'trait'
                                 else judge.coherence_scores(q, answer) if metric == 'coherence'
                                 else judge.task_scores(q, answer))[0]
                        row[metric] = value
                        row['judge_order_index'] = rank
                        with (out / 'score_records.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                            f.write(json.dumps({'id': row['id'], 'metric': metric, 'score': value,
                                                'dim': dim, 'pole': pole, 'judge_order_index': rank}) + '\n')
                        meta['scored'] += 1
                        if meta['scored'] % 50 == 0:
                            save()
                    write_json(out / 'scored_answers.json', rows)
                    save()
                    print(f'[双胞胎评分] {dim}/{pole}/{metric} {meta["scored"]}/7560', flush=True)
        write_json(out / 'summary.json', summarize(rows, data))
        check()
        if any(sha256(p) != h for p, h in model_hashes.items()):
            raise ValueError('模型文件发生变化')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error='本批停止，保留已有证据，不自动重试。')
        raise
    finally:
        if any(any(m in r for m in ('trait', 'coherence', 'task')) for r in rows):
            write_json(out / 'scored_answers.json', rows)
        meta['generated'] = len(rows)
        meta['end_time'] = now()
        save()
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
