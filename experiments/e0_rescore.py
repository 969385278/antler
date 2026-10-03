"""C8：用独立缓存复评已保存的calib回答；不生成文本、不替换原评分。"""
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.judge import Judge, COHERENCE_PROMPT, TASK_PROMPT
from antler.traits.data import BIG_FIVE, KIT, load_trait
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted


def add_arguments(ap):
    ap.add_argument('--sources', nargs='+', required=True, help='原始calibration_judge_samples.jsonl')
    ap.add_argument('--dims', nargs='+', choices=list(BIG_FIVE), required=True)
    ap.add_argument('--coefs', nargs='+', type=float, required=True)
    ap.add_argument('--audit-responses', action='store_true', help='保存评委响应身份与请求哈希，默认关闭')


def valid_score(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100)


def load_inputs(paths, dims, coefs):
    """全部校验后才调用评委；重复条件必须逐条相同，只复评一次。"""
    if not dims or not coefs or any(not math.isfinite(c) for c in coefs):
        raise ValueError('维度和有限系数列表不能为空')
    records, sources, reference, conditions = [], {}, None, {}
    for source in paths:
        source = Path(source).resolve()
        meta_path = source.parent / 'run_meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('status') != 'completed':
            raise ValueError('源实验必须已完成')
        identity = {k: meta.get(k) for k in ('model', 'judge', 'prompt_hashes')}
        identity['model'] = identity['model'] or meta.get('config', {}).get('model')
        if not identity['model'] or not identity['judge'] or not identity['prompt_hashes']:
            raise ValueError('源实验缺少模型、评委或提示哈希')
        if reference is not None and identity != reference:
            raise ValueError('源实验的模型、评委或提示不一致')
        reference = identity
        sources.update({str(source): sha256(source), str(meta_path): sha256(meta_path)})
        for line_index, line in enumerate(source.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row['dim'] not in dims or row['coef'] not in coefs:
                continue
            if row.get('split') != 'calib' or row.get('pole') not in ('high', 'low'):
                raise ValueError('复评仅允许calib高低极回答')
            trait_name = BIG_FIVE[row['dim']][row['pole'] == 'low']
            if row.get('trait_name') != trait_name:
                raise ValueError('维度与极点名称不匹配')
            samples = row['result']['samples']
            indices = [(s['question_index'], s['sample_index']) for s in samples]
            if not samples or len(set(indices)) != len(indices):
                raise ValueError('样本为空或题目采样下标重复')
            for s in samples:
                if (not isinstance(s.get('q'), str) or not isinstance(s.get('a'), str)
                        or any(m not in s or not valid_score(s[m]) for m in ('trait', 'coherence', 'task'))):
                    raise ValueError('回答或原评分格式无效')
            key = tuple(row[k] for k in ('layer', 'dim', 'coef', 'pole', 'axis', 'split', 'seed', 'max_new_tokens'))
            if key in conditions:
                if samples != conditions[key]['result']['samples']:
                    raise ValueError('重复条件的回答或评分不同，不能任选其一')
                continue
            record = copy.deepcopy(row)
            record['rescore_source'] = {'path': str(source), 'sha256': sources[str(source)], 'line_index': line_index}
            conditions[key] = record
            records.append(record)
    if not records:
        raise ValueError('没有符合条件的回答')
    return records, sources, reference


def rescore_records(records, judge, prompts):
    """逐条件保存旧分与新分，缺失不填零；原对象不修改。"""
    output = []
    for row in records:
        item = copy.deepcopy(row)
        samples = item['result']['samples']
        questions, answers = [s['q'] for s in samples], [s['a'] for s in samples]
        scored = {'trait': judge.trait_scores(prompts[row['trait_name']], questions, answers),
                  'coherence': judge.coherence_scores(questions, answers),
                  'task': judge.task_scores(questions, answers)}
        for metric, values in scored.items():
            if len(values) != len(samples) or any(not valid_score(v) for v in values):
                raise ValueError('复评分数数量或范围无效')
            for sample, score in zip(samples, values):
                sample.setdefault('original_scores', {})[metric] = sample[metric]
                sample[metric] = score
            valid = [v for v in values if v is not None]
            item['result'][metric] = sum(valid) / len(valid) if valid else None
            item['result']['n_valid' if metric == 'trait' else f'n_valid_{metric}'] = len(valid)
        item['measurement_role'] = 'same_text_rerating_only'
        item['replaces_original_scores'] = False
        output.append(item)
        print(f"[复评] seed={row['seed']} {row['dim']} {row['pole']} {row['coef']:+g}，{len(samples)}条", flush=True)
    return output


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    records, sources, identity = load_inputs(args.sources, args.dims, args.coefs)
    prompts = {r['trait_name']: load_trait(r['trait_name'], 'eval')['eval_prompt'] for r in records}
    fixed = {'COHERENCE_PROMPT': COHERENCE_PROMPT, 'TASK_PROMPT': TASK_PROMPT}
    if any(identity['prompt_hashes'].get(k) != hashlib.sha256(v.encode()).hexdigest() for k, v in fixed.items()):
        raise ValueError('当前固定评分提示与源实验不一致')
    for name in prompts:
        path = KIT / 'trait_eval' / f'{name}.json'
        sources[str(path)] = sha256(path)
        for source in args.sources:
            meta = json.loads((Path(source).parent/'run_meta.json').read_text(encoding='utf-8'))
            original = [v for k, v in meta.get('data_hashes', {}).items() if k.replace('\\', '/').endswith('/trait_eval/'+name+'.json')]
            if original != [sources[str(path)]]:
                raise ValueError('源实验题库哈希缺失或与当前评分提示不一致')
    out = cfg['out_path'] / 'rescore' / (now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    audit_responses = getattr(args, 'audit_responses', False)
    judge = Judge(cache_path=str(out/'judge_cache.jsonl'),
                  **({'response_audit_path': str(out/'judge_responses.jsonl')} if audit_responses else {}))
    judge_identity = {k: getattr(judge, k) for k in ('model', 'base_url', 'max_tokens', 'extra_body')}
    if _redacted(judge_identity) != identity['judge']:
        raise ValueError('当前评委配置与原评委不一致')
    meta = {'task': 'C8', 'experiment': 'e0_rescore', 'audit_responses': audit_responses, 'status': 'running', 'start_time': now(),
            'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'argv': _redacted(sys.argv[1:]), 'config': _redacted(cfg), 'config_sha256': sha256(Path(args.config)),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'judge': _redacted(judge_identity), 'data_hashes': sources, 'prompt_hashes': identity['prompt_hashes'],
            'trait_prompt_hashes': {k:hashlib.sha256(v.encode()).hexdigest() for k,v in prompts.items()},
            'n_records': len(records), 'n_answers': sum(len(r['result']['samples']) for r in records),
            'generation_performed': False, 'model_loaded': False, 'replaces_original_scores': False,
            'cache_initially_empty': not (out/'judge_cache.jsonl').exists()}
    write_json(out/'run_meta.json', meta)
    try:
        results = rescore_records(records, judge, prompts)
        if any(sha256(Path(p)) != h for p,h in sources.items()):
            raise RuntimeError('复评期间源文件变化')
        with (out/'rescored_samples.jsonl').open('w', encoding='utf-8', newline='\n') as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
        meta['status'] = 'completed'
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = now()
        if audit_responses:
            meta['response_audit_sha256'] = sha256(out/'judge_responses.jsonl') if (out/'judge_responses.jsonl').exists() else None
        write_json(out/'run_meta.json', meta)
    print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
