"""C19：固定C/N幅度的注入时段对照；仅calib，不据此改变正式干预机制。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import uuid
import zipfile

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.calibration import assess_point
from antler.eval.judge import Judge, COHERENCE_PROMPT, TASK_PROMPT
from antler.traits.data import BIG_FIVE, DIMS, KIT, load_trait
from antler.traits.vectors import steer_vector_poles
from e0_diagnostics import now, write_json, sha256
from e0_extract_coherence import _redacted
from e0_refilter import _git


LAYER = 21
COEFS = {'C': -.175, 'N': .165}
ARMS = ('zero', 'all', 'prefill', 'decode')
METRICS = ('trait', 'coherence', 'task')


def add_arguments(ap):
    ap.add_argument('--vectors-dir', required=True, help='已核验的完整paired向量目录')


def load_vectors(directory):
    """在加载大模型之前检查同一目录的尺度与四个极点向量。"""
    directory = Path(directory)
    directions = torch.load(directory / 'directions.pt', map_location='cpu', weights_only=True)
    u, scale = directions['U'], directions['s']
    if (directions.get('dims') != DIMS or u.ndim != 3 or u.shape[1] != 5
            or u.shape[0] <= LAYER or scale.shape != (u.shape[0],)
            or not torch.isfinite(u).all() or not torch.isfinite(scale).all() or scale[LAYER] <= 0):
        raise ValueError('方向维度、层号或尺度无效')
    vectors, files = {}, [directory / 'directions.pt']
    for dim in COEFS:
        poles = {}
        for pole in BIG_FIVE[dim]:
            path = directory / 'vectors' / f'{pole}.pt'
            tensor = torch.load(path, map_location='cpu', weights_only=True)
            if tensor.shape != (u.shape[0], u.shape[2]) or not torch.isfinite(tensor).all():
                raise ValueError('单极向量形状或数值无效')
            poles[pole] = tensor[LAYER]
            files.append(path)
        z = torch.zeros(5)
        z[DIMS.index(dim)] = COEFS[dim]
        vectors[dim] = steer_vector_poles(poles, float(scale[LAYER]), z)
    return vectors, tuple(u.shape), files


def validate_generation(items, expected, limit, eos_ids):
    """计数、EOS与原始token必须一致；不把截断当作自然结束。"""
    if len(items) != expected:
        raise ValueError('生成条数不符合预登记')
    for item in items:
        tokens = item['token_ids']
        if (not isinstance(item['text'], str) or not item['text'].strip() or not tokens
                or any(type(t) is not int or t < 0 for t in tokens)
                or len(tokens) != item['n_generated_tokens'] or len(tokens) > limit):
            raise ValueError('生成token或文本无效')
        first_eos = next((i for i, t in enumerate(tokens) if t in eos_ids), None)
        if first_eos is None:
            valid = item['finish_reason'] == 'length' and len(tokens) == limit
        else:
            valid = item['finish_reason'] == 'eos' and first_eos == len(tokens)-1
        if not valid:
            raise ValueError('EOS、长度与结束原因不一致')


def preflight(model, tok, vectors, traits, eos_ids):
    """仅8条16-token短预检；失败时不进入480条生成或评分。"""
    records = []
    for dim in COEFS:
        target = BIG_FIVE[dim][1 if dim == 'C' else 0]
        prompt = chat_prompt(tok, traits[target]['questions'][0])
        baseline = None
        for arm in ARMS:
            traces = []
            kwargs = {} if arm == 'zero' else {'steer_phase': arm}
            steer = None if arm == 'zero' else {LAYER: torch.zeros_like(vectors[dim])}
            items = generate(model, tok, [prompt], steer=steer, max_new_tokens=16,
                             temperature=1., top_p=1., batch_size=16, seed=0,
                             return_details=True, return_token_ids=True, phase_trace_out=traces, **kwargs)
            validate_generation(items, 1, 16, eos_ids)
            if baseline is None:
                baseline = items
            elif items != baseline:
                raise ValueError(f'{dim}零向量在{arm}模式下改变生成，停止')
            records.append({'dim': dim, 'arm': arm, 'generation': items[0], 'phase_trace': traces})
    return records


def validate_rows(rows, traits):
    """每臂每套题各30条；相同题目的三次生成不算三个独立题目。"""
    expected = {(d, p, a, q, s) for d in COEFS for p in ('high', 'low')
                for a in ARMS for q in range(10) for s in range(3)}
    actual = [(r['dim'], r['pole'], r['arm'], r['question_index'], r['sample_index']) for r in rows]
    if len(rows) != 480 or len(set(actual)) != 480 or set(actual) != expected:
        raise ValueError('实验身份缺失、重复或越过calib范围')
    for row in rows:
        target = BIG_FIVE[row['dim']][row['pole'] == 'low']
        if row['target'] != target or row['q'] != traits[target]['questions'][row['question_index']]:
            raise ValueError('题目与极点来源不一致')
        coefficient = 0 if row['arm'] == 'zero' else COEFS[row['dim']]
        if (row.get('split') != 'calib' or row.get('seed') != 0 or row.get('layer') != LAYER
                or row.get('axis') != 'pole' or row.get('coef') != coefficient):
            raise ValueError('实验条件与固定登记不一致')


def score_rows(judge, rows, traits, out):
    """一次固定乱序评分；相同完整输入去重，不根据中途分数追加请求。"""
    jobs = []
    for index, row in enumerate(rows):
        prompts = {'trait': traits[row['target']]['eval_prompt'],
                   'coherence': COHERENCE_PROMPT, 'task': TASK_PROMPT}
        for metric, template in prompts.items():
            jobs.append((index, metric, template.format(question=row['q'], answer=row['a'])))
    random.Random(0).shuffle(jobs)
    unique = list(dict.fromkeys(prompt for _, _, prompt in jobs))
    write_json(out / 'score_plan.json', {'seed': 0, 'n_inputs': len(jobs), 'n_unique': len(unique),
               'ordered_inputs': [{'row': i, 'metric': m, 'sha256': hashlib.sha256(p.encode()).hexdigest()}
                                  for i, m, p in jobs]})
    values = judge.score_many(unique)
    if len(values) != len(unique):
        raise ValueError('评分返回条数不一致')
    if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 100)
           for v in values):
        raise ValueError('评分不是0–100的有限数字或缺失值')
    scores = dict(zip(unique, values))
    for index, metric, prompt in jobs:
        rows[index][metric] = scores[prompt]
    return all(v is not None for v in values)


def analyze(rows):
    """相对零注入与all分别配对；不对时段差值套用正式可达判定。"""
    report = []
    for dim in COEFS:
        for pole in ('high', 'low'):
            groups = {a: [r for r in rows if (r['dim'], r['pole'], r['arm']) == (dim, pole, a)]
                      for a in ARMS}
            for arm in ARMS[1:]:
                versus_zero = assess_point(groups['zero'], groups[arm])
                versus_all = None if arm == 'all' else assess_point(groups['all'], groups[arm])['metrics']
                report.append({'dim': dim, 'pole': pole, 'arm': arm, 'versus_zero': versus_zero,
                               'versus_all': versus_all,
                               'finish_reasons': dict(Counter(r['finish_reason'] for r in groups[arm]))})
    return {'conditions': report, 'n_samples': len(rows), 'formal_e0_pass': False,
            'scope': '单seed的calib时段消融；评委质量尺度仍有争议，未测MMLU，不改变正式默认机制。'}


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    source = Path(args.vectors_dir).resolve()
    out = cfg['out_path'] / 'phase_diagnostics' / (now().replace(':', '').replace('+', '_')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C19', 'status': 'running', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'argv': sys.argv[1:], 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'judge': None, 'data_hashes': {},
            'vectors_dir': str(source), 'layer': LAYER, 'coefs': COEFS, 'arms': ARMS,
            'split': 'calib', 'seed': 0, 'n_samples_per_question': 3, 'max_new_tokens': 320,
            'batch_size': 16, 'temperature': 1., 'top_p': 1., 'expected_samples': 480,
            'preflight_samples': 8, 'preflight_tokens': 16, 'formal_e0_pass': False,
            'prompt_hashes': {name: hashlib.sha256(prompt.encode()).hexdigest()
                              for name, prompt in [('coherence', COHERENCE_PROMPT), ('task', TASK_PROMPT)]}}
    rows = []
    phase_traces = []

    def save():
        write_json(out / 'run_meta.json', meta)
        (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n'
                                                   for r in rows), encoding='utf-8', newline='\n')

    save()
    try:
        vectors, shape, inputs = load_vectors(source)
        traits = {p: load_trait(p, 'eval') for d in COEFS for p in BIG_FIVE[d]}
        if any(len(t['questions']) != 20 or len(set(t['questions'][:10])) != 10 for t in traits.values()):
            raise ValueError('题库不符合固定calib/test切分')
        inputs += [Path(args.config).resolve(), *[KIT / 'trait_eval' / f'{p}.json' for p in traits]]
        inputs += list((ROOT / 'antler').rglob('*.py')) + [Path(__file__).resolve()]
        model_dir = ROOT / cfg['model']
        inputs += sorted(model_dir.glob('*.json'))
        meta['data_hashes'] = {str(p): sha256(p) for p in inputs}
        # 先检查固定评委身份；初始化不会发出请求，缓存也不继承旧轮。
        judge = Judge(cache_path=str(out / 'judge_cache.jsonl'), response_audit_path=str(out / 'judge_responses.jsonl'))
        if judge.model != 'deepseek-flash':
            raise ValueError('本轮必须沿用deepseek-flash')
        meta['judge'] = _redacted({k: getattr(judge, k) for k in ('model', 'base_url', 'max_tokens', 'extra_body')})
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if (model.config.num_hidden_layers, model.config.hidden_size) != (shape[0], shape[2]):
            raise ValueError('模型与向量形状不一致')
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        if not eos_ids:
            raise ValueError('实际生成配置缺少EOS')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'generation_settings.json', model.generation_config.to_dict())
        write_json(out / 'preflight.json', preflight(model, tok, vectors, traits, eos_ids))
        meta['stage'] = 'generation'
        save()
        for dim in COEFS:
            for pole in ('high', 'low'):
                target = BIG_FIVE[dim][pole == 'low']
                questions = traits[target]['questions'][:10] * 3
                prompts = [chat_prompt(tok, q) for q in questions]
                for arm in ARMS:
                    trace = []
                    steer = None if arm == 'zero' else {LAYER: vectors[dim]}
                    kwargs = {} if arm == 'zero' else {'steer_phase': arm}
                    items = generate(model, tok, prompts, steer=steer, max_new_tokens=320,
                                     temperature=1., top_p=1., batch_size=16, seed=0,
                                     return_details=True, return_token_ids=True, phase_trace_out=trace, **kwargs)
                    validate_generation(items, 30, 320, eos_ids)
                    phase_traces.append({'dim': dim, 'pole': pole, 'arm': arm, 'batches': trace})
                    write_json(out / 'phase_traces.json', phase_traces)
                    for index, item in enumerate(items):
                        if tok.batch_decode([item['token_ids']], skip_special_tokens=True)[0] != item['text']:
                            raise ValueError('保存的token与回答文本不一致')
                        rows.append({'dim': dim, 'pole': pole, 'target': target, 'arm': arm,
                                     'layer': LAYER, 'coef': 0 if arm == 'zero' else COEFS[dim],
                                     'axis': 'pole', 'seed': 0, 'split': 'calib',
                                     'question_index': index % 10, 'sample_index': index // 10,
                                     'q': questions[index], 'a': item['text'],
                                     **{k: item[k] for k in ('token_ids', 'n_generated_tokens', 'finish_reason')}})
                    meta['generated'] = len(rows)
                    save()
                    print(f'[生成] {dim}/{pole}/{arm} 完成，总计{len(rows)}/480', flush=True)
        validate_rows(rows, traits)
        meta['stage'] = 'scoring'
        save()
        complete = score_rows(judge, rows, traits, out)
        save()
        if not complete:
            raise ValueError('存在缺失评分，保留原结果，不自动重评')
        if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('实验期间输入发生变化')
        write_json(out / 'assessment.json', analyze(rows))
        meta['status'] = 'completed'
        meta['stage'] = 'completed'
    except BaseException as exc:
        # 不把外部异常原文写入可分享产物，避免URL或认证信息进入归档。
        meta.update(status='failed', error_type=type(exc).__name__, error='实验失败，未自动重试；请核对本地日志与当前阶段。')
        raise
    finally:
        meta['end_time'] = now()
        save()
        names = ['run_meta.json', 'samples.jsonl', 'generation_settings.json', 'preflight.json',
                 'score_plan.json', 'judge_cache.jsonl', 'judge_responses.jsonl', 'assessment.json', 'phase_traces.json']
        manifest = {name: sha256(out / name) for name in names if (out / name).exists()}
        write_json(out / 'artifact_hashes.json', manifest)
        with zipfile.ZipFile(out / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in [*manifest, 'artifact_hashes.json']:
                archive.write(out / name, name)
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
