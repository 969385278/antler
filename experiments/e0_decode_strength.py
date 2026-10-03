"""C19：固定1800条decode强度诊断，只使用calib，不修改正式机制或门槛。"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import stdev
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
from antler.eval.stats import benjamini_hochberg
from antler.traits.data import BIG_FIVE, KIT, load_trait
from e0_diagnostics import now, write_json, sha256
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e0_phase_diagnostics import load_vectors, validate_generation, score_rows


LAYER = 21
SEEDS = (1, 2, 3)
ARMS = ('zero', 'all', 'decode_base', 'decode_mid', 'decode_upper')
DECODE_ARMS = ARMS[2:]
METRICS = ('trait', 'coherence', 'task')
COEFS = {
    'C': dict(zip(ARMS, (0., -.175, -.175, -.225, -.300))),
    'N': dict(zip(ARMS, (0., .165, .165, .200, .250))),
}
REGISTERED_VECTORS = Path('outputs/main/refilter_diagnostics/20261003T022259Z_099e5cd2/variants/paired')
REGISTERED_HASHES = {
    'directions.pt': '5e99e6e40abbdcd6f5c3d3833f0d0c24c4d41ef676fc0b0c923afabc4db7cfaa',
    'vectors/careless.pt': 'cc6c27f273e0e78d3ed069ab51ea360f00ea1e76d8ec0bfcf749d003c4e6b539',
    'vectors/nervous.pt': '4caa81abb73a3ee01b6bde595dab19a7f879057d7190d7325468d955b4f9f507',
}


def add_arguments(ap):
    ap.add_argument('--vectors-dir', required=True, help='登记的完整paired向量目录')


def arm_phase(arm):
    if arm not in ARMS:
        raise ValueError('实验臂不在固定登记中')
    return 'decode' if arm in DECODE_ARMS else 'all'


def arm_vector(vectors, dim, arm):
    """原幅度保持原张量；较大幅度只按同极点的绝对系数比缩放。"""
    if arm == 'zero':
        return None
    coef = COEFS[dim][arm]
    base = COEFS[dim]['all']
    return {LAYER: vectors[dim] if coef == base else vectors[dim] * abs(coef / base)}


def validate_rows(rows, traits):
    """每个题目×种子×采样身份恰好一次；只允许登记的1800条calib回答。"""
    expected = {(d, p, a, seed, q, sample) for d in COEFS for p in ('high', 'low')
                for a in ARMS for seed in SEEDS for q in range(10) for sample in range(3)}
    actual = []
    for row in rows:
        if any(type(row.get(k)) is not int for k in ('seed', 'question_index', 'sample_index', 'layer')):
            raise ValueError('实验身份必须是整数，不能使用布尔值或浮点数')
        identity = tuple(row.get(k) for k in ('dim', 'pole', 'arm', 'seed', 'question_index', 'sample_index'))
        if identity not in expected:
            raise ValueError('实验身份越过固定calib范围')
        actual.append(identity)
        target = BIG_FIVE[row['dim']][row['pole'] == 'low']
        if row.get('target') != target or row.get('q') != traits[target]['questions'][row['question_index']]:
            raise ValueError('题目与极点来源不一致')
        if (row.get('split') != 'calib' or row['layer'] != LAYER or row.get('axis') != 'pole'
                or type(row.get('coef')) not in (int, float) or row['coef'] != COEFS[row['dim']][row['arm']]
                or row.get('steer_phase') != arm_phase(row['arm'])):
            raise ValueError('实验条件与固定登记不一致')
    if len(actual) != 1800 or len(set(actual)) != 1800 or set(actual) != expected:
        raise ValueError('1800条实验身份缺失或重复')


def validate_phase_traces(batches, items, phase, limit):
    """依据有效缓存记录核验阶段；允许GPU延迟停止产生EOS后的额外forward。"""
    if phase not in ('all', 'prefill', 'decode') or not isinstance(batches, list):
        raise ValueError('缓存阶段或轨迹格式无效')
    starts = list(range(0, len(items), 16))
    if len(batches) != len(starts):
        raise ValueError('缓存轨迹批次数不一致')
    for batch, start in zip(batches, starts):
        if (set(batch) != {'batch_start', 'trace'} or type(batch['batch_start']) is not int
                or batch['batch_start'] != start or not isinstance(batch['trace'], list)):
            raise ValueError('缓存轨迹批次身份无效')
        trace = batch['trace']
        if phase == 'all':
            if trace:
                raise ValueError('默认all路径不应产生阶段轨迹')
            continue
        longest = max(item['n_generated_tokens'] for item in items[start:start+16])
        if not longest <= len(trace) <= limit:
            raise ValueError('缓存forward数与生成长度不一致')
        previous = None
        for i, entry in enumerate(trace):
            if set(entry) != {'phase', 'cached_tokens', 'cache_position_start', 'applied'}:
                raise ValueError('缓存轨迹字段不完整')
            cached, position = entry['cached_tokens'], entry['cache_position_start']
            stage = 'prefill' if i == 0 else 'decode'
            if (entry['phase'] != stage or type(cached) is not int or cached < 0
                    or (i == 0 and cached != 0) or (i > 0 and cached <= 0)
                    or (i > 1 and cached != previous + 1)
                    or (position is not None and (type(position) is not int or position != cached))
                    or type(entry['applied']) is not bool or entry['applied'] != (stage == phase)):
                raise ValueError('缓存阶段、位置或注入状态不一致')
            previous = cached


def preflight(model, tok, vectors, traits, eos_ids, out):
    """复用既有8条16-token零向量预检设计，逐条保存后再核验。"""
    records = []
    for dim in COEFS:
        target = BIG_FIVE[dim][1 if dim == 'C' else 0]
        prompt = chat_prompt(tok, traits[target]['questions'][0])
        baseline = None
        for arm in ('zero', 'all', 'prefill', 'decode'):
            trace = []
            phase = 'all' if arm == 'zero' else arm
            steer = None if arm == 'zero' else {LAYER: torch.zeros_like(vectors[dim])}
            items = generate(model, tok, [prompt], steer=steer, steer_phase=phase,
                             max_new_tokens=16, temperature=1., top_p=1., batch_size=16, seed=0,
                             return_details=True, return_token_ids=True, phase_trace_out=trace)
            records.append({'dim': dim, 'arm': arm, 'generations': items, 'phase_trace': trace})
            write_json(out / 'preflight.json', records)
            validate_generation(items, 1, 16, eos_ids)
            validate_phase_traces(trace, items, phase, 16)
            if tok.batch_decode([items[0]['token_ids']], skip_special_tokens=True)[0] != items[0]['text']:
                raise ValueError('预检token与文本不一致')
            if baseline is None:
                baseline = items
            elif items != baseline:
                raise ValueError('零向量在阶段模式下改变生成，停止')
    return records


def exact_sign_flip(differences):
    """十题双侧精确符号翻转；零假设下题目差值须独立且关于零对称、符号可交换。"""
    if len(differences) != 10 or any(not math.isfinite(v) for v in differences):
        raise ValueError('精确符号翻转需要十个有限题目差值')
    values = [v if isinstance(v, Fraction) else Fraction(str(float(v))) for v in differences]
    observed = abs(sum(values, Fraction(0)))
    extreme = sum(abs(sum((v * sign for v, sign in zip(values, signs)), Fraction(0))) >= observed
                  for signs in itertools.product((-1, 1), repeat=10))
    return extreme / 1024


def _question_differences(left, right, metric):
    """有理数聚合避免符号翻转中的相等统计量被浮点尾差拆开。"""
    def means(rows):
        return {q: sum((Fraction(str(float(r[metric]))) for r in rows if r['question_index'] == q),
                       Fraction(0)) / sum(r['question_index'] == q for r in rows) for q in range(10)}
    a, b = means(left), means(right)
    return [b[q] - a[q] for q in range(10)]


def _candidate(poles, target_pole, threshold):
    target = poles[target_pole]['trait']
    checks = {'target_trait_gain': target['mean'] >= threshold,
              'target_trait_ci_low_gt10': target['ci_low'] > 10}
    uncertain = []
    for pole, metrics in poles.items():
        checks[pole + '_coherence_absolute_ge75'] = metrics['coherence']['injected'] >= 75
        for metric in ('coherence', 'task'):
            value = metrics[metric]
            checks[pole + '_' + metric + '_drop_le10'] = value['mean'] >= -10
            if value['ci_low'] <= -10 <= value['ci_high']:
                uncertain.append({'pole': pole, 'metric': metric,
                                  'ci_low': value['ci_low'], 'ci_high': value['ci_high']})
    return {'trait_mean_cutoff': threshold, 'checks': checks, 'numeric_criteria_met': all(checks.values()),
            'failed_checks': [key for key, passed in checks.items() if not passed],
            'quality_boundary_uncertainty': uncertain, 'formal_e0_pass': False}


def analyze(rows, traits):
    """题内三个采样和三个种子等权，再对十题配对bootstrap；所有条件完整报告。"""
    validate_rows(rows, traits)
    if any(type(r.get(m)) not in (int, float) or not math.isfinite(r[m]) or not 0 <= r[m] <= 100
           for r in rows for m in METRICS):
        raise ValueError('分析要求三项评分全部完整有效')
    groups = {(d, p, a): [r for r in rows if (r['dim'], r['pole'], r['arm']) == (d, p, a)]
              for d in COEFS for p in ('high', 'low') for a in ARMS}
    conditions, indexed = [], {}
    for dim in COEFS:
        for pole in ('high', 'low'):
            for arm in ARMS[1:]:
                right = groups[dim, pole, arm]
                condition = {'dim': dim, 'pole': pole, 'arm': arm, 'coef': COEFS[dim][arm],
                             'pooled': {}, 'per_seed': [], 'seed_stddev': {},
                             'finish_reasons': dict(Counter(r['finish_reason'] for r in right)),
                             'formal_e0_pass': False}
                references = ('zero',) if arm == 'all' else ('zero', 'all')
                for reference in references:
                    condition['pooled']['versus_' + reference] = assess_point(groups[dim, pole, reference], right)['metrics']
                for seed in SEEDS:
                    seed_result = {'seed': seed, 'formal_e0_pass': False}
                    for reference in references:
                        seed_result['versus_' + reference] = assess_point(
                            [r for r in groups[dim, pole, reference] if r['seed'] == seed],
                            [r for r in right if r['seed'] == seed])['metrics']
                    condition['per_seed'].append(seed_result)
                for reference in references:
                    key = 'versus_' + reference
                    condition['seed_stddev'][key] = {m: stdev(s[key][m]['mean'] for s in condition['per_seed'])
                                                     for m in METRICS}
                indexed[dim, pole, arm] = condition
                conditions.append(condition)
    quality_tests = []
    for dim in COEFS:
        for arm in DECODE_ARMS:
            for pole in ('high', 'low'):
                for metric in ('coherence', 'task'):
                    differences = _question_differences(groups[dim, pole, 'all'], groups[dim, pole, arm], metric)
                    quality_tests.append({'dim': dim, 'arm': arm, 'pole': pole, 'metric': metric,
                                          'question_differences': [float(d) for d in differences],
                                          'p': exact_sign_flip(differences),
                                          **indexed[dim, pole, arm]['pooled']['versus_all'][metric]})
    correction = benjamini_hochberg([test['p'] for test in quality_tests], q=.05)
    for test, adjusted, reject in zip(quality_tests, correction['adjusted_p'], correction['reject']):
        target_pole = 'low' if test['dim'] == 'C' else 'high'
        trait = indexed[test['dim'], target_pole, test['arm']]['pooled']['versus_all']['trait']
        ci_close = trait['ci_low'] >= -5 and trait['ci_high'] <= 5
        test.update(adjusted_p=adjusted, reject=reject, formal_e0_pass=False,
                    quality_improvement_supported=test['ci_low'] > 0 and reject,
                    similar_trait_quality_improvement_supported=ci_close and test['ci_low'] > 0 and reject)
    candidates, selected, sensitivity = [], {}, []
    for dim in COEFS:
        target_pole = 'low' if dim == 'C' else 'high'
        selected[dim] = None
        for arm in DECODE_ARMS:
            poles = {p: indexed[dim, p, arm]['pooled']['versus_zero'] for p in ('high', 'low')}
            target_all = indexed[dim, target_pole, arm]['pooled']['versus_all']['trait']
            pooled = _candidate(poles, target_pole, 20)
            seed_results = []
            for i, seed in enumerate(SEEDS):
                by_seed = {p: indexed[dim, p, arm]['per_seed'][i]['versus_zero'] for p in ('high', 'low')}
                seed_results.append({'seed': seed, **_candidate(by_seed, target_pole, 20)})
            candidates.append({'dim': dim, 'arm': arm, 'coef': COEFS[dim][arm], 'target_pole': target_pole,
                               'pooled': pooled, 'per_seed': seed_results, 'formal_e0_pass': False,
                               'target_trait_versus_all': {**target_all,
                                   'mean_close_descriptive': abs(target_all['mean']) <= 5,
                                   'ci_within_minus5_plus5': target_all['ci_low'] >= -5 and target_all['ci_high'] <= 5,
                                   'note': '均值接近不证明等效；相近效应下质量改善还须对应质量CI下限>0及24项BH支持。'}})
            for cutoff in (20, 15, 10):
                sensitivity.append({'dim': dim, 'arm': arm, **_candidate(poles, target_pole, cutoff)})
            if pooled['numeric_criteria_met'] and (selected[dim] is None
                    or abs(COEFS[dim][arm]) < abs(selected[dim]['coef'])):
                selected[dim] = {'arm': arm, 'coef': COEFS[dim][arm], 'formal_e0_pass': False}
    return {'conditions': conditions, 'candidates': candidates, 'selected': selected, 'sensitivity': sensitivity,
            'quality_comparisons': {'family': '2维×3档decode×2极×连贯/任务，相对同轮all，共24项',
                'n_tests': 24, 'q': .05, 'method': '双侧精确符号翻转，枚举2^10种符号；BH校正',
                'assumption': '零假设下十个题目差值独立、关于零对称，符号可交换；不保证一般相关结构下的BH控制。',
                'tests': quality_tests},
            'n_samples': len(rows), 'seeds': list(SEEDS), 'seed_stddev_ddof': 1,
            'n_questions': 10, 'n_boot': 10000, 'formal_e0_pass': False,
            'notes': ['合并先在题内平均三个采样及三个种子，bootstrap分析单位始终为10道题。',
                      '候选筛选是calib描述性数字检查；质量CI跨−10只标边界不确定，不新增门槛。',
                      '敏感性仅改变人格均值20/15/10；10分仍受CI下限>10限制，不更改正式标准。',
                      '旧seed0不进入统计；seed1/2的zero/all可与旧文本相同，不能声称新增独立种子。',
                      '未检查有效decode干预下能力；首token MMLU不暴露于decode，不宣布能力或E0通过。',
                      '完成固定矩阵即停止；未找到候选只适用于已测范围，不追加网格或重评。']}


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    source = Path(args.vectors_dir).resolve()
    out = cfg['out_path'] / 'decode_strength' / (now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C19', 'status': 'running', 'stage': 'validation', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'argv': sys.argv[1:], 'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'judge': None, 'data_hashes': {},
            'vectors_dir': str(source), 'layer': LAYER, 'coefs': COEFS, 'arms': ARMS,
            'split': 'calib', 'seeds': SEEDS, 'n_samples_per_question': 3, 'max_new_tokens': 320,
            'batch_size': 16, 'temperature': 1., 'top_p': 1., 'expected_samples': 1800,
            'preflight_samples': 8, 'preflight_tokens': 16, 'max_score_inputs': 5400,
            'score_shuffle_seed': 0, 'formal_e0_pass': False,
            'prompt_hashes': {name: hashlib.sha256(prompt.encode()).hexdigest()
                              for name, prompt in [('coherence', COHERENCE_PROMPT), ('task', TASK_PROMPT)]}}
    rows, phase_traces = [], []

    def save():
        write_json(out / 'run_meta.json', meta)
        (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                                   for r in rows), encoding='utf-8', newline='\n')

    def check_sources():
        if any(sha256(p) != digest for p, digest in meta['data_hashes'].items()):
            raise ValueError('实验期间输入发生变化')

    save()
    try:
        if source != (ROOT / REGISTERED_VECTORS).resolve():
            raise ValueError('向量目录不符合本轮固定登记')
        if any(sha256(source / name) != digest for name, digest in REGISTERED_HASHES.items()):
            raise ValueError('向量源哈希与已核验登记不一致')
        model_dir = ROOT / cfg['model']
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise ValueError('本轮要求已下载的本地模型目录及config.json')
        vectors, shape, inputs = load_vectors(source)
        traits = {p: load_trait(p, 'eval') for d in COEFS for p in BIG_FIVE[d]}
        if any(len(t['questions']) != 20 or len(set(t['questions'][:10])) != 10 for t in traits.values()):
            raise ValueError('题库不符合固定calib/test切分')
        meta['prompt_hashes'].update({p: hashlib.sha256(t['eval_prompt'].encode()).hexdigest() for p, t in traits.items()})
        inputs += [Path(args.config).resolve(), *[KIT / 'trait_eval' / f'{p}.json' for p in traits]]
        # 含实际复用的实验助手文件，防止只记录入口而遗漏运行依赖。
        inputs += list((ROOT / 'antler').rglob('*.py')) + list(Path(__file__).resolve().parent.glob('*.py'))
        inputs += sorted(model_dir.glob('*.json'))
        inputs += [ROOT / 'docs' / 'RESEARCH_SPEC.md', ROOT / 'docs' / 'E0_DECODE_STRENGTH_2026-10-04.md']
        meta['data_hashes'] = {str(p): sha256(p) for p in inputs}
        judge = Judge(cache_path=str(out / 'judge_cache.jsonl'), response_audit_path=str(out / 'judge_responses.jsonl'))
        if judge.model != 'deepseek-flash':
            raise ValueError('本轮必须沿用deepseek-flash')
        meta['judge'] = _redacted({k: getattr(judge, k) for k in ('model', 'base_url', 'max_tokens', 'extra_body')})
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        if (model.config.num_hidden_layers, model.config.hidden_size) != (shape[0], shape[2]):
            raise ValueError('模型与向量形状不一致')
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if type(eos) is int else set(eos or [])
        if not eos_ids or any(type(t) is not int or t < 0 for t in eos_ids):
            raise ValueError('实际生成配置缺少有效EOS')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'generation_settings.json', model.generation_config.to_dict())
        meta['stage'] = 'preflight'
        save()
        preflight(model, tok, vectors, traits, eos_ids, out)
        check_sources()
        meta['stage'] = 'generation'
        save()
        for dim in COEFS:
            for pole in ('high', 'low'):
                target = BIG_FIVE[dim][pole == 'low']
                questions = traits[target]['questions'][:10] * 3
                prompts = [chat_prompt(tok, q) for q in questions]
                for seed in SEEDS:
                    for arm in ARMS:
                        trace = []
                        items = generate(model, tok, prompts, steer=arm_vector(vectors, dim, arm),
                                         steer_phase=arm_phase(arm), max_new_tokens=320, temperature=1., top_p=1.,
                                         batch_size=16, seed=seed, return_details=True, return_token_ids=True,
                                         phase_trace_out=trace)
                        identity = {'dim': dim, 'pole': pole, 'arm': arm, 'seed': seed}
                        # 先留原始批次；随后身份/阶段/文本核验失败也可追溯，不选择性补生成。
                        with (out / 'generation_batches.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                            stream.write(json.dumps({**identity, 'items': items}, ensure_ascii=False, allow_nan=False) + '\n')
                        phase_traces.append({**identity, 'batches': trace})
                        write_json(out / 'phase_traces.json', phase_traces)
                        validate_generation(items, 30, 320, eos_ids)
                        validate_phase_traces(trace, items, arm_phase(arm), 320)
                        if any(tok.batch_decode([item['token_ids']], skip_special_tokens=True)[0] != item['text'] for item in items):
                            raise ValueError('保存的token与回答文本不一致')
                        for index, item in enumerate(items):
                            rows.append({**identity, 'target': target, 'layer': LAYER, 'coef': COEFS[dim][arm],
                                         'axis': 'pole', 'split': 'calib', 'steer_phase': arm_phase(arm),
                                         'question_index': index % 10, 'sample_index': index // 10,
                                         'q': questions[index], 'a': item['text'],
                                         **{k: item[k] for k in ('token_ids', 'n_generated_tokens', 'finish_reason')}})
                        meta['generated'] = len(rows)
                        save()
                        print(f'[生成] {dim}/{pole}/seed{seed}/{arm} 完成，总计{len(rows)}/1800', flush=True)
        validate_rows(rows, traits)
        check_sources()
        meta['stage'] = 'scoring'
        save()
        complete = score_rows(judge, rows, traits, out)
        save()
        if not complete:
            raise ValueError('存在缺失评分，保留原结果，不自动重评')
        check_sources()
        write_json(out / 'assessment.json', analyze(rows, traits))
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        # 不把外部异常原文写入可分享产物，避免认证信息进入归档。
        meta.update(status='failed', error_type=type(exc).__name__, error='实验失败，未自动重试；请核对本地日志与当前阶段。')
        raise
    finally:
        meta['end_time'] = now()
        save()
        names = ['run_meta.json', 'samples.jsonl', 'generation_settings.json', 'preflight.json',
                 'generation_batches.jsonl', 'phase_traces.json', 'score_plan.json', 'judge_cache.jsonl',
                 'judge_responses.jsonl', 'assessment.json']
        manifest = {name: sha256(out / name) for name in names if (out / name).exists()}
        write_json(out / 'artifact_hashes.json', manifest)
        with zipfile.ZipFile(out / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in [*manifest, 'artifact_hashes.json']:
                archive.write(out / name, name)
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
