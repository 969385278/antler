"""C12：独立验证终态双胞胎实验，重建身份、请求和所有数值汇总。"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re

import numpy as np


DATA_SHA = '04a160c4c41c0ea9ebb7e97d61b7071136ab51a9b9c3b2c0ac7dcd17bea4e5cb'
FINGERPRINT_SHA = 'bc1de2777cf71a729570fefafcaea5b40cf5e03d2e8acea1bc39ab25d68eb1ae'
TRAITS = {'O': ('inventive', 'consistent'), 'E': ('outgoing', 'solitary'), 'A': ('compassionate', 'self-interested')}
TRAIT_HASHES = {
    'inventive': '2444e9dafb70d5ed453a8199a06bcf81b38b7a5c11c9f8e956e59f3c07578dca',
    'consistent': '28fb740e85b012818b73dc696b6f1a80b65ee844598f247cb728254c49984825',
    'outgoing': 'a9b85b03cc57ab4343fcb86a76ba6b3fda33677d0e476541f2c2bbb773c5865d',
    'solitary': '85c84ff203207b947b5244c85c0709dc609d30ca5d1ebb50ad535b5dd07fb1f2',
    'compassionate': 'd43d4ef27bc637c5cdf0f6258f9c4ebf8c0b7cc5bf13b1947e43a9e646729bb1',
    'self-interested': '375bdf9046f785d710c4338ff72e4fb7b4e32f14ef34e4a3c20101cad2441b69',
}
PRIOR_JUDGE = 'outputs/main/prompt_persona/d09a117a119541a2985f54a77c8e4391/run_meta.json'
PROTOCOL = {'n_scenarios': 6, 'n_stories': 24, 'n_answers': 2520, 'n_scores_max': 7560,
            'dims': ['O', 'E', 'A'], 'split': 'calib', 'question_indices': list(range(10)),
            'max_new_tokens': 768, 'temperature': 1., 'top_p': 1., 'batch_size': 8,
            'seeds': [0, 1, 2], 'score_shuffle_seed': 0, 'bootstrap_seed': 0,
            'bootstrap_samples': 10000, 'steer': None, 'retry': False}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def decode(text):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, 'JSON含重复键')
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def read(path):
    return decode(Path(path).read_text(encoding='utf-8-sig'))


def lines(path):
    return [decode(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def contained(root, relative):
    path = (root / relative).resolve()
    need(path.is_relative_to(root.resolve()), '归档路径越界')
    return path


def verify_manifest(run, manifest):
    files = {p.relative_to(run).as_posix() for p in run.rglob('*')
             if p.is_file() and p.name != 'artifact_hashes.json'}
    need(set(manifest) == files, '产物清单不完整或有额外文件')
    for relative, expected in manifest.items():
        need(digest(contained(run, relative)) == expected, '产物哈希不符')


def constants(path, wanted):
    """读取封存代码中的字面常量；不导入或执行归档代码。"""
    result = {}
    def value(node):
        if isinstance(node, ast.Name):
            need(node.id in result, '常量引用未定义')
            return result[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = value(node.left), value(node.right)
            need(isinstance(left, str) and isinstance(right, str), '只允许常量字符串连接')
            return left + right
        return ast.literal_eval(node)
    for statement in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                result[target.id] = value(statement.value)
    need(set(result) == set(wanted), '封存提示常量缺失')
    return result


def same(actual, expected, label):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected), f'{label}字段不一致')
        for key, value in expected.items():
            same(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected), f'{label}长度不一致')
        for index, value in enumerate(expected):
            same(actual[index], value, f'{label}[{index}]')
    elif isinstance(expected, float):
        need(type(actual) in (int, float) and math.isfinite(actual)
             and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), f'{label}数值不一致')
    else:
        need(type(actual) is type(expected) and actual == expected, f'{label}值不一致')


def parse_output(text):
    if not isinstance(text, str) or 'REFUSAL' in text.upper():
        return None
    match = re.search(r'-?\d+(\.\d+)?', text.strip())
    if match:
        value = float(match.group())
        if math.isfinite(value) and 0 <= value <= 100:
            return value
    return None


def source_snapshots(run, meta):
    original_root = PurePosixPath(meta['config_path'].replace('\\', '/')).parent.parent
    mapped = {}
    for original, expected in meta['data_hashes'].items():
        original_path = PurePosixPath(original.replace('\\', '/'))
        try:
            relative = original_path.relative_to(original_root).as_posix()
        except ValueError:
            relative = 'external/' + expected[:12] + '_' + original_path.name
        snapshot = contained(run / 'source_snapshot', relative)
        need(digest(snapshot) == expected, '封存来源哈希不符')
        need(relative not in mapped, '封存来源身份重复')
        mapped[relative] = snapshot
    return mapped


def expected_jobs(materials, memory, traits):
    stories = []
    need(len(materials['scenarios']) == 6, '须保留六情境')
    for scene in materials['scenarios']:
        for action in (-1, 1):
            for reward in (-1, 1):
                text = ' '.join((scene['prefix'], scene['actions'][str(action)], scene['outcomes'][str(reward)]))
                stories.append({'id': f'{scene["id"]}_a{action}_r{reward}', 'scenario_id': scene['id'],
                                'dim': scene['dim'], 'action_sign': action, 'outcome_sign': reward,
                                'predicted_sign': action * reward, 'text': text,
                                'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    jobs = []
    for dim in TRAITS:
        for pole in ('high', 'low'):
            qset = traits[dim, pole]['questions']
            need(len(qset) == 20, '评测切分题数改变')
            arms = [('none', None, None, memory['BASE_SYSTEM'])]
            arms += [('neutral', i, None, memory['MEMORY_TEMPLATE'].format(experience=m))
                     for i, m in enumerate(memory['NEUTRAL_MEMORIES'])]
            arms += [('context', None, e, memory['MEMORY_TEMPLATE'].format(experience=e['text']))
                     for e in stories if e['dim'] == dim]
            for arm, index, episode, system in arms:
                for seed in (0, 1, 2):
                    for q in range(10):
                        jobs.append({'id': len(jobs), 'dim': dim, 'pole': pole, 'split': 'calib', 'arm': arm,
                                     'memory_index': index, 'episode_id': episode['id'] if episode else None,
                                     'scenario_id': episode['scenario_id'] if episode else None,
                                     'action_sign': episode['action_sign'] if episode else None,
                                     'outcome_sign': episode['outcome_sign'] if episode else None,
                                     'seed': seed, 'sample_index': seed, 'question_index': q,
                                     'q': qset[q], 'system': system})
    need(len(jobs) == 2520 and len(stories) == 24, '固定设计预算改变')
    return jobs, stories


def verify_statistics(rows, materials, summary):
    """用四单元的线性权重独立复算，不调用实验的summarize。"""
    lookup = {(r['dim'], r['pole'], r['arm'], r['memory_index'], r['episode_id'], r['question_index'], r['seed']): r for r in rows}
    bootstrap = np.random.default_rng(0).integers(0, 10, size=(10000, 10))
    def stats(array):
        questions = np.sum(array, axis=1) / 3
        draws = np.sum(questions[bootstrap], axis=1) / 10
        return {'mean': float(np.sum(questions) / 10),
                'question_bootstrap_ci': np.percentile(draws, [2.5, 97.5]).tolist(),
                'per_seed_means': (np.sum(array, axis=0) / 10).tolist()}
    counts = dict.fromkeys(('established', 'partial', 'opposite', 'not_established', 'incomplete'), 0)
    need(len(summary['reports']) == 6, '汇总缺情境')
    for scene, report in zip(materials['scenarios'], summary['reports']):
        need(report['scenario_id'] == scene['id'] and report['dim'] == scene['dim'], '汇总情境身份错误')
        need(len(report['poles']) == 2, '汇总缺极点')
        decisions = []
        for pole_name, observed in zip(('high', 'low'), report['poles']):
            need((observed['scenario_id'], observed['dim'], observed['pole']) == (scene['id'], scene['dim'], pole_name), '极点身份错误')
            def group(arm, memory=None, episode=None):
                return [lookup[scene['dim'], pole_name, arm, memory, episode, q, s] for q in range(10) for s in (0, 1, 2)]
            keys = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            cells = [group('context', episode=f'{scene["id"]}_a{a}_r{r}') for a, r in keys]
            neutral, none = [group('neutral', i) for i in range(5)], group('none')
            selected = sum(cells, []) + sum(neutral, []) + none
            valid = lambda x: type(x) in (float, int) and math.isfinite(x) and 0 <= x <= 100
            missing = sum(any(not valid(row.get(m)) for m in ('trait', 'coherence', 'task')) for row in selected)
            need(observed['missing_or_invalid_rows'] == missing, '缺分计数错误')
            finish = {f'a{a}_r{r}': {s: sum(row['finish_reason'] == s for row in cell) for s in ('eos', 'length')}
                      for (a, r), cell in zip(keys, cells)}
            same(observed['finish_counts'], finish, '截断统计')
            decision = 'incomplete'
            for metric in ('trait', 'coherence', 'task'):
                missing_metric = sum(not valid(row.get(metric)) for row in selected)
                found = observed['metrics'][metric]
                if missing_metric:
                    same(found, {'status': 'incomplete', 'missing_or_invalid_rows': missing_metric}, '缺分指标')
                    continue
                values = lambda rs: np.asarray([r[metric] for r in rs], dtype=np.float64).reshape(10, 3)
                cube = np.stack([values(cell) for cell in cells])
                baseline = np.sum(np.stack([values(g) for g in neutral]), axis=0) / 5
                no_memory = values(none)
                contrast = lambda w: np.einsum('i,ijk->jk', np.asarray(w, dtype=np.float64), cube)
                interaction, valence = contrast([.5, -.5, -.5, .5]), contrast([-.5, .5, -.5, .5])
                result = {'status': 'complete', 'neutral': stats(baseline), 'none': stats(no_memory),
                          'cells': {f'a{a}_r{r}': {'absolute': stats(cube[i]), 'minus_neutral': stats(cube[i] - baseline),
                                    'minus_none': stats(cube[i] - no_memory)} for i, (a, r) in enumerate(keys)},
                          'feedback_positive_action': stats(contrast([0, 0, -1, 1])),
                          'feedback_negative_action': stats(contrast([-1, 1, 0, 0])),
                          'interaction_raw': stats(interaction), 'valence_raw': stats(valence)}
                if metric == 'trait':
                    orient = 1 if pole_name == 'high' else -1
                    result['interaction_high_axis'] = stats(orient * interaction)
                    result['valence_high_axis'] = stats(orient * valence)
                    lo, hi = result['interaction_high_axis']['question_bootstrap_ci']
                    expected = 'expected' if lo > 0 else 'opposite' if hi < 0 else 'not_established'
                    need(observed['trait_gate'] == expected, '人格闸门分类错误')
                    if not missing:
                        decision = expected
                same(found, result, f'{scene["id"]}/{pole_name}/{metric}')
            need(observed['gate'] == decision, '极点闸门分类错误')
            decisions.append(decision)
        category = ('incomplete' if 'incomplete' in decisions else 'opposite' if 'opposite' in decisions
                    else 'established' if all(d == 'expected' for d in decisions)
                    else 'partial' if 'expected' in decisions else 'not_established')
        need(report['gate'] == category, '情境闸门分类错误')
        counts[category] += 1
    same(summary['gate_counts'], counts, '闸门总计')
    for key, expected in {'n_scenarios': 6, 'n_stories': 24, 'n_answers': 2520,
                          'scope': 'six_scenario_action_outcome_context_gate_dev_calib',
                          'formal_e1_pass': False, 'behavior_validated': False, 'injection_performed': False}.items():
        same(summary[key], expected, key)
    return counts


def audit(run, tokenizer, model_dir):
    run, model_dir = Path(run).resolve(), Path(model_dir).resolve()
    meta = read(run / 'run_meta.json')
    need(meta.get('status') == 'completed' and meta.get('stage') == 'completed', '运行未到完成终态，不能验收')
    need(meta.get('generated') == 2520 and meta.get('scored') == 7560, '终态预算计数错误')
    same(meta.get('protocol'), PROTOCOL, '固定运行参数')
    need(meta.get('task') == 'C12' and meta.get('seed') == 0 and meta.get('end_time'), '终态任务信息缺失')
    need(meta.get('all_parameters_frozen') is True and all(meta.get(k) is False for k in (
        'formal_e1_pass', 'behavior_validated', 'injection_performed')), '实验范围或冻结标志错误')
    manifest = read(run / 'artifact_hashes.json')
    manifest_sha = digest(run / 'artifact_hashes.json')
    verify_manifest(run, manifest)
    sources = source_snapshots(run, meta)
    material_path = sources['data/diagnostics/twin_outcome_dev_v1.json']
    need(digest(material_path) == DATA_SHA, '实验材料不是固定版本')
    materials = read(material_path)
    memory = constants(sources['antler/memory/write.py'], ('BASE_SYSTEM', 'MEMORY_TEMPLATE', 'NEUTRAL_MEMORIES'))
    prompts = constants(sources['antler/eval/judge.py'], ('COHERENCE_PROMPT', 'TASK_PROMPT'))
    traits = {}
    for dim, names in TRAITS.items():
        for pole, name in zip(('high', 'low'), names):
            p = sources[f'data/kit/trait_eval/{name}.json']
            need(digest(p) == TRAIT_HASHES[name], '评测材料不是固定版本')
            traits[dim, pole] = read(p)
    fingerprint_paths = [p for p in sources.values() if digest(p) == FINGERPRINT_SHA]
    need(len(fingerprint_paths) == 1, '模型指纹不是封存版本')
    fingerprint = read(fingerprint_paths[0])
    need(len(fingerprint['files']) == 11
         and len({r['file'] for r in fingerprint['files']}) == 11
         and 'generation_config.json' in {r['file'] for r in fingerprint['files']}, '模型指纹须有11个唯一文件')
    original_model = fingerprint['model_directory'].replace('\\', '/').rstrip('/')
    needed_hashes = {}
    for row in fingerprint['files']:
        p = contained(model_dir, row['file'])
        need(p.stat().st_size == row['bytes'] and digest(p) == row['sha256'], '模型文件指纹不符')
        needed_hashes[original_model + '/' + row['file']] = row['sha256']
    same(meta['model_hashes'], needed_hashes, '运行模型身份')
    prior = read(sources[PRIOR_JUDGE])['judge']
    same(meta['judge'], prior, '固定主评委')
    jobs, stories = expected_jobs(materials, memory, traits)
    same(read(run / 'jobs.json'), jobs, '工作清单')
    same(read(run / 'stories.json'), stories, '故事原文')
    rows = lines(run / 'answers.jsonl')
    need(len(rows) == 2520, '原始回答缺失')
    eos = meta['stop_config']['model_eos_token_id']
    eos = {eos} if type(eos) is int else set(eos or [])
    need(eos, '缺少EOS停止配置')
    actual_generation = read(model_dir / 'generation_config.json')
    same(meta['stop_config']['model_eos_token_id'], actual_generation['eos_token_id'], '模型EOS配置')
    need(meta['stop_config']['tokenizer_eos_token_id'] == tokenizer.eos_token_id
         and meta['stop_config']['pad_token_id'] == tokenizer.pad_token_id, '分词器停止配置改变')
    for row, job in zip(rows, jobs):
        same({k: row.get(k) for k in job}, job, '原始回答身份')
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': job['system']},
                                                {'role': 'user', 'content': job['q']}],
                                               tokenize=False, add_generation_prompt=True)
        need(row['prompt'] == prompt, '实际chat提示不符')
        ids = row['token_ids']
        need(isinstance(ids, list) and all(type(t) is int and t >= 0 for t in ids)
             and 0 < len(ids) <= 768 and type(row['n_generated_tokens']) is int
             and row['n_generated_tokens'] == len(ids), '生成token计数错误')
        if row['finish_reason'] == 'eos':
            need(ids[-1] in eos and not any(t in eos for t in ids[:-1]), 'EOS截断错误')
        else:
            need(row['finish_reason'] == 'length' and len(ids) == 768 and not any(t in eos for t in ids), '长度停止错误')
        need(tokenizer.batch_decode([ids], skip_special_tokens=True)[0] == row['a'], 'token解码与回答不符')
    scored = read(run / 'scored_answers.json')
    need(len(scored) == 2520, '评分回答不完整')
    for raw, enriched in zip(rows, scored):
        same({k: enriched.get(k) for k in raw}, raw, '评分原文关联')
    responses_list, cache_list = lines(run / 'judge_responses.jsonl'), lines(run / 'judge_cache.jsonl')
    responses = {r['key']: r for r in responses_list}
    cache = {r['key']: r['output'] for r in cache_list}
    need(len(responses) == len(responses_list) and 0 < len(responses) <= 7560, '评委请求重复或超预算')
    need(len(cache) == len(cache_list), '评委缓存重复')
    events = lines(run / 'score_records.jsonl')
    need(len(events) == 7560, '逐项评分记录缺失')
    used, request_order, cursor, invalid = set(), [], 0, 0
    for dim in TRAITS:
        for pole in ('high', 'low'):
            order = [r['id'] for r in scored if (r['dim'], r['pole']) == (dim, pole)]
            random.Random(0).shuffle(order)
            same(read(run / f'{dim}_{pole}_score_order.json'), order, '盲评顺序')
            for metric in ('trait', 'coherence', 'task'):
                template = traits[dim, pole]['eval_prompt'] if metric == 'trait' else prompts[
                    'COHERENCE_PROMPT' if metric == 'coherence' else 'TASK_PROMPT']
                for rank, identity in enumerate(order):
                    row = scored[identity]
                    prompt = template.format(question=row['q'], answer=row['a'])
                    key = hashlib.sha256((prior['model'] + '\n' + prompt).encode()).hexdigest()
                    need(key in responses, '评分缺少原始响应证据')
                    response = responses[key]
                    payload = {'model': prior['model'], 'temperature': 0, 'max_tokens': prior['max_tokens'],
                               'messages': [{'role': 'user', 'content': prompt}], **prior['extra_body']}
                    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    need(response['requested_model'] == prior['model'] and response['returned_model'] == prior['model']
                         and response['request_sha256'] == request_hash, '评委模型或请求正文不符')
                    value = parse_output(response['output'])
                    same(row.get(metric), value, '原始响应评分')
                    same(events[cursor], {'id': identity, 'metric': metric, 'score': value, 'dim': dim,
                                         'pole': pole, 'judge_order_index': rank}, '逐项评分身份')
                    need(row['judge_order_index'] == rank, '评分顺序关联错误')
                    if key not in used:
                        request_order.append(key)
                        used.add(key)
                    invalid += value is None
                    cursor += 1
    need(used == set(responses), '存在未关联的评委请求')
    same([r['key'] for r in responses_list], request_order, '实际评委请求顺序')
    expected_cache = {key: r['output'] for key, r in responses.items()
                      if parse_output(r['output']) is not None or 'REFUSAL' in r['output'].upper()}
    same(cache, expected_cache, '响应缓存关联')
    summary = read(run / 'summary.json')
    gate_counts = verify_statistics(scored, materials, summary)
    verify_manifest(run, manifest)
    need(digest(run / 'artifact_hashes.json') == manifest_sha, '审计期间产物清单发生变化')
    return {'status': 'passed', 'run_id': run.name, 'source_files_verified': len(sources),
            'artifact_files_verified': len(manifest), 'model_files_rechecked': len(needed_hashes),
            'answers_verified': 2520, 'scores_verified': 7560, 'unique_judge_requests': len(responses),
            'invalid_or_refusal_score_assignments': invalid,
            'token_decode_and_prompt_verified': True, 'independent_statistics_verified': True,
            'gate_counts': gate_counts, 'finish_counts': {s: sum(r['finish_reason'] == s for r in rows)
                                                        for s in ('eos', 'length')},
            'returned_model_names': sorted({r['returned_model'] for r in responses.values()}),
            'system_fingerprints': sorted({str(r.get('system_fingerprint')) for r in responses.values()}),
            'run_meta_sha256': digest(run / 'run_meta.json'), 'summary_sha256': digest(run / 'summary.json'),
            'formal_e1_pass': False, 'behavior_validated': False}
