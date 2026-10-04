"""C12：独立审计合成终态归档，不联网、不下载或加载模型。"""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import pytest

from antler.eval import twin_context_gate as gate
from antler.eval import twin_gate_audit as auditor


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8', newline='\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                    encoding='utf-8', newline='\n')


def seal(run):
    write(run / 'artifact_hashes.json', {
        p.relative_to(run).as_posix(): auditor.digest(p) for p in run.rglob('*')
        if p.is_file() and p.name != 'artifact_hashes.json'})


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return '\n'.join(m['role'] + ':' + m['content'] for m in messages) + '\nassistant:'

    def batch_decode(self, batches, *, skip_special_tokens):
        assert skip_special_tokens
        return ['回答' + ','.join(str(i) for i in ids if i != 2) for ids in batches]


@pytest.fixture(scope='module')
def archive(tmp_path_factory):
    """生成器的汇总作被审计对象；审计器独立用线性权重复算。"""
    root = tmp_path_factory.mktemp('twin_archive')
    run, model = root / 'run', root / 'model'
    run.mkdir()
    model.mkdir()
    original = '/archive/antler'
    meta = {'task': 'C12', 'seed': 0, 'status': 'completed', 'stage': 'completed',
            'generated': 2520, 'scored': 7560, 'end_time': '固定终态时间',
            'config_path': original + '/configs/main.yaml', 'data_hashes': {},
            'all_parameters_frozen': True, 'formal_e1_pass': False, 'behavior_validated': False,
            'injection_performed': False, 'protocol': deepcopy(auditor.PROTOCOL),
            'stop_config': {'model_eos_token_id': 2, 'tokenizer_eos_token_id': 2, 'pad_token_id': 2}}
    sources = ['data/diagnostics/twin_outcome_dev_v1.json', 'antler/memory/write.py', 'antler/eval/judge.py']
    sources += ['data/kit/trait_eval/' + name + '.json' for names in auditor.TRAITS.values() for name in names]
    for relative in sources:
        target = run / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    prior = {'model': '固定测试评委', 'base_url': 'https://example.invalid',
             'max_tokens': 16, 'extra_body': {'thinking': {'type': 'disabled'}}}
    write(run / 'source_snapshot' / auditor.PRIOR_JUDGE, {'judge': prior})
    meta['judge'] = prior
    write(model / 'generation_config.json', {'eos_token_id': 2})
    for i in range(10):
        (model / f'file{i}').write_bytes(f'模型测试文件{i}'.encode())
    fingerprint = {'model_directory': original + '/models/fixed', 'files': [
        {'file': p.name, 'bytes': p.stat().st_size, 'sha256': auditor.digest(p)}
        for p in sorted(model.iterdir())]}
    fp = run / 'source_snapshot/outputs/fingerprint.json'
    write(fp, fingerprint)
    meta['model_hashes'] = {fingerprint['model_directory'] + '/' + r['file']: r['sha256']
                            for r in fingerprint['files']}
    for p in (run / 'source_snapshot').rglob('*'):
        if p.is_file():
            relative = p.relative_to(run / 'source_snapshot').as_posix()
            meta['data_hashes'][original + '/' + relative] = auditor.digest(p)
    data = auditor.read(run / 'source_snapshot/data/diagnostics/twin_outcome_dev_v1.json')
    rows = gate.jobs_for(data)
    jobs = deepcopy(rows)
    tokenizer = Tokenizer()
    for row in rows:
        row.update(token_ids=[row['id'] + 3, 2], n_generated_tokens=2, finish_reason='eos')
        row['prompt'] = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': row['system']}, {'role': 'user', 'content': row['q']}],
            tokenize=False, add_generation_prompt=True)
        row['a'] = tokenizer.batch_decode([row['token_ids']], skip_special_tokens=True)[0]
        direction = 1 if row['pole'] == 'high' else -1
        a, r = row['action_sign'], row['outcome_sign']
        row.update(trait=50., coherence=92., task=94.)
        if row['arm'] == 'context':
            interaction = 4. + row['question_index'] / 10 + row['seed'] / 5
            if row['scenario_id'] == 'twin01_study':
                interaction *= -1
            if row['scenario_id'] == 'twin02_walking' or (
                    row['scenario_id'] == 'twin03_language' and row['pole'] == 'low'):
                interaction = 0.
            row['trait'] += direction * (interaction * a * r + 6 * r)
            row['coherence'] += r
            row['task'] -= a
        elif row['arm'] == 'neutral':
            row['trait'] += row['memory_index']
    missing = next(row for row in rows if row['scenario_id'] == 'twin04_ideas')
    missing['trait'] = None
    # 一条合法长度停止，其他为自然结束；同时核对两种停止证据。
    rows[0].update(token_ids=[3] * 768, n_generated_tokens=768, finish_reason='length')
    rows[0]['a'] = tokenizer.batch_decode([rows[0]['token_ids']], skip_special_tokens=True)[0]
    raw = [{k: v for k, v in row.items() if k not in ('trait', 'coherence', 'task')} for row in rows]
    responses, cache, events = {}, {}, []
    prompts = auditor.constants(ROOT / 'antler/eval/judge.py', ('COHERENCE_PROMPT', 'TASK_PROMPT'))
    for dim, names in auditor.TRAITS.items():
        for pole, name in zip(('high', 'low'), names):
            order = [row['id'] for row in rows if (row['dim'], row['pole']) == (dim, pole)]
            random.Random(0).shuffle(order)
            write(run / f'{dim}_{pole}_score_order.json', order)
            trait = auditor.read(ROOT / f'data/kit/trait_eval/{name}.json')
            for metric in ('trait', 'coherence', 'task'):
                template = trait['eval_prompt'] if metric == 'trait' else prompts[
                    'COHERENCE_PROMPT' if metric == 'coherence' else 'TASK_PROMPT']
                for rank, identity in enumerate(order):
                    row = rows[identity]
                    row['judge_order_index'] = rank
                    prompt = template.format(question=row['q'], answer=row['a'])
                    key = hashlib.sha256((prior['model'] + '\n' + prompt).encode()).hexdigest()
                    payload = {'model': prior['model'], 'temperature': 0, 'max_tokens': prior['max_tokens'],
                               'messages': [{'role': 'user', 'content': prompt}], **prior['extra_body']}
                    output = str(row[metric]) if row[metric] is not None else 'unparseable'
                    if key in responses:
                        assert responses[key]['output'] == output
                    responses[key] = {'key': key, 'requested_model': prior['model'], 'returned_model': prior['model'],
                                      'request_sha256': hashlib.sha256(json.dumps(
                                          payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                                      'output': output, 'system_fingerprint': 'test'}
                    if row[metric] is not None:
                        cache[key] = output
                    events.append({'id': identity, 'metric': metric, 'score': row[metric], 'dim': dim,
                                   'pole': pole, 'judge_order_index': rank})
    write(run / 'run_meta.json', meta)
    write(run / 'jobs.json', jobs)
    write(run / 'stories.json', gate.expand(data))
    jsonl(run / 'answers.jsonl', raw)
    write(run / 'scored_answers.json', rows)
    jsonl(run / 'judge_responses.jsonl', list(responses.values()))
    jsonl(run / 'judge_cache.jsonl', [{'key': key, 'output': value} for key, value in cache.items()])
    jsonl(run / 'score_records.jsonl', events)
    write(run / 'summary.json', gate.summarize(rows, data))
    seal(run)
    return root, auditor.digest(fp)


@pytest.fixture
def case(archive, tmp_path, monkeypatch):
    root, fingerprint_sha = archive
    shutil.copytree(root / 'run', tmp_path / 'run')
    shutil.copytree(root / 'model', tmp_path / 'model')
    monkeypatch.setattr(auditor, 'FINGERPRINT_SHA', fingerprint_sha)
    return tmp_path / 'run', tmp_path / 'model'


def test_independent_complete_archive_and_missing_scores(case):
    run, model = case
    result = auditor.audit(run, Tokenizer(), model)
    assert result['status'] == 'passed'
    assert result['answers_verified'] == 2520 and result['scores_verified'] == 7560
    assert result['invalid_or_refusal_score_assignments'] == 1
    assert result['gate_counts'] == {'established': 2, 'partial': 1, 'opposite': 1,
                                      'not_established': 1, 'incomplete': 1}
    assert result['finish_counts'] == {'eos': 2519, 'length': 1}
    assert result['model_files_rechecked'] == 11
    assert not result['formal_e1_pass'] and not result['behavior_validated']


@pytest.mark.parametrize('field,value', [('status', 'running'), ('stage', 'generate'), ('generated', 2510),
                                        ('scored', 7559), ('injection_performed', True),
                                        ('formal_e1_pass', True), ('all_parameters_frozen', False)])
def test_reject_nonterminal_and_scope_changes(case, field, value):
    run, model = case
    meta = auditor.read(run / 'run_meta.json')
    meta[field] = value
    write(run / 'run_meta.json', meta)
    seal(run)
    with pytest.raises(ValueError):
        auditor.audit(run, Tokenizer(), model)


@pytest.mark.parametrize('field,value', [('temperature', 0.), ('split', 'test'), ('max_new_tokens', 320),
                                        ('seeds', [0, 1]), ('batch_size', 4), ('retry', True)])
def test_reject_protocol_changes(case, field, value):
    run, model = case
    meta = auditor.read(run / 'run_meta.json')
    meta['protocol'][field] = value
    write(run / 'run_meta.json', meta)
    seal(run)
    with pytest.raises(ValueError, match='固定运行参数'):
        auditor.audit(run, Tokenizer(), model)


@pytest.mark.parametrize('fault', ['ci', 'gate', 'low_sign', 'seed', 'missing', 'count'])
def test_reject_statistics_tampering(case, fault):
    run, model = case
    summary = auditor.read(run / 'summary.json')
    pole = summary['reports'][0]['poles'][1]
    effect = pole['metrics']['trait']['interaction_high_axis']
    if fault == 'ci':
        effect['question_bootstrap_ci'][0] += .1
    elif fault == 'gate':
        summary['reports'][0]['gate'] = 'established'
    elif fault == 'low_sign':
        effect['mean'] *= -1
    elif fault == 'seed':
        effect['per_seed_means'][1] += .1
    elif fault == 'missing':
        summary['reports'][3]['poles'][0]['metrics']['trait'] = {'status': 'complete'}
    else:
        summary['gate_counts']['opposite'] = 0
    write(run / 'summary.json', summary)
    seal(run)
    with pytest.raises(ValueError):
        auditor.audit(run, Tokenizer(), model)


@pytest.mark.parametrize('fault', ['identity', 'tokens', 'eos', 'length', 'prompt', 'decode'])
def test_reject_generation_evidence_tampering(case, fault):
    run, model = case
    rows = auditor.lines(run / 'answers.jsonl')
    row = rows[1]
    if fault == 'identity':
        row['episode_id'] = '错配经历'
    elif fault == 'tokens':
        row['n_generated_tokens'] = 3
    elif fault == 'eos':
        row['token_ids'][-1] = 9999
    elif fault == 'length':
        row['finish_reason'] = 'length'
    elif fault == 'prompt':
        row['prompt'] = '另一个提示'
    else:
        row['a'] = '另一个回答'
    jsonl(run / 'answers.jsonl', rows)
    seal(run)
    with pytest.raises(ValueError):
        auditor.audit(run, Tokenizer(), model)


@pytest.mark.parametrize('fault', ['request', 'model', 'duplicate', 'request_order', 'cache', 'event', 'order', 'score'])
def test_reject_judge_evidence_tampering(case, fault):
    run, model = case
    if fault in ('request', 'model', 'duplicate', 'request_order'):
        path = run / 'judge_responses.jsonl'
        rows = auditor.lines(path)
        if fault == 'request':
            rows[0]['request_sha256'] = '错误请求'
        elif fault == 'model':
            rows[0]['returned_model'] = '另一个模型'
        elif fault == 'duplicate':
            rows.append(rows[0])
        else:
            rows.reverse()
        jsonl(path, rows)
    elif fault == 'cache':
        path = run / 'judge_cache.jsonl'
        rows = auditor.lines(path)
        rows[0]['output'] = '99'
        jsonl(path, rows)
    elif fault == 'event':
        path = run / 'score_records.jsonl'
        rows = auditor.lines(path)
        rows[0]['id'] = -1
        jsonl(path, rows)
    elif fault == 'order':
        path = run / 'O_high_score_order.json'
        write(path, list(reversed(auditor.read(path))))
    else:
        path = run / 'scored_answers.json'
        rows = auditor.read(path)
        rows[0]['trait'] += 1
        write(path, rows)
    seal(run)
    with pytest.raises(ValueError):
        auditor.audit(run, Tokenizer(), model)


def test_reject_changed_model_file(case):
    run, model = case
    (model / 'file0').write_bytes(b'changed')
    with pytest.raises(ValueError, match='模型文件指纹'):
        auditor.audit(run, Tokenizer(), model)


def test_reject_source_hash_and_unlisted_artifacts(case):
    run, model = case
    path = run / 'source_snapshot/antler/memory/write.py'
    path.write_text('BASE_SYSTEM = "修改"', encoding='utf-8')
    with pytest.raises(ValueError, match='产物哈希'):
        auditor.audit(run, Tokenizer(), model)
    seal(run)
    with pytest.raises(ValueError, match='封存来源哈希'):
        auditor.audit(run, Tokenizer(), model)
    (run / 'extra.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='产物清单'):
        auditor.audit(run, Tokenizer(), model)


def test_reject_archive_drift_during_audit(case, monkeypatch):
    run, model = case
    original = auditor.verify_statistics
    def drift(*args):
        result = original(*args)
        (run / 'summary.json').write_text('{}', encoding='utf-8')
        return result
    monkeypatch.setattr(auditor, 'verify_statistics', drift)
    with pytest.raises(ValueError, match='产物哈希'):
        auditor.audit(run, Tokenizer(), model)


@pytest.mark.parametrize('json_lines', [False, True])
def test_duplicate_json_keys_rejected(tmp_path, json_lines):
    p = tmp_path / 'data.json'
    p.write_text('{"score":1,"score":2}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='重复键'):
        (auditor.lines if json_lines else auditor.read)(p)


def test_snapshot_constants_do_not_execute_code(tmp_path):
    p = tmp_path / 'constants.py'
    p.write_text('raise RuntimeError("不能执行")\nA="固定"\nB=A+"文本"\n', encoding='utf-8')
    assert auditor.constants(p, ('A', 'B')) == {'A': '固定', 'B': '固定文本'}
    p.write_text('A=open("secret").read()\n', encoding='utf-8')
    with pytest.raises(ValueError):
        auditor.constants(p, ('A',))


def test_archive_path_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match='越界'):
        auditor.contained(tmp_path, '../secret')


def test_cli_uses_local_tokenizer_and_keeps_failure_meta(case, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'experiments'))
    runner = importlib.import_module('e1_twin_gate_audit')
    run, model = case
    config = tmp_path / 'config.yaml'
    config.write_text('model: fixed\n', encoding='utf-8')
    monkeypatch.setattr(runner, 'setup', lambda *a, **k: ({'out_path': tmp_path / 'outputs'}, None, None, None))
    calls = []
    def local(path, **kwargs):
        calls.append((path, kwargs))
        raise ValueError('模拟本地分词器缺失')
    monkeypatch.setattr(runner.AutoTokenizer, 'from_pretrained', local)
    args = SimpleNamespace(config=str(config), run=str(run), model_dir=str(model))
    with pytest.raises(ValueError, match='分词器缺失'):
        runner.run(args)
    assert calls == [(str(model), {'local_files_only': True})]
    out = next((tmp_path / 'outputs/e1_twin_gate_audit').iterdir())
    meta = auditor.read(out / 'run_meta.json')
    assert meta['status'] == 'failed' and meta['model_forward_calls'] == meta['judge_calls'] == 0
    assert (out / 'artifact_hashes.json').exists()
