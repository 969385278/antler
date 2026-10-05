"""C12：有界行为预算、同轮盲评、能力失败与匹配误差的CPU测试。"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import e1_twin_raw_behavior as entry
from antler.core.model import chat_prompt, get_layers
from antler.eval.twin_gate_audit import read, digest
from antler.eval.twin_context_gate import jobs_for
from antler.eval.twin_raw_behavior import build_jobs, reused_answers, expected_ids, summarize, score_order
from test_core import model
from test_twin_state_extract import TinyTokenizer


def materials():
    return read(Path(__file__).resolve().parents[1] / 'data/diagnostics/twin_outcome_dev_v1.json')


def capability():
    return {key: {'within_2pp': True} for key in expected_ids()}


def synthetic():
    caps = capability()
    jobs = build_jobs(materials(), caps)
    rows = []
    for job in jobs:
        offset = job['memory_index'] - 2 if job['memory_index'] is not None else 0
        effect = job['action_sign'] * job['outcome_sign'] * 10 if job['action_sign'] is not None else 0
        if job['format'] == 'reconstructed_state':
            effect = 0
        sign = 1 if job['pole'] == 'high' else -1
        rows.append({**job, 'trait': 50 + sign * effect + offset, 'coherence': 90 + offset,
                     'task': 90 + offset, 'finish_reason': 'eos'})
    return rows, jobs, caps


def test_exact_budget_no_test_questions_and_neutral_coverage():
    jobs = build_jobs(materials(), capability())
    assert len(jobs) == 490 and sum(j['arm'] == 'injected' for j in jobs) == 400
    assert sum(j['arm'] == 'context' for j in jobs) == 40
    assert sum(j['arm'] == 'neutral' for j in jobs) == 50
    assert {j['question_index'] for j in jobs} == set(range(5))
    assert {j['seed'] for j in jobs} == {0} and {j['split'] for j in jobs} == {'calib'}
    for identity in expected_ids():
        selected = [j for j in jobs if j['vector_id'] == identity]
        assert len(selected) == 50 and {j['memory_index'] for j in selected} == set(range(5))


def test_failed_capability_omits_generation_but_preserves_comparison_failure():
    rows, jobs, caps = synthetic()
    caps['a1_r1_full_state']['within_2pp'] = False
    jobs = build_jobs(materials(), caps)
    kept = [r for r in rows if r['vector_id'] != 'a1_r1_full_state']
    rows = [{**r, **j} for r, j in zip(kept, jobs)]
    result = summarize(rows, jobs, caps)['reports'][0]['metrics']['trait']['formats']['full_state']
    assert len(jobs) == 440
    assert result['cells']['a1_r1']['status'] == 'capability_failure'
    assert result['cells']['a1_r-1']['match_comparison']['status'] == 'insufficient'
    assert result['contrast']['status'] == 'insufficient'


def test_reused_answers_do_not_carry_old_scores_or_order():
    jobs = build_jobs(materials(), capability())
    upstream = [{**j, 'a': '原文', 'prompt': '提示', 'token_ids': [2], 'n_generated_tokens': 1,
                 'finish_reason': 'eos', 'trait': 99, 'task': 99, 'coherence': 99, 'judge_order_index': 88}
                for j in jobs_for(materials())]
    reused = reused_answers(jobs, upstream)
    assert len(reused) == 90
    assert all(not any(k in r for k in ('trait', 'task', 'coherence', 'judge_order_index')) for r in reused)
    upstream[reused[0]['upstream_id']]['q'] = '换题'
    with pytest.raises(ValueError):
        reused_answers(jobs, upstream)


def test_interaction_mismatch_and_low_pole_orientation():
    rows, jobs, caps = synthetic()
    result = summarize(rows, jobs, caps)
    for pole in result['reports']:
        metric = pole['metrics']['trait']
        assert metric['neutral']['mean'] == 50
        assert metric['context_contrast']['interaction_high_axis']['mean'] == 20
        assert metric['context_contrast']['d_plus_high_axis']['mean'] == 20
        assert metric['context_contrast']['d_minus_high_axis']['mean'] == -20
        full = metric['formats']['full_state']
        assert full['contrast']['interaction_high_axis']['mean'] == 20
        for cell in full['cells'].values():
            comp = cell['match_comparison']
            assert comp['matched_absolute_error']['mean'] == 0
            assert comp['mismatched_absolute_error']['mean'] == 20
            assert comp['matched_closer_on_mean'] is True
        rebuilt = metric['formats']['reconstructed_state']
        assert rebuilt['contrast']['interaction_raw']['mean'] == 0
        assert all(c['match_comparison']['matched_closer_on_mean'] is False for c in rebuilt['cells'].values())
        assert pole['metrics']['coherence']['formats']['full_state']['cells']['a1_r1']['quality']['within_10_loss']
    assert result['formal_e1_pass'] is False and result['behavior_validated'] is False


def test_common_valence_does_not_count_as_interaction():
    rows, jobs, caps = synthetic()
    for row in rows:
        if row['arm'] != 'neutral':
            row['trait'] = 50 + row['outcome_sign'] * 10
    result = summarize(rows, jobs, caps)
    contrast = result['reports'][0]['metrics']['trait']['formats']['full_state']['contrast']
    assert contrast['interaction_raw']['mean'] == 0
    assert contrast['valence_raw']['mean'] == 20


def test_absolute_error_after_neutral_average_and_before_question_average():
    rows, jobs, caps = synthetic()
    for row in rows:
        if row['arm'] == 'injected' and row['vector_id'] == 'a1_r1_full_state' and row['pole'] == 'high':
            row['trait'] += [4, -4, 4, -4, 0][row['question_index']]
    comp = summarize(rows, jobs, caps)['reports'][0]['metrics']['trait']['formats']['full_state']['cells']['a1_r1']['match_comparison']
    assert comp['matched_absolute_error']['mean'] == pytest.approx(3.2)
    assert comp['matched_absolute_error']['per_question'] == [4, 4, 4, 4, 0]


@pytest.mark.parametrize('bad', [None, float('nan'), True, 101])
def test_missing_scores_never_drop_a_question_or_neutral(bad):
    rows, jobs, caps = synthetic()
    next(r for r in rows if r['vector_id'] == 'a1_r1_full_state' and r['pole'] == 'high')['trait'] = bad
    result = summarize(rows, jobs, caps)
    assert result['missing_score_assignments'] == 1
    cell = result['reports'][0]['metrics']['trait']['formats']['full_state']['cells']['a1_r1']
    assert cell['absolute'] is None and cell['status'] == 'incomplete_scores'
    assert cell['match_comparison']['status'] == 'insufficient'


def test_identity_changes_rejected():
    rows, jobs, caps = synthetic()
    rows.reverse()
    with pytest.raises(ValueError):
        summarize(rows, jobs, caps)


def test_fresh_blind_shuffle_contains_all_arms_and_same_order_per_metric(tmp_path):
    rows, jobs, _ = synthetic()
    class FakeJudge:
        def __init__(self):
            self.prompts = []
        def _key(self, p):
            import hashlib
            return hashlib.sha256(p.encode()).hexdigest()
        def _call(self, p):
            self.prompts.append(p)
            return 'REFUSAL' if len(self.prompts) == 1 else '90'
    judge = FakeJudge()
    for row in rows:
        row['a'] = '回答'
        for metric in ('trait', 'coherence', 'task'):
            row.pop(metric)
    assert entry.score_rows(judge, rows, tmp_path) == 1470
    records = [json.loads(s) for s in (tmp_path / 'score_records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(records) == 1470 and records[0]['score'] is None
    for pole in ('high', 'low'):
        wanted = score_order(rows, pole)
        assert read(tmp_path / f'E_{pole}_score_order.json') == wanted
        assert wanted != [r['id'] for r in rows if r['pole'] == pole]
        for metric in ('trait', 'coherence', 'task'):
            assert [r['id'] for r in records if r['pole'] == pole and r['metric'] == metric] == wanted
    assert all('vector_id' not in p and 'full_state' not in p for p in judge.prompts)


def test_api_failure_records_attempt_without_retry(tmp_path):
    rows, _, _ = synthetic()
    for r in rows:
        r['a'] = '回答'
    class Broken:
        calls = 0
        def _key(self, p):
            return '键'
        def _call(self, p):
            self.calls += 1
            raise RuntimeError('接口异常')
    judge = Broken()
    with pytest.raises(RuntimeError):
        entry.score_rows(judge, rows, tmp_path)
    assert judge.calls == 1
    assert len((tmp_path / 'judge_requests.jsonl').read_text(encoding='utf-8').splitlines()) == 1


@pytest.mark.parametrize('tamper', [False, True])
def test_response_identity_and_request_hash_must_match(tmp_path, tamper):
    import hashlib
    from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT
    from antler.traits.data import BIG_FIVE, load_trait
    row = {'id': 0, 'pole': 'high', 'q': '题目', 'a': '回答', 'trait': 90., 'coherence': 90., 'task': 90.}
    judge = {'model': '固定评委', 'max_tokens': 16, 'extra_body': {}}
    requests, responses, records = [], [], []
    for metric, template in zip(('trait', 'coherence', 'task'),
                                (load_trait(BIG_FIVE['E'][0], 'eval')['eval_prompt'], COHERENCE_PROMPT, TASK_PROMPT)):
        prompt = template.format(question=row['q'], answer=row['a'])
        key = hashlib.sha256(f'{judge["model"]}\n{prompt}'.encode()).hexdigest()
        payload = {'model': judge['model'], 'temperature': 0, 'max_tokens': 16,
                   'messages': [{'role': 'user', 'content': prompt}]}
        requests.append({'id': 0, 'metric': metric, 'pole': 'high', 'key': key, 'prompt': prompt})
        records.append({'id': 0, 'metric': metric, 'pole': 'high', 'key': key, 'score': 90.})
        responses.append({'key': key, 'requested_model': judge['model'], 'returned_model': judge['model'],
                          'request_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                          'finish_reason': 'stop', 'output': '90'})
    if tamper:
        responses[0]['returned_model'] = '另一个模型'
    for name, values in (('judge_requests', requests), ('score_records', records), ('judge_responses', responses)):
        (tmp_path / (name + '.jsonl')).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf-8')
    if tamper:
        with pytest.raises(ValueError):
            entry.verify_scores(tmp_path, [row], judge)
    else:
        assert entry.verify_scores(tmp_path, [row], judge) == 3


def test_preflight_does_not_load_model_or_judge(monkeypatch):
    monkeypatch.setattr(entry, 'setup', lambda *a, **kw: ({}, None, None, None))
    monkeypatch.setattr(entry, 'validate', lambda _: ([], {}, [{}] * 90, []))
    monkeypatch.setattr(entry, 'load_model', lambda *a, **kw: pytest.fail('预检禁止模型'))
    monkeypatch.setattr(entry, 'SequentialJudge', lambda *a, **kw: pytest.fail('预检禁止评委'))
    assert entry.run(SimpleNamespace(execute=False))['model_loaded'] is False


class Tokenizer(TinyTokenizer):
    pad_token_id, eos_token_id = 0, 2
    def batch_decode(self, sequences, skip_special_tokens=True):
        return ['回答' for _ in sequences]


def test_cpu_real_steer_generation_amplitude_and_cleanup(monkeypatch, model):
    from antler.core.hooks import Recorder
    jobs = [j for j in build_jobs(materials(), capability()) if j['arm'] == 'injected'][:5]
    vector = torch.arange(32).float() / 100
    tok = Tokenizer()
    monkeypatch.setattr(model.generation_config, 'eos_token_id', 2)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = tok([chat_prompt(tok, j['q'], system=j['system']) for j in jobs], return_tensors='pt', padding=True)
    with Recorder(model, [2]) as before:
        model(**inputs)
    seen = []
    def fake_generate(**kw):
        assert kw['max_new_tokens'] == 768 and kw['do_sample'] and kw['temperature'] == kw['top_p'] == 1
        seen.append(bool(get_layers(model)[2]._forward_hooks))
        with Recorder(model, [2]) as after:
            model(input_ids=kw['input_ids'], attention_mask=kw['attention_mask'])
        assert torch.allclose(after.acts[2] - before.acts[2], vector, atol=1e-6)
        return torch.cat((kw['input_ids'], torch.full((5, 1), 2, dtype=torch.long)), dim=1)
    monkeypatch.setattr(model, 'generate', fake_generate)
    result = entry.collect(model, tok, jobs, [{'id': jobs[0]['vector_id'], 'vector': vector}], [], layer=2)
    assert len(result) == 5 and seen == [True]
    assert all(r['input_ids'] and r['n_generated_tokens'] == 1 for r in result)
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert all(not b._forward_hooks for b in get_layers(model))


@pytest.mark.parametrize('problem', ['prompt', 'decode', 'eos', 'count'])
def test_reused_raw_prompt_tokens_must_verify(problem):
    tok = Tokenizer()
    row = {'q': '题目', 'system': '记忆', 'prompt': chat_prompt(tok, '题目', system='记忆'),
           'a': '回答', 'token_ids': [2], 'n_generated_tokens': 1, 'finish_reason': 'eos'}
    if problem == 'prompt':
        row['prompt'] = '不同提示'
    elif problem == 'decode':
        row['a'] = '旧文不符'
    elif problem == 'eos':
        row['token_ids'] = [3]
    else:
        row['n_generated_tokens'] = 2
    with pytest.raises(ValueError):
        entry.verify_answer(row, tok, {2})


def test_capability_prediction_recompute_and_tamper_rejection(tmp_path, monkeypatch):
    from e1_twin_raw_capability import PROTOCOL, fixed_vectors
    from test_twin_raw_capability import states
    from antler.eval.capability import capability_summary
    vectors = fixed_vectors(*states())
    sample = [{'source_index': i, 'subject': '科目', 'answer': 0} for i in range(1400)]
    indices = tmp_path / 'source_snapshot/sample_indices.json'
    indices.parent.mkdir()
    entry.write_json(indices, [{'source_index': r['source_index'], 'subject': r['subject']} for r in sample])
    monkeypatch.setattr(entry, 'INDICES_SHA', digest(indices))
    entry.write_json(tmp_path / 'sample.json', sample)
    entry.write_json(tmp_path / 'run_meta.json', {'status': 'completed', 'protocol': PROTOCOL,
                    'conditions_completed': 9, 'question_conditions_completed': 12600, 'all_parameters_frozen': True})
    baseline = {'n': 1400, 'samples': [{**r, 'prediction': 0, 'correct': True} for r in sample]}
    entry.write_json(tmp_path / 'baseline.json', baseline)
    torch.save({v['id']: v['vector'] for v in vectors}, tmp_path / 'vectors.pt')
    reports = {}
    for i, v in enumerate(vectors):
        injected = copy.deepcopy(baseline)
        for row in injected['samples'][:29 if i == 0 else 28]:
            row.update(prediction=1, correct=False)
        entry.write_json(tmp_path / (v['id'] + '.json'), injected)
        report = capability_summary(baseline, injected)
        report.update(vector_sha256=v['vector_sha256'], status='capability_pass' if report['within_2pp'] else 'capability_failure',
                      formal_e1_pass=False, behavior_validated=False)
        reports[v['id']] = report
        entry.write_json(tmp_path / (v['id'] + '_summary.json'), report)
    entry.write_json(tmp_path / 'summary.json', reports)
    def seal():
        entry.write_json(tmp_path / 'artifact_hashes.json', {p.relative_to(tmp_path).as_posix(): digest(p)
                         for p in tmp_path.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    seal()
    result = entry.validate_capability(tmp_path, vectors)
    assert not result[vectors[0]['id']]['within_2pp']
    assert result[vectors[1]['id']]['within_2pp']
    reports[vectors[0]['id']]['within_2pp'] = True
    entry.write_json(tmp_path / 'summary.json', reports)
    seal()
    with pytest.raises(ValueError, match='逐题预测'):
        entry.validate_capability(tmp_path, vectors)


@pytest.mark.parametrize('problem', [None, 'generated', 'scored', 'judge', 'answers', 'extra_file'])
def test_only_explicitly_approved_fixed_empty_failure_can_get_new_claim(tmp_path, monkeypatch, problem):
    failed = tmp_path / entry.EMPTY_FAILURE_RUN
    failed.mkdir()
    claim = tmp_path / (entry.STATE_RUN + '.claim')
    claim.write_text(str(failed), encoding='utf-8')
    meta = {'status': 'failed', 'stage': 'validation', 'protocol': entry.PROTOCOL,
            'generated': 0, 'reused': 0, 'scored': 0, 'judge': None}
    answers = []
    if problem in ('generated', 'scored'):
        meta[problem] = 1
    elif problem == 'judge':
        meta['judge'] = {'model': '评委'}
    elif problem == 'answers':
        answers = [{'a': '旧回答'}]
    elif problem == 'extra_file':
        (failed / 'judge_requests.jsonl').write_text('{}\n', encoding='utf-8')
    entry.write_json(failed / 'run_meta.json', meta)
    entry.write_json(failed / 'scored_answers.json', answers)
    monkeypatch.setattr(entry, 'EMPTY_FAILURE_SHA', digest(failed / 'run_meta.json'))
    entry.write_json(failed / 'artifact_hashes.json', {p.name: digest(p) for p in failed.iterdir()})
    original = claim.read_bytes()
    assert entry.output_root(tmp_path) == tmp_path
    if problem:
        with pytest.raises(ValueError):
            entry.output_root(tmp_path, approved=True)
    else:
        assert entry.output_root(tmp_path, approved=True) == tmp_path / 'approved_empty_restart'
    assert claim.read_bytes() == original
