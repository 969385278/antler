"""C11：固定上下文诊断的预算、隔离、配对区间与单次请求规则。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_core import model
from test_e1_behavior_collection import Tok


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    m = importlib.import_module('e1_context_control')
    monkeypatch.setattr(m, 'load_trait', lambda *a: {'questions': [f'问题{i}' for i in range(20)], 'eval_prompt': '固定提示'})
    return m


def episodes():
    return [{'id': f'{d}{s}', 'intended': {d: s}, 'split': 'dev', 'text': f'经历{d}{s}'}
            for d in 'OCEAN' for s in (-1, 1)]


def scored(runner):
    return [{**j, 'trait': 30 if j['arm'] == 'neutral' else 50 if j['arm'] == 'context' else 20,
             'coherence': 90, 'task': 85} for j in runner.jobs_for(episodes())]


def test_exact_budget_and_baseline_reuse(runner):
    jobs = runner.jobs_for(episodes())
    assert len(jobs) == 2400
    assert {a: sum(j['arm'] == a for j in jobs) for a in ('none', 'neutral', 'context')} == {
        'none': 300, 'neutral': 1500, 'context': 600}
    assert all(j['split'] == 'calib' and 0 <= j['question_index'] < 10 for j in jobs)
    assert {j['seed'] for j in jobs} == {0, 1, 2}
    assert len({j['id'] for j in jobs}) == 2400
    for start in range(0, 2400, 10):
        batch = jobs[start:start+10]
        assert len({(j['dim'],j['pole'],j['arm'],j['memory_index'],j['episode_id'],j['seed']) for j in batch}) == 1


@pytest.mark.parametrize('fault', ['count', 'split', 'target', 'duplicate'])
def test_changed_episode_plan_rejected(runner, fault):
    rows = episodes()
    if fault == 'count': rows.pop()
    elif fault == 'split': rows[0]['split'] = 'test'
    elif fault == 'target': rows[0]['intended'] = {'O': 1}
    else: rows[0]['id'] = rows[1]['id']
    with pytest.raises(ValueError): runner.jobs_for(rows)


def test_summary_uses_five_neutral_and_low_pole_sign(runner):
    rows = scored(runner)
    for r in rows:
        if r['arm'] == 'neutral': r['trait'] = 10 * r['memory_index'] + 10
    summary = runner.summarize(rows, episodes())
    high, low = summary['reports'][:2]
    assert high['metrics']['trait']['neutral'] == 30
    assert high['metrics']['trait']['context_minus_neutral'] == 20
    assert high['metrics']['trait']['question_bootstrap_ci'] == [20, 20]
    assert not high['context_matches_intended'] and low['context_matches_intended']
    assert len(summary['reports']) == 20 and not summary['formal_e1_pass']


def test_missing_score_retained_without_partial_average(runner):
    rows = scored(runner)
    next(r for r in rows if r['arm'] == 'neutral')['trait'] = None
    summary = runner.summarize(rows, episodes())
    assert sum(r['status'] == 'incomplete_scores' for r in summary['reports']) == 2
    assert 'metrics' not in summary['reports'][0]


@pytest.mark.parametrize('fault', ['missing', 'system', 'question', 'reordered'])
def test_summary_requires_exact_full_plan(runner, fault):
    rows = scored(runner)
    if fault == 'missing': rows.pop()
    elif fault == 'system': rows[0]['system'] = '其他提示'
    elif fault == 'question': rows[0]['question_index'] = 10
    else: rows.reverse()
    with pytest.raises(ValueError): runner.summarize(rows, episodes())


def test_default_preflight_does_not_load_model_or_judge(runner, monkeypatch, tmp_path):
    source = tmp_path/'episodes.json'; source.write_text(json.dumps({'episodes': episodes()}))
    plan = tmp_path/'plan.json'; plan.write_text('{}')
    monkeypatch.setattr(runner, 'sha256', lambda p: runner.EPISODES_SHA if Path(p) == source else runner.PLAN_SHA)
    monkeypatch.setattr(runner, 'setup', lambda *a, **k: ({}, None, None, None))
    monkeypatch.setattr(runner, 'load_model', lambda *a, **k: pytest.fail('不应加载模型'))
    monkeypatch.setattr(runner, 'SingleAttemptJudge', lambda *a, **k: pytest.fail('不应加载评委'))
    args = SimpleNamespace(experiences=str(source), plan=str(plan), execute=False)
    assert runner.run(args) is None


def test_no_api_retry(runner, monkeypatch):
    calls = []
    def fail(self, prompt, retries):
        calls.append(retries)
        raise RuntimeError('模拟接口错误')
    monkeypatch.setattr(runner.Judge, '_call', fail)
    judge = object.__new__(runner.SingleAttemptJudge)
    with pytest.raises(RuntimeError): judge._call('固定提示')
    assert calls == [1]


def test_tiny_generation_failure_archives_and_prevents_restart(runner, monkeypatch, tmp_path, model):
    source = tmp_path/'episodes.json'; source.write_text(json.dumps({'episodes': episodes()}))
    plan = tmp_path/'plan.json'; plan.write_text('{}')
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'sha256', lambda p: runner.EPISODES_SHA if Path(p) == source else runner.PLAN_SHA)
    cfg = {'model': 'tiny', 'out_path': tmp_path/'output', 'device': 'cpu', 'dtype': 'float32'}
    monkeypatch.setattr(runner, 'setup', lambda *a, **k: (cfg, None, None, None))
    monkeypatch.setattr(runner, 'verify_model', lambda *a: {})
    monkeypatch.setattr(runner, 'load_model', lambda *a: (model, Tok()))
    judge_meta = dict(model='fixed', base_url='https://example.invalid', max_tokens=16, extra_body={})
    monkeypatch.setattr(runner, 'SingleAttemptJudge', lambda **k: SimpleNamespace(**judge_meta))
    prior = tmp_path/'outputs/main/prompt_persona/d09a117a119541a2985f54a77c8e4391/run_meta.json'
    prior.parent.mkdir(parents=True); prior.write_text(json.dumps({'judge': judge_meta}))
    original_generate = runner.generate
    calls = []
    def interrupted(model, tok, prompts, **kwargs):
        calls.append(prompts)
        assert 'steer' not in kwargs
        if len(calls) == 2: raise RuntimeError('固定模拟中断')
        kwargs.update(max_new_tokens=1, temperature=0)
        return original_generate(model, tok, prompts, **kwargs)
    monkeypatch.setattr(runner, 'generate', interrupted)
    args = SimpleNamespace(config='dummy', experiences=str(source), plan=str(plan), execute=True,
                           model_fingerprint='unused')
    with pytest.raises(RuntimeError, match='固定模拟中断'): runner.run(args)
    root = cfg['out_path']/'e1_context_control'
    out = next(p for p in root.iterdir() if p.is_dir())
    meta = json.loads((out/'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'failed' and meta['generated'] == 10
    rows = [json.loads(line) for line in (out/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 10 and all(len(r['token_ids']) == 1 for r in rows)
    assert (out/'artifact_hashes.json').exists()
    with pytest.raises(FileExistsError): runner.run(args)
    assert len(calls) == 2
