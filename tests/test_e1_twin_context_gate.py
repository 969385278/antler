"""C12：因素配对、反馈交互和CPU微型模型的失败停止测试。"""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from antler.eval import twin_context_gate as gate
from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from test_core import model
from test_e1_behavior_collection import Tok


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    return json.loads((ROOT / 'data/diagnostics/twin_outcome_dev_v1.json').read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def trait_data(monkeypatch):
    monkeypatch.setattr(gate, 'load_trait', lambda *args: {
        'questions': [f'固定测量问题{i}' for i in range(20)], 'eval_prompt': '{question}\n{answer}'})


def scored(data, *, interaction=5., valence=7.):
    rows = gate.jobs_for(data)
    for row in rows:
        direction = 1 if row['pole'] == 'high' else -1
        a, r = row['action_sign'], row['outcome_sign']
        row.update(trait=50., coherence=93., task=95., finish_reason='eos')
        if row['arm'] == 'context':
            row['trait'] += direction * (interaction * a * r + valence * r)
            row['coherence'] += r
        elif row['arm'] == 'neutral':
            row['trait'] += 10 * (row['memory_index'] - 2)
        else:
            row['trait'] = 45.
    return rows


def test_exact_budget_and_calib_only(data):
    rows = gate.jobs_for(data)
    assert len(rows) == 2520
    assert {name: sum(r['arm'] == name for r in rows) for name in ('context', 'neutral', 'none')} == {
        'context': 1440, 'neutral': 900, 'none': 180}
    assert {r['dim'] for r in rows} == {'O', 'E', 'A'}
    assert len({r['id'] for r in rows}) == 2520
    assert {r['question_index'] for r in rows} == set(range(10))
    assert {r['seed'] for r in rows} == {0, 1, 2}
    assert all(r['split'] == 'calib' for r in rows)
    for offset in range(0, len(rows), 10):
        assert len({(r['dim'], r['pole'], r['arm'], r['episode_id'], r['memory_index'], r['seed'])
                    for r in rows[offset:offset + 10]}) == 1


def test_shared_prefix_action_and_outcome_are_controlled(data):
    episodes = gate.expand(data)
    assert len(episodes) == 24
    for scene in data['scenarios']:
        subset = [e for e in episodes if e['scenario_id'] == scene['id']]
        assert {(e['action_sign'], e['outcome_sign']) for e in subset} == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        for episode in subset:
            assert episode['text'] == ' '.join((scene['prefix'], scene['actions'][str(episode['action_sign'])],
                                               scene['outcomes'][str(episode['outcome_sign'])]))
            assert episode['predicted_sign'] == episode['action_sign'] * episode['outcome_sign']
    rows = gate.jobs_for(data)
    assert all(r['system'] == BASE_SYSTEM for r in rows if r['arm'] == 'none')
    assert all(r['system'] == MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORIES[r['memory_index']])
               for r in rows if r['arm'] == 'neutral')


@pytest.mark.parametrize('change', [
    lambda d: d.update(split='test'),
    lambda d: d.update(formal_d1_data=True),
    lambda d: d['scenarios'].pop(),
    lambda d: d['scenarios'].reverse(),
    lambda d: d['scenarios'][0].update(dim='C'),
    lambda d: d['scenarios'][0]['outcomes'].pop('1'),
    lambda d: d['scenarios'][0]['actions'].update({'1': d['scenarios'][0]['actions']['-1']}),
    lambda d: d['scenarios'][0]['outcomes'].update({'1': 'After that I began acting differently.'}),
])
def test_changed_design_rejected(data, change):
    change(data)
    with pytest.raises(ValueError):
        gate.expand(data)


def test_interaction_is_separate_from_generic_valence_and_quality(data):
    report = gate.summarize(scored(data), data)
    assert report['gate_counts']['established'] == 6
    assert not report['formal_e1_pass'] and not report['behavior_validated']
    for scene in report['reports']:
        high, low = scene['poles']
        for pole in (high, low):
            trait = pole['metrics']['trait']
            assert trait['neutral']['mean'] == 50.
            assert trait['interaction_high_axis']['mean'] == 10.
            assert trait['interaction_high_axis']['question_bootstrap_ci'] == [10., 10.]
            assert trait['valence_high_axis']['mean'] == 14.
            assert trait['interaction_high_axis']['per_seed_means'] == [10., 10., 10.]
            quality = pole['metrics']['coherence']
            assert quality['valence_raw']['mean'] == 2.
            assert 'interaction_high_axis' not in quality
            assert quality['cells']['a1_r1']['minus_neutral']['mean'] == 1.
        assert low['metrics']['trait']['interaction_raw']['mean'] == -10.


def test_valence_alone_does_not_pass_gate_and_opposite_is_retained(data):
    report = gate.summarize(scored(data, interaction=0), data)
    assert report['gate_counts']['not_established'] == 6
    assert all(p['metrics']['trait']['valence_high_axis']['mean'] == 14.
               for r in report['reports'] for p in r['poles'])
    reversed_report = gate.summarize(scored(data, interaction=-5), data)
    assert reversed_report['gate_counts']['opposite'] == 6


def test_question_pairing_and_seed_differences_retained(data):
    rows = scored(data)
    for row in rows:
        if row['arm'] == 'context':
            direction = 1 if row['pole'] == 'high' else -1
            row['trait'] += direction * row['seed'] * row['action_sign'] * row['outcome_sign']
    result = gate.summarize(rows, data)
    for scene in result['reports']:
        for pole in scene['poles']:
            effect = pole['metrics']['trait']['interaction_high_axis']
            assert effect['mean'] == 12.
            assert effect['per_seed_means'] == [10., 12., 14.]
            assert effect['question_bootstrap_ci'] == [12., 12.]


def test_missing_scores_keep_quality_and_prevent_partial_pass(data):
    rows = scored(data)
    rows[0]['trait'] = None
    result = gate.summarize(rows, data)
    assert result['gate_counts']['incomplete'] == 2
    assert result['reports'][0]['poles'][0]['metrics']['trait']['status'] == 'incomplete'
    assert result['reports'][0]['poles'][0]['metrics']['coherence']['status'] == 'complete'
    assert len(result['reports']) == 6


@pytest.mark.parametrize('fault', ['missing', 'reordered', 'system', 'question', 'identity'])
def test_summary_requires_full_exact_work_plan(data, fault):
    rows = scored(data)
    if fault == 'missing':
        rows.pop()
    elif fault == 'reordered':
        rows.reverse()
    elif fault == 'system':
        rows[0]['system'] = '另一系统提示'
    elif fault == 'question':
        rows[0]['question_index'] = 10
    else:
        rows[0]['id'] = 100
    with pytest.raises(ValueError):
        gate.summarize(rows, data)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'experiments'))
    return importlib.import_module('e1_twin_context_gate')


def test_default_preflight_has_no_model_judge_or_claim(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'setup', lambda *a, **k: ({'out_path': tmp_path}, None, None, None))
    monkeypatch.setattr(runner, 'load_model', lambda *a: pytest.fail('预检不得加载模型'))
    monkeypatch.setattr(runner, 'SequentialJudge', lambda *a, **k: pytest.fail('预检不得加载评委'))
    args = SimpleNamespace(materials=str(ROOT / 'data/diagnostics/twin_outcome_dev_v1.json'), execute=False)
    result = runner.run(args)
    assert result['n_jobs'] == 2520 and not result['model_loaded'] and not result['judge_loaded']
    assert not list(tmp_path.iterdir())


def test_judge_stops_immediately_and_never_queues_other_calls(runner, monkeypatch):
    calls = []
    def fail(self, prompt, retries):
        calls.append((prompt, retries))
        raise RuntimeError('模拟接口故障')
    monkeypatch.setattr(runner.Judge, '_call', fail)
    judge = object.__new__(runner.SequentialJudge)
    monkeypatch.setattr(judge, '_key', lambda p: p)
    with pytest.raises(RuntimeError):
        judge.score_many(['第一请求', '第二请求'])
    assert calls == [('第一请求', 1)]


def test_unparseable_duplicate_is_not_re_requested(runner, monkeypatch):
    calls = []
    def invalid(self, prompt, retries):
        calls.append((prompt, retries))
        return 'unparseable'
    monkeypatch.setattr(runner.Judge, '_call', invalid)
    judge = object.__new__(runner.SequentialJudge)
    monkeypatch.setattr(judge, '_key', lambda p: p)
    assert judge.score_many(['同一请求']) == [None]
    assert judge.score_many(['同一请求']) == [None]
    assert calls == [('同一请求', 1)]


def test_tiny_generation_failure_archives_tokens_and_blocks_restart(runner, monkeypatch, tmp_path, model):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    cfg = {'model': 'tiny', 'out_path': tmp_path / 'output', 'device': 'cpu', 'dtype': 'float32'}
    monkeypatch.setattr(runner, 'setup', lambda *a, **k: (cfg, None, None, None))
    monkeypatch.setattr(runner, 'verify_model', lambda *a: {})
    monkeypatch.setattr(runner, 'seal_sources', lambda *a: None)
    monkeypatch.setattr(runner, 'load_model', lambda *a: (model, Tok()))
    actual_judge = {'model': 'fixed', 'base_url': 'https://example.invalid', 'max_tokens': 16, 'extra_body': {}}
    monkeypatch.setattr(runner, 'SequentialJudge', lambda **k: SimpleNamespace(**actual_judge))
    prior = tmp_path / runner.PRIOR_JUDGE
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({'judge': actual_judge}), encoding='utf-8')
    monkeypatch.setattr(runner, 'sha256', lambda p: runner.DATA_SHA)
    original = runner.generate
    weights = [p.detach().clone() for p in model.parameters()]
    calls = []
    def interrupted(model_, tok, prompts, **kwargs):
        calls.append(prompts)
        assert 'steer' not in kwargs and kwargs['max_new_tokens'] == 768
        if len(calls) == 2:
            raise RuntimeError('模拟生成中断')
        kwargs.update(max_new_tokens=1, temperature=0)
        return original(model_, tok, prompts, **kwargs)
    monkeypatch.setattr(runner, 'generate', interrupted)
    args = SimpleNamespace(config='test-config', materials=str(ROOT / 'data/diagnostics/twin_outcome_dev_v1.json'),
                           model_fingerprint='unused', execute=True)
    with pytest.raises(RuntimeError, match='模拟生成中断'):
        runner.run(args)
    out = next(p for p in (cfg['out_path'] / 'e1_twin_context_gate').iterdir() if p.is_dir())
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'failed' and meta['generated'] == 10 and meta['scored'] == 0
    rows = [json.loads(line) for line in (out / 'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 10 and all(len(r['token_ids']) == 1 for r in rows)
    assert all(torch.equal(before, after) for before, after in zip(weights, model.parameters()))
    assert (out / 'artifact_hashes.json').exists()
    with pytest.raises(FileExistsError):
        runner.run(args)
    assert len(calls) == 2


def test_material_hash_guard(runner, tmp_path):
    path = tmp_path / 'changed.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='封存'):
        runner.read_materials(path)
