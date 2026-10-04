"""C11：采集身份、五记忆与固定评委乱序映射；真实CPU微型模型验证钩子。"""
from copy import deepcopy

import pytest
import torch

from antler.eval import e1_behavior_collection as collection
from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES
from antler.core.model import get_layers
from test_core import model
from test_generate import Tokenizer


class Tok(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        return '4 5'


@pytest.fixture(autouse=True)
def trait_data(monkeypatch):
    monkeypatch.setattr(collection, 'load_trait', lambda *a: {
        'questions': [f'题目{i}' for i in range(20)], 'eval_prompt': '原始评分提示'})


def collect(model, **kwargs):
    defaults = dict(experience='一次经历', dim='O', pole='high', seeds=[3],
                    conditions={'A': {'layer': 1, 'vector': torch.zeros(32), 'phase': 'all'}},
                    temperature=0, max_new_tokens=1)
    defaults.update(kwargs)
    return collection.collect_answers(model, Tok(), **defaults)


def test_tiny_model_all_controls_and_zero_injection_identity(model):
    weights = [p.detach().clone() for p in model.parameters()]
    saved = []
    rows = collect(model, on_record=lambda r: saved.append(deepcopy(r)))
    assert len(rows) == 120 and rows == saved
    assert {r['question_index'] for r in rows} == set(range(10, 20))
    for arm in ('neutral', 'A'):
        selected = [r for r in rows if r['arm'] == arm]
        assert len(selected) == 50
        assert {r['memory_index'] for r in selected} == set(range(5))
        for r in selected:
            assert r['system'] == MEMORY_TEMPLATE.format(experience=NEUTRAL_MEMORIES[r['memory_index']])
    assert all(r['system'] == BASE_SYSTEM for r in rows if r['arm'] == 'none')
    assert all(r['system'] == MEMORY_TEMPLATE.format(experience='一次经历') for r in rows if r['arm'] == 'context')
    assert len({tuple(r['token_ids']) for r in rows}) == 1
    assert all(r['n_generated_tokens'] == len(r['token_ids']) == 1 for r in rows)
    assert all(torch.equal(before, after) for before, after in zip(weights, model.parameters()))
    assert all(not layer._forward_hooks for layer in get_layers(model))


def test_phase_forwarded_and_condition_copied(model, monkeypatch):
    calls = []
    def fake_generate(*args, **kwargs):
        calls.append(kwargs)
        return [{'text': '回答', 'token_ids': [7], 'n_generated_tokens': 1, 'finish_reason': 'length'}]
    monkeypatch.setattr(collection, 'generate', fake_generate)
    v = torch.ones(32)
    rows = collect(model, conditions={'B': {'layer': 2, 'vector': v, 'phase': 'decode'}})
    assert len(calls) == 120
    assert all(c['steer'] is None for c in calls[:70])
    assert all(c['steer_phase'] == 'decode' for c in calls[70:])
    assert all(torch.equal(c['steer'][2], v) for c in calls[70:])
    assert all(r['condition']['phase'] == 'decode' for r in rows if r['arm'] == 'B')


@pytest.mark.parametrize('change', [
    {'seeds': [0, 0]}, {'seeds': []}, {'pole': 'bad'}, {'temperature': float('nan')},
    {'conditions': {'neutral': {'layer': 1, 'vector': torch.zeros(32), 'phase': 'all'}}},
    {'conditions': {'A': {'layer': 99, 'vector': torch.zeros(32), 'phase': 'all'}}},
    {'conditions': {'A': {'layer': 1, 'vector': torch.full((32,), float('nan')), 'phase': 'all'}}},
])
def test_invalid_request_rejected_before_generation(model, monkeypatch, change):
    monkeypatch.setattr(collection, 'generate', lambda *a, **k: pytest.fail('不应执行生成'))
    with pytest.raises(ValueError): collect(model, **change)


def records():
    return [{'dim': 'O', 'pole': 'high', 'split': 'test', 'arm': arm, 'memory_index': None,
             'question_index': 10, 'sample_index': 0, 'q': '同一题', 'a': str(i)}
            for i, arm in enumerate(['none', 'context', 'A', 'B'])]


class Judge:
    def trait_scores(self, prompt, qs, answers):
        assert prompt == '原始评分提示' and qs == ['同一题'] * 4
        self.order = list(answers)
        return [int(a) for a in answers]

    def coherence_scores(self, qs, answers):
        assert answers == self.order
        return [90 for _ in answers]

    def task_scores(self, qs, answers):
        assert answers == self.order
        return [None if a == '2' else 85 for a in answers]


def test_blind_shuffle_restores_identity_without_retry_or_mutation():
    source = records(); original = deepcopy(source); judge = Judge()
    rows = collection.score_answers(judge, source)
    assert judge.order != ['0', '1', '2', '3']
    assert [r['trait'] for r in rows] == [0, 1, 2, 3]
    assert rows[2]['task'] is None
    assert source == original
    assert sorted(r['judge_order_index'] for r in rows) == list(range(4))


def test_duplicate_identity_rejected_before_judge():
    source = records()
    with pytest.raises(ValueError): collection.score_answers(None, source + source[:1])


def test_incomplete_judge_response_rejected():
    class ShortJudge(Judge):
        def task_scores(self, *args): return []
    with pytest.raises(ValueError): collection.score_answers(ShortJudge(), records())


def test_collection_to_summary_preserves_all_five_controls(model, monkeypatch):
    monkeypatch.setattr(collection, 'generate', lambda *a, **k: [
        {'text': '回答', 'token_ids': [7], 'n_generated_tokens': 1, 'finish_reason': 'length'}])
    rows = collect(model)
    for r in rows:
        r.update(trait={'neutral': 30, 'context': 50, 'none': 20, 'A': 40}[r['arm']],
                 coherence=90, task=85)
    result = collection.summarize_scored_answers(rows, intended_sign=1)
    assert result['arms']['A']['transfer_ratio'] == .5
    assert result['arms']['A']['neutral']['trait'] == 30
    assert result['arms']['A']['condition']['layer'] == 1
    broken = deepcopy(rows)
    next(r for r in broken if r['arm'] == 'A')['condition']['layer'] = 2
    with pytest.raises(ValueError): collection.summarize_scored_answers(broken, intended_sign=1)
    broken = deepcopy(rows)
    next(r for r in broken if r['arm'] == 'neutral')['system'] = BASE_SYSTEM
    with pytest.raises(ValueError): collection.summarize_scored_answers(broken, intended_sign=1)
    broken = deepcopy(rows)
    next(r for r in broken if r['arm'] == 'A')['seed'] = 19
    with pytest.raises(ValueError): collection.summarize_scored_answers(broken, intended_sign=1)
