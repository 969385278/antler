"""C12：固定能力入口的幅度、身份、能力失败与CPU真实干预测试。"""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import e1_twin_raw_capability as entry
from antler.core.model import get_layers
from antler.eval.capability import stratified_sample
from antler.memory.twin_state import analyze
from test_core import model
from test_twin_state_extract import TinyTokenizer


def states():
    acts = torch.zeros(9, 60, 32)
    for i in range(4):
        acts[i, :, 2] = .02 * (i + 1)
        acts[i, :, 10] = .1 * (i + 1)
    basis = {'U': torch.eye(5, 32), 's': 2., 'dims': ['O', 'C', 'E', 'A', 'N']}
    return analyze(acts, basis['U'], basis['s']), acts, basis


def rows():
    return [{'question': f'题{i}', 'choices': ['甲', '乙', '丙', '丁'], 'answer': i % 4,
             'subject': '科目'} for i in range(6)]


def test_physical_amplitude_no_renormalization_or_feedback_substitution():
    summary, acts, basis = states()
    vectors = entry.fixed_vectors(summary, acts, basis)
    assert len(vectors) == 8
    assert [v['cell'] for v in vectors] == ['a-1_r-1'] * 2 + ['a-1_r1'] * 2 + ['a1_r-1'] * 2 + ['a1_r1'] * 2
    assert vectors[0]['vector'][10] == pytest.approx(.1)
    assert vectors[1]['vector'][10] == 0
    assert vectors[1]['vector'][2] == pytest.approx(.02)
    assert vectors[0]['norm_physical'] != pytest.approx(1.)
    assert vectors[0]['vector_sha256'] != vectors[1]['vector_sha256']
    assert vectors[2]['vector'][10] == pytest.approx(.2)


@pytest.mark.parametrize('problem', ['tamper', 'nan', 'scale', 'dims', 'layer'])
def test_bad_state_stops(problem):
    summary, acts, basis = states()
    if problem == 'tamper':
        summary['groups']['all60']['a1_r1']['full_state'][10] += 1
    elif problem == 'nan':
        summary['groups']['all60']['a1_r1']['full_state'][10] = float('nan')
    elif problem == 'scale':
        summary['scale'] = 3.
    elif problem == 'dims':
        basis['dims'] = list(reversed(basis['dims']))
    else:
        summary['layer'] = 17
    with pytest.raises(ValueError):
        entry.fixed_vectors(summary, acts, basis)


@pytest.mark.parametrize('problem', ['order', 'identity', 'missing'])
def test_canonical_must_match_every_identity_in_order(problem):
    pool = rows()
    sample = stratified_sample(pool, total=6, seed=0)
    identities = [{'source_index': r['source_index'], 'subject': r['subject']} for r in sample]
    assert entry.fixed_sample(pool, identities, total=6) == sample
    altered = copy.deepcopy(identities)
    if problem == 'order':
        altered.reverse()
    elif problem == 'identity':
        altered[0]['source_index'] = 999
    else:
        altered.pop()
    with pytest.raises(ValueError):
        entry.fixed_sample(pool, altered, total=6)


def test_preflight_does_not_load_model_or_judge(monkeypatch):
    monkeypatch.setattr(entry, 'setup', lambda *a, **kw: ({}, None, None, None))
    monkeypatch.setattr(entry, 'validate', lambda _: ([], [], []))
    monkeypatch.setattr(entry, 'load_model', lambda *a, **kw: pytest.fail('不应加载模型'))
    monkeypatch.setattr(entry, 'verify_model', lambda *a: pytest.fail('不应打开权重'))
    result = entry.run(SimpleNamespace(execute=False))
    assert not result['model_loaded'] and not result['judge_loaded']


class CapabilityTokenizer(TinyTokenizer):
    def encode(self, text, add_special_tokens=False):
        return ['ABCD'.index(text) + 4]


def test_actual_tiny_cpu_forwards_and_hook_cleanup(model):
    sample = stratified_sample(rows(), total=6, seed=0)
    vectors = entry.fixed_vectors(*states())
    saved = []
    original = {k: v.clone() for k, v in model.state_dict().items()}
    reports = entry.collect(model, CapabilityTokenizer(), sample, vectors, layer=2,
                            progress=lambda *a: saved.append(a))
    assert len(reports) == 8 and len(saved) == 9
    assert all(r['n'] == 6 and not r['formal_e1_pass'] and not r['behavior_validated'] for r in reports.values())
    assert all(not layer._forward_hooks for layer in get_layers(model))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())


def test_capability_failure_preserved_and_no_selection(monkeypatch, model):
    calls = []
    def score(m, tok, sample, **kw):
        calls.append(kw)
        prediction = 0 if len(calls) == 1 else 1
        records = [{'source_index': i, 'subject': '科目', 'answer': 0,
                    'prediction': prediction, 'correct': prediction == 0} for i in range(100)]
        return {'n': 100, 'accuracy': float(prediction == 0), 'samples': records}
    monkeypatch.setattr(entry, 'mmlu_accuracy', score)
    reports = entry.collect(model, None, None, entry.fixed_vectors(*states()), layer=2)
    assert len(calls) == 9
    assert all(r['status'] == 'capability_failure' and not r['within_2pp'] for r in reports.values())


def test_interruption_preserves_prefix_and_removes_steer_hooks(model):
    sample = stratified_sample(rows(), total=6, seed=0)
    saved = []
    def save(*args):
        saved.append(args)
        if len(saved) == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        entry.collect(model, CapabilityTokenizer(), sample, entry.fixed_vectors(*states()), layer=2, progress=save)
    assert len(saved) == 2
    assert all(not layer._forward_hooks for layer in get_layers(model))
