"""C11：共享老师分布、留出评估和冻结模型的CPU数值验证。"""
from dataclasses import replace

import pytest
import torch

from antler.core.model import get_layers
from antler.memory.kl_validation import KLItem, evaluate_kl, fit_and_evaluate, validate_items
from antler.memory.write import _seq_logprobs
from test_core import model, D


def item(model, name, teacher_vector=None, length=2):
    prompts = tuple(torch.tensor([5, 6, 7 + i]) for i in range(5))
    response = torch.tensor([12, 13, 14][:length])
    with torch.no_grad():
        teacher = _seq_logprobs(model, prompts[0], response,
                               steer=None if teacher_vector is None else {2: teacher_vector}).cpu()
    return KLItem(name, response, teacher, prompts)


def test_five_neutrals_and_equal_probe_weight(model):
    rows = [item(model, 'a', length=1), item(model, 'b', length=3)]
    result = evaluate_kl(model, rows, layer=2)
    independent = []
    with torch.no_grad():
        for r in rows:
            values = []
            for ids in r.neutral_prompt_ids:
                student = _seq_logprobs(model, ids, r.response_ids)
                values.append(float(torch.nn.functional.kl_div(student, r.teacher_logp.exp(), reduction='sum') / len(r.response_ids)))
            independent.append(sum(values) / 5)
    assert result['kl'] == pytest.approx(sum(independent) / 2, abs=2e-7)
    assert all(len(r['neutral_kls']) == 5 for r in result['per_probe'])
    zero = evaluate_kl(model, rows, layer=2, vector=torch.zeros(D))
    assert zero['kl'] == pytest.approx(result['kl'], abs=1e-7)


@pytest.mark.parametrize('basis', [torch.eye(5, D), None])
def test_final_update_is_evaluated_and_weights_frozen(model, basis):
    teacher_vector = torch.arange(D).float() / D * .04
    train = [item(model, 'fit', teacher_vector)]
    heldout = [item(model, 'heldout', teacher_vector)]
    before = [p.detach().clone() for p in model.parameters()]
    result = fit_and_evaluate(model, train, heldout, basis=basis, scale=1., layer=2, steps=2, lr=.005)
    actual = evaluate_kl(model, heldout, layer=2, vector=result['vector'])
    assert result['final'] == actual
    assert result['vector'].abs().sum() > 0
    assert result['coordinates'].shape == ((D,) if basis is None else (5,))
    assert all(torch.equal(old, p) and p.grad is None for old, p in zip(before, model.parameters()))
    assert all(not layer._forward_hooks for layer in get_layers(model))
    assert result['behavior_validated'] is False


def test_eval_labels_do_not_affect_fit(model):
    train = [item(model, 'fit', torch.ones(D) * .02)]
    test = item(model, 'eval')
    altered = replace(test, teacher_logp=torch.log_softmax(torch.arange(64).float().repeat(2, 1) / 10, -1))
    kwargs = dict(basis=torch.eye(5, D), scale=1., layer=2, steps=1)
    a = fit_and_evaluate(model, train, [test], **kwargs)
    b = fit_and_evaluate(model, train, [altered], **kwargs)
    assert torch.equal(a['coordinates'], b['coordinates'])
    assert a['final']['kl'] != b['final']['kl']


def test_overlap_and_invalid_inputs_rejected(model):
    r = item(model, 'Probe  one')
    with pytest.raises(ValueError, match='重叠'):
        fit_and_evaluate(model, [r], [replace(r, probe=' probe ONE ')], basis=None, scale=1, layer=2)
    with pytest.raises(ValueError, match='五条'):
        validate_items([replace(r, neutral_prompt_ids=r.neutral_prompt_ids[:1])])
    with pytest.raises(ValueError, match='老师分布'):
        validate_items([replace(r, teacher_logp=r.teacher_logp + 1)])
    with pytest.raises(ValueError, match='非空'):
        validate_items([replace(r, response_ids=torch.tensor([], dtype=torch.long))])
    with pytest.raises(ValueError, match='满行秩'):
        fit_and_evaluate(model, [r], [replace(r, probe='other')], basis=torch.zeros(5, D), scale=1, layer=2)


def test_hook_cleanup_after_teacher_shape_failure(model):
    r = item(model, 'eval')
    smaller_vocab = replace(r, teacher_logp=torch.log_softmax(torch.ones(2, 32), -1))
    with pytest.raises(ValueError, match='词表'):
        evaluate_kl(model, [smaller_vocab], layer=2, vector=torch.ones(D))
    assert all(not layer._forward_hooks for layer in get_layers(model))


@pytest.mark.parametrize('basis', [torch.eye(5, D), None])
def test_opt_in_diagnostics_measure_full_fit_and_actual_update_without_changing_fit(model, basis):
    fit = [item(model, 'fit1', torch.ones(D) * .02), item(model, 'fit2', torch.arange(D).float() * .001)]
    heldout = [item(model, 'heldout')]
    size = D if basis is None else 5
    init = torch.ones(size) * .001
    initial_vector = 2 * (init if basis is None else init @ basis)
    kwargs = dict(basis=basis, scale=2., layer=2, steps=1, probes_per_step=1, lr=.001, init=init)
    plain = fit_and_evaluate(model, fit, heldout, **kwargs)
    measured = fit_and_evaluate(model, fit, heldout, **kwargs, diagnostics=True)
    assert 'diagnostics' not in plain
    assert torch.equal(plain['coordinates'], measured['coordinates'])
    assert plain['final'] == measured['final']
    d = measured['diagnostics']
    assert d['fit_final'] == evaluate_kl(model, fit, layer=2, vector=measured['vector'])
    assert d['fit_initial'] == evaluate_kl(model, fit, layer=2, vector=initial_vector)
    assert d['fit_baseline'] == evaluate_kl(model, fit, layer=2)
    assert d['fit_final']['n_probes'] == 2
    expected = float((measured['vector'] - initial_vector).norm())
    assert d['step_metrics'][0]['vector_update_norm'] == pytest.approx(expected)
    assert d['step_metrics'][0]['vector_update_norm_over_scale'] == pytest.approx(expected / 2)
    assert d['parameter_count'] == size and d['optimizer']['lr'] == .001
    assert all(p.grad is None for p in model.parameters())
