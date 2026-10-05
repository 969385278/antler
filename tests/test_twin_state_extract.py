"""固定提取诊断的CPU测试：非正交幅度、反馈交互、预算与钩子清理。"""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from antler.memory.twin_state import analyze, select_probes
from e1_twin_state_extract import collect
from test_core import model  # 复用CPU随机微型模型fixture。


def test_probe_selection_disjoint_and_not_mutating_pool():
    pool = [{'question': f'题{i}', 'index': i} for i in range(100)]
    probes = select_probes(pool)
    assert len(probes) == 60 and probes == select_probes(pool)
    assert {p['question'] for p in probes[:30]}.isdisjoint(p['question'] for p in probes[30:])
    assert pool[0]['index'] == 0
    with pytest.raises(ValueError):
        select_probes(pool[:59])
    with pytest.raises(ValueError):
        select_probes(pool + [pool[0]])


def test_default_preflight_never_loads_model_or_judge(monkeypatch):
    from types import SimpleNamespace
    import e1_twin_state_extract as entry
    calls = []
    def setup(args, **kwargs):
        calls.append(kwargs)
        return {}, None, None, None
    monkeypatch.setattr(entry, 'setup', setup)
    monkeypatch.setattr(entry, 'validate', lambda _: ([], [], None, None))
    monkeypatch.setattr(entry, 'load_model', lambda *a, **kw: pytest.fail('预检不应加载模型'))
    result = entry.run(SimpleNamespace(execute=False))
    assert calls == [{'need_model': False, 'need_judge': False}]
    assert result['model_loaded'] is False and result['judge_loaded'] is False


def test_nonorthogonal_norm_and_neutral_mean():
    basis = torch.eye(5, 7)
    basis[1] = torch.tensor([.6, .8, 0, 0, 0, 0, 0])
    acts = torch.zeros(9, 60, 7)
    acts[4:, :, 6] = torch.arange(5.).reshape(5, 1)
    v = torch.tensor([1., 1., 0, 0, 0]) @ basis
    acts[:4] = v + torch.tensor([0., 0, 0, 0, 0, 0, 4.])
    result = analyze(acts, basis, 2.)['groups']['all60']['a-1_r-1']
    assert result['dual_coordinates'][:2] == pytest.approx([.5, .5])
    assert result['residual_norm_s'] == pytest.approx(1.)
    assert result['reconstructed_norm_s'] == pytest.approx(float(v.norm()) / 2)
    assert result['full_norm_physical'] == pytest.approx(float(torch.cat((v[:6], torch.tensor([2.]))).norm()))


def test_valence_alone_cannot_create_interaction():
    acts = torch.zeros(9, 60, 8)
    acts[1, :, 2] = 2.; acts[3, :, 2] = 2.
    result = analyze(acts, torch.eye(5, 8), 1.)
    assert result['groups']['all60']['interaction']['full_norm_s'] == 0
    assert result['groups']['all60']['valence']['full_norm_s'] == 2
    assert result['probe_stability']['interaction']['full_cosine'] is None
    assert result['probe_stability']['interaction']['second_over_first_norm'] is None


def test_opposite_feedback_and_probe_instability_are_preserved():
    acts = torch.zeros(9, 60, 8)
    acts[1, :, 2] = -2.; acts[3, :, 2] = 2.
    acts[:4, 30:] *= -1
    result = analyze(acts, torch.eye(5, 8), 1.)
    assert result['groups']['first30']['interaction']['axis_projection'][2] == 2
    assert result['groups']['second30']['interaction']['axis_projection'][2] == -2
    assert result['probe_stability']['interaction']['full_cosine'] == -1
    assert result['groups']['all60']['interaction']['full_norm_s'] == 0
    assert result['formal_e1_pass'] is False and result['behavior_validated'] is False


@pytest.mark.parametrize('problem', ['missing', 'nan', 'rank', 'scale'])
def test_invalid_input_stops(problem):
    acts, basis, scale = torch.zeros(9, 60, 8), torch.eye(5, 8), 1.
    if problem == 'missing':
        acts = acts[:8]
    elif problem == 'nan':
        acts[0, 0, 0] = float('nan')
    elif problem == 'rank':
        basis[1] = basis[0]
    else:
        scale = 0
    with pytest.raises(ValueError):
        analyze(acts, basis, scale)


class TinyTokenizer:
    padding_side = 'left'

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        return '\n'.join(m['content'] for m in messages)

    def __call__(self, text, return_tensors=None, padding=False, add_special_tokens=False):
        from transformers import BatchEncoding
        encode = lambda t: [1, 2, 3, len(t) % 60 + 4]
        if isinstance(text, str):
            return {'input_ids': encode(text)}
        ids = torch.tensor([encode(t) for t in text])
        return BatchEncoding({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})


def test_actual_tiny_model_forward_budget_and_no_weight_changes(model):
    episodes = [{'action_sign': a, 'outcome_sign': r, 'text': f'行动{a}反馈{r}'}
                for a, r in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    probes = [{'question': f'问题{i}'} for i in range(60)]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    shapes = []
    handle = model.register_forward_pre_hook(lambda _, args, kwargs: shapes.append(kwargs['input_ids'].shape), with_kwargs=True)
    try:
        acts, prompts = collect(model, TinyTokenizer(), episodes, probes, layer=1)
    finally:
        handle.remove()
    assert len(shapes) == 72 and sum(s[0] for s in shapes) == 540
    assert acts.shape == (9, 60, 32) and len(prompts) == 540
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not block._forward_hooks for block in model.model.layers)


def test_collection_interruption_preserves_prefix_and_removes_hooks(model):
    episodes = [{'action_sign': a, 'outcome_sign': r, 'text': f'{a}/{r}'}
                for a, r in ((-1, -1), (-1, 1), (1, -1), (1, 1))]
    prefix = []
    def stop(n, groups, prompts):
        prefix.append((n, len(groups), len(prompts)))
        raise RuntimeError('模拟中断')
    with pytest.raises(RuntimeError):
        collect(model, TinyTokenizer(), episodes, [{'question': str(i)} for i in range(60)], layer=1, progress=stop)
    assert prefix == [(1, 1, 60)]
    assert all(not block._forward_hooks for block in model.model.layers)
