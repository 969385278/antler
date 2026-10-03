"""C19：用冻结的随机微型 Qwen 验证注入阶段，不下载模型或运行 GPU。"""
from copy import deepcopy
import importlib
import json

import pytest
import torch
from transformers.cache_utils import DynamicCache

from antler.core.generate import generate
from antler.core.hooks import Recorder, Steer
from antler.core.model import get_layers
from test_core import D, model
from test_generate import Tokenizer


def assert_detached(model):
    assert not model._forward_pre_hooks and not model._forward_hooks
    assert all(not layer._forward_hooks for layer in get_layers(model))


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('width', [1, 4])
def test_cache_metadata_identifies_single_and_multi_token_prompts(model, phase, width):
    prompt = torch.full((1, width), 7, dtype=torch.long)
    empty_cache = DynamicCache(config=model.config)
    with torch.no_grad(), Steer(model, {1: torch.zeros(D)}, phase=phase) as steering:
        first = model(input_ids=prompt, past_key_values=empty_cache, use_cache=True)
        # 解码调用也可一次处理两个 token；不能根据本次输入长度判定阶段。
        model(input_ids=torch.tensor([[8, 9]]), past_key_values=first.past_key_values, use_cache=True)
    assert [(r['phase'], r['cached_tokens']) for r in steering.phase_trace] == [('prefill', 0), ('decode', width)]
    assert [r['applied'] for r in steering.phase_trace] == [phase == 'prefill', phase == 'decode']
    assert_detached(model)


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('stage', ['prefill', 'decode'])
def test_recorder_observes_only_active_stage_vector(model, phase, stage):
    ids = torch.tensor([[7, 8]])
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([[4, 5, 6]]), use_cache=True).past_key_values if stage == 'decode' else None
        with Recorder(model, [2]) as baseline:
            model(input_ids=ids, past_key_values=deepcopy(cache), use_cache=True)
        vector = torch.linspace(-.2, .2, D)
        with Steer(model, {2: vector}, phase=phase) as steering, Recorder(model, [2]) as changed:
            model(input_ids=ids, past_key_values=deepcopy(cache), use_cache=True)
    expected = vector.expand_as(baseline.acts[2]) if phase == stage else torch.zeros_like(baseline.acts[2])
    assert torch.allclose(changed.acts[2]-baseline.acts[2], expected, atol=1e-6)
    assert steering.phase_trace[0]['phase'] == stage
    assert_detached(model)


def test_default_and_explicit_all_preserve_logits_generation_and_gradient(model, monkeypatch):
    monkeypatch.setattr(model.generation_config, 'eos_token_id', None)
    ids = torch.tensor([[5, 6]])
    vector = torch.linspace(-.1, .1, D, requires_grad=True)
    with Steer(model, {2: vector}) as default:
        first = model(input_ids=ids, use_cache=False).logits
    with Steer(model, {2: vector}, phase='all') as explicit:
        second = model(input_ids=ids, use_cache=False).logits
    assert torch.equal(first, second)
    first.square().mean().backward()
    assert vector.grad is not None and vector.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    assert default.phase_trace == explicit.phase_trace == []
    kwargs = dict(steer={2: vector.detach()}, max_new_tokens=3, temperature=1., seed=17,
                  return_details=True, return_token_ids=True)
    trace = []
    assert generate(model, Tokenizer(), ['5 6'], **kwargs) == generate(
        model, Tokenizer(), ['5 6'], steer_phase='all', phase_trace_out=trace, **kwargs)
    assert trace == [{'batch_start': 0, 'trace': []}]
    assert_detached(model)


@pytest.mark.parametrize('phase', ['all', 'prefill', 'decode'])
def test_zero_vector_matches_unsteered_tokens_and_details(model, monkeypatch, phase):
    monkeypatch.setattr(model.generation_config, 'eos_token_id', None)
    prompts = ['4', '5 6', '7']
    kwargs = dict(max_new_tokens=3, temperature=1., batch_size=2, seed=23,
                  return_details=True, return_token_ids=True)
    baseline = generate(model, Tokenizer(), prompts, **kwargs)
    result = generate(model, Tokenizer(), prompts, steer={2: torch.zeros(D)}, steer_phase=phase, **kwargs)
    assert result == baseline
    assert all(set(row) == {'text', 'n_generated_tokens', 'finish_reason', 'token_ids'} for row in result)
    assert_detached(model)


def test_decode_cannot_change_first_generated_token(model, monkeypatch):
    monkeypatch.setattr(model.generation_config, 'eos_token_id', None)
    kwargs = dict(max_new_tokens=1, temperature=1., seed=19, return_details=True, return_token_ids=True)
    baseline = generate(model, Tokenizer(), ['4', '5 6'], **kwargs)
    result = generate(model, Tokenizer(), ['4', '5 6'], steer={2: torch.linspace(-5, 5, D)},
                      steer_phase='decode', **kwargs)
    assert result == baseline
    assert_detached(model)


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_each_batch_uses_fresh_phase_context(model, monkeypatch, phase):
    module = importlib.import_module('antler.core.generate')
    monkeypatch.setattr(model.generation_config, 'eos_token_id', None)
    instances = []

    class TracedSteer(Steer):
        def __enter__(self):
            instances.append(self)
            return super().__enter__()

    monkeypatch.setattr(module, 'Steer', TracedSteer)
    trace = []
    result = module.generate(model, Tokenizer(), ['4', '5 6', '7', '8 9', '10'],
                             steer={2: torch.zeros(D)}, steer_phase=phase, batch_size=2,
                             max_new_tokens=3, temperature=0, return_details=True, phase_trace_out=trace)
    assert len(instances) == 3 and len(result) == 5
    assert [entry['batch_start'] for entry in trace] == [0, 2, 4]
    assert [entry['trace'] for entry in trace] == [context.phase_trace for context in instances]
    assert json.loads(json.dumps(trace, allow_nan=False)) == trace
    for steering in instances:
        assert [entry['phase'] for entry in steering.phase_trace] == ['prefill', 'decode', 'decode']
        assert steering.phase_trace[0]['cached_tokens'] == 0
        assert steering._active_phase is None and not steering._handles
    assert_detached(model)


@pytest.mark.parametrize('phase', ['prefill', 'decode'])
def test_no_cache_rejected_but_all_remains_supported(model, monkeypatch, phase):
    monkeypatch.setattr(model.generation_config, 'use_cache', False)
    with pytest.raises(ValueError, match='use_cache=True'):
        generate(model, Tokenizer(), ['4'], steer_phase=phase, max_new_tokens=1)
    with pytest.raises(ValueError, match='use_cache=True'):
        with Steer(model, {1: torch.zeros(D)}, phase=phase):
            model(input_ids=torch.tensor([[4]]), use_cache=False)
    result = generate(model, Tokenizer(), ['4'], steer={1: torch.zeros(D)}, steer_phase='all',
                      max_new_tokens=1, temperature=0)
    assert len(result) == 1
    assert_detached(model)


@pytest.mark.parametrize('phase', ['bad', '', None, 1])
def test_unknown_phase_is_rejected_before_generation(model, phase):
    with pytest.raises(ValueError, match='phase'):
        Steer(model, {}, phase=phase)
    with pytest.raises(ValueError, match='steer_phase'):
        generate(None, None, [], steer_phase=phase)


@pytest.mark.parametrize('trace', [{}, (), '', True])
def test_invalid_trace_output_rejected_before_generation(trace):
    with pytest.raises(ValueError, match='phase_trace_out'):
        generate(None, None, [], phase_trace_out=trace)


@pytest.mark.parametrize('position', [torch.tensor([1]), torch.tensor([0, 2]), torch.tensor([]),
                                     torch.tensor([[0]]), torch.tensor([0.]), torch.tensor([False]), [0]])
def test_invalid_cache_position_rejected_and_hooks_removed(model, position):
    with pytest.raises(ValueError, match='cache_position'):
        with Steer(model, {1: torch.zeros(D)}, phase='prefill'):
            model(input_ids=torch.tensor([[4]]), use_cache=True, cache_position=position)
    assert_detached(model)


def test_explicit_cache_positions_match_empty_and_nonempty_cache(model):
    with torch.no_grad(), Steer(model, {1: torch.zeros(D)}, phase='decode') as steering:
        first = model(input_ids=torch.tensor([[4, 5]]), use_cache=True, cache_position=torch.tensor([0, 1]))
        model(input_ids=torch.tensor([[6]]), use_cache=True, past_key_values=first.past_key_values,
              cache_position=torch.tensor([2]))
    assert [(r['phase'], r['cache_position_start']) for r in steering.phase_trace] == [('prefill', 0), ('decode', 2)]
    assert_detached(model)


@pytest.mark.parametrize('initial_cached', [False, True])
def test_cache_reset_requires_a_new_context(model, initial_cached):
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([[4]]), use_cache=True).past_key_values if initial_cached else None
        with Steer(model, {1: torch.zeros(D)}, phase='prefill') as steering:
            model(input_ids=torch.tensor([[5]]), past_key_values=cache, use_cache=True)
            assert steering.phase_trace[0]['phase'] == ('decode' if initial_cached else 'prefill')
            with pytest.raises(ValueError, match='清空缓存'):
                model(input_ids=torch.tensor([[6]]), use_cache=True)
            assert steering._active_phase is None
        # 退出后同一实例也可以开启新的、状态已重置的上下文。
        with steering:
            model(input_ids=torch.tensor([[7]]), use_cache=True)
            assert len(steering.phase_trace) == 1 and steering.phase_trace[0]['phase'] == 'prefill'
    assert_detached(model)


@pytest.mark.parametrize('cache', [object(), (), []])
def test_unknown_cache_type_fails_explicitly(model, cache):
    with pytest.raises(ValueError, match='缓存类型'):
        with Steer(model, {1: torch.zeros(D)}, phase='decode'):
            model(input_ids=torch.tensor([[4]]), past_key_values=cache, use_cache=True)
    assert_detached(model)


@pytest.mark.parametrize('length', [True, -1, 1.5, float('nan'), torch.tensor([1, 2])])
def test_invalid_cache_length_fails_explicitly(model, monkeypatch, length):
    cache = DynamicCache(config=model.config)
    monkeypatch.setattr(cache, 'get_seq_length', lambda: length)
    with pytest.raises(ValueError, match='非负整数标量'):
        with Steer(model, {1: torch.zeros(D)}, phase='decode'):
            model(input_ids=torch.tensor([[4]]), past_key_values=cache, use_cache=True)
    assert_detached(model)


@pytest.mark.parametrize('phase', ['all', 'prefill', 'decode'])
def test_partial_registration_failure_keeps_unrelated_hooks(model, phase):
    with Recorder(model, [0]):
        with pytest.raises(IndexError):
            with Steer(model, {0: torch.ones(D), 999: torch.ones(D)}, phase=phase):
                pytest.fail('层号无效时不应进入上下文')
        assert len(get_layers(model)[0]._forward_hooks) == 1
        assert all(not layer._forward_hooks for layer in get_layers(model)[1:])
        assert not model._forward_hooks and not model._forward_pre_hooks
    assert_detached(model)


def test_forward_failure_clears_phase_inside_context_and_detaches_on_exit(model, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('模拟前向异常')

    monkeypatch.setattr(get_layers(model)[2], 'forward', fail)
    with Steer(model, {1: torch.ones(D)}, phase='prefill') as steering:
        with pytest.raises(RuntimeError, match='模拟前向异常'):
            model(input_ids=torch.tensor([[4]]), use_cache=True)
        assert steering._active_phase is None
    assert_detached(model)


def test_reentering_same_context_is_rejected_without_losing_outer_hooks(model):
    steering = Steer(model, {1: torch.zeros(D)}, phase='prefill')
    with steering:
        with pytest.raises(ValueError, match='重复进入'):
            with steering:
                pytest.fail('不能重复进入同一个上下文')
        assert steering._handles
    assert_detached(model)


@pytest.mark.parametrize('name,value', [('prefill_chunk_size', 2), ('cache_implementation', 'static'),
                                       ('num_beams', 2), ('prompt_lookup_num_tokens', 2),
                                       ('assistant_early_exit', 1)])
def test_unsupported_generation_modes_rejected(model, monkeypatch, name, value):
    monkeypatch.setattr(model.generation_config, name, value, raising=False)
    with pytest.raises(ValueError, match='阶段注入'):
        generate(model, Tokenizer(), ['4'], steer_phase='decode', max_new_tokens=1)
    assert_detached(model)
