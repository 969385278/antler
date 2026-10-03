"""C5延迟选项离线回归：微型冻结模型、身份锁定、真实decode和失败留档。"""
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch
from transformers import BatchEncoding

from test_core import D, model
from antler.core.model import get_layers
from antler.eval import delayed_capability as dc


class Tokenizer:
    all_special_ids = [0, 1, 2]
    pad_token_id, bos_token_id = 0, 1

    def encode(self, text, **kwargs):
        if text == '\n': return [8]
        if text in 'ABCD' and len(text) == 1: return [10 + 'ABCD'.index(text)]
        if text.startswith('\n') and text[1:] in 'ABCD': return [8, 10 + 'ABCD'.index(text[1:])]
        raise ValueError('测试分词器收到未知文本')

    def decode(self, ids, **kwargs):
        return ''.join('\n' if i == 8 else 'ABCD'[i-10] for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return messages[0]['content']

    def __call__(self, prompts, **kwargs):
        assert self.padding_side == 'left'
        sequences = [list(range(20, 20 + int(prompt.split('\n')[0]))) for prompt in prompts]
        width = max(map(len, sequences))
        return BatchEncoding({'input_ids': torch.tensor([[0]*(width-len(row))+row for row in sequences]),
                              'attention_mask': torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in sequences])})


def rows():
    return [{'source_index': i, 'subject': '学科', 'question': str(width),
             'choices': ['选项甲', '选项乙', '选项丙', '选项丁'], 'answer': i % 4}
            for i, width in enumerate((2, 5, 1))]


def assert_clean(model):
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in get_layers(model))


def test_fixed_token_contract_and_canonical_prompt():
    contract = dc.token_contract(Tokenizer(), {2})
    assert contract['prefix_token_id'] == 8 and contract['answer_token_ids'] == [10, 11, 12, 13]
    prompt = dc.canonical_prompt(Tokenizer(), rows()[0])
    assert prompt == ('2\n\nA. 选项甲\nB. 选项乙\nC. 选项丙\nD. 选项丁'
                      '\n\nAnswer with a single letter: A, B, C, or D.')


@pytest.mark.parametrize('change', ['multiple', 'special', 'eos', 'letter', 'boundary', 'decode'])
def test_invalid_newline_contract_fails_closed(change):
    tok, eos = Tokenizer(), {2}
    old_encode = tok.encode
    if change == 'multiple': tok.encode = lambda s, **kw: [8, 9] if s == '\n' else old_encode(s, **kw)
    if change == 'special': tok.all_special_ids = [0, 1, 2, 8]
    if change == 'eos': eos.add(8)
    if change == 'letter': tok.encode = lambda s, **kw: [10] if s == '\n' else old_encode(s, **kw)
    if change == 'boundary': tok.encode = lambda s, **kw: [14] if s == '\nA' else old_encode(s, **kw)
    if change == 'decode': tok.decode = lambda *a, **kw: ' '
    with pytest.raises(ValueError):
        dc.token_contract(tok, eos)


def test_micro_model_nonzero_changes_only_second_step_and_zero_matches(model):
    baseline = dc.delayed_mmlu(model, Tokenizer(), rows(), batch_size=2)
    zero = dc.delayed_mmlu(model, Tokenizer(), rows(), steer={1: torch.zeros(D)}, batch_size=2,
                           reference_first=baseline['samples'])
    injected = dc.delayed_mmlu(model, Tokenizer(), rows(), steer={1: torch.linspace(-2, 2, D)},
                               batch_size=2, reference_first=baseline['samples'])
    for first, empty, changed in zip(baseline['samples'], zero['samples'], injected['samples']):
        assert first['prefill'] == empty['prefill'] == changed['prefill']
        assert first['delayed'] == empty['delayed']
        assert [t['applied'] for t in first['phase_trace']] == [False, False]
        assert [t['applied'] for t in changed['phase_trace']] == [False, True]
        assert changed['nonzero_intervention'] and changed['vectors']['1']['norm'] > 0
        assert changed['prefix_token_id'] == 8
        assert changed['decode_cache_position'] == changed['padded_prompt_length']
        assert changed['decode_position_id'] == changed['prompt_length']
    assert any(a['delayed']['choice_logits'] != b['delayed']['choice_logits']
               for a, b in zip(baseline['samples'], injected['samples']))
    assert baseline['formal_gate_eligible'] is False
    assert_clean(model)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_left_padding_batch_matches_individual_and_fresh_cache(model, monkeypatch):
    calls, caches = [], []
    original = model.forward
    def capture(*args, **kwargs):
        cache = kwargs['past_key_values']
        calls.append((cache, int(cache.get_seq_length()), kwargs['attention_mask'].clone(),
                      kwargs['position_ids'].clone(), kwargs['cache_position'].clone()))
        caches.append(cache)
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', capture)
    batch = dc.delayed_mmlu(model, Tokenizer(), rows(), batch_size=2)
    assert [call[1] for call in calls] == [0, 5, 0, 1]
    assert calls[0][0] is calls[1][0] and calls[2][0] is calls[3][0]
    assert calls[0][0] is not calls[2][0]
    assert calls[0][3].tolist() == [[0, 0, 0, 0, 1], [0, 1, 2, 3, 4]]
    assert calls[1][3].tolist() == [[2], [5]]
    assert calls[1][4].tolist() == [5]
    assert calls[1][2].tolist() == [[0, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    individual = dc.delayed_mmlu(model, Tokenizer(), rows(), batch_size=1)
    for left, right in zip(batch['samples'], individual['samples']):
        for step in ('prefill', 'delayed'):
            assert torch.allclose(torch.tensor(left[step]['choice_logits']),
                                  torch.tensor(right[step]['choice_logits']), atol=1e-6, rtol=1e-6)
    assert len({id(cache) for cache in caches}) == 5
    assert_clean(model)


def test_first_step_reference_mismatch_and_forward_error_remove_hooks(model, monkeypatch):
    baseline = dc.delayed_mmlu(model, Tokenizer(), rows(), batch_size=2)
    baseline['samples'][0]['prefill']['choice_logits'][0] += 1
    with pytest.raises(ValueError, match='首步'):
        dc.delayed_mmlu(model, Tokenizer(), rows(), steer={1: torch.ones(D)}, batch_size=2,
                        reference_first=baseline['samples'])
    assert_clean(model)
    original = model.forward
    def fail_decode(*args, **kwargs):
        if int(kwargs['past_key_values'].get_seq_length()) > 0:
            raise RuntimeError('合成decode失败')
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', fail_decode)
    with pytest.raises(RuntimeError, match='合成'):
        dc.delayed_mmlu(model, Tokenizer(), rows(), steer={1: torch.ones(D)})
    assert_clean(model)


@pytest.mark.parametrize('change', ['nan_letter', 'nan_other', 'inf', 'wrong_batch', 'integer'])
def test_invalid_full_vocab_logits_rejected(change):
    logits = torch.zeros(2, 20)
    if change == 'nan_letter': logits[0, 10] = float('nan')
    if change == 'nan_other': logits[0, 19] = float('nan')
    if change == 'inf': logits[0, 10] = float('inf')
    if change == 'wrong_batch': logits = logits[:1]
    if change == 'integer': logits = logits.long()
    with pytest.raises(ValueError, match='logit'):
        dc.logit_records(logits, [10, 11, 12, 13], [0, 1])


def small_source():
    data = [{'subject': f'学科{i%2}', 'question': str(i), 'choices': ['a', 'b', 'c', 'd'], 'answer': i%4}
            for i in range(8)]
    indices = [{'source_index': i, 'subject': data[i]['subject']} for i in (5, 0, 3, 6)]
    reference = {'n': 4, 'accuracy': 1., 'samples': [{**entry, 'answer': data[entry['source_index']]['answer'],
        'prediction': data[entry['source_index']]['answer'], 'correct': True} for entry in indices]}
    return data, indices, reference


def test_existing_indices_preserve_order_and_gold_content():
    data, indices, reference = small_source()
    selected = dc.existing_sample(data, indices, reference, total=8, n=4, subjects=2)
    assert [r['source_index'] for r in selected] == [5, 0, 3, 6]
    assert [r['question'] for r in selected] == ['5', '0', '3', '6']


@pytest.mark.parametrize('change', ['duplicate', 'order', 'subject', 'answer', 'count', 'accuracy', 'bool_index'])
def test_source_identity_failure_is_not_dropped_or_replaced(change):
    data, indices, reference = small_source()
    if change == 'duplicate': indices[-1], reference['samples'][-1] = indices[0], reference['samples'][0]
    if change == 'order': reference['samples'].reverse()
    if change == 'subject': data[5]['subject'] = '别的学科'
    if change == 'answer': data[5]['answer'] = 3
    if change == 'count': data.pop()
    if change == 'accuracy': reference['accuracy'] = .5
    if change == 'bool_index': indices[0]['source_index'] = True
    with pytest.raises(ValueError):
        dc.existing_sample(data, indices, reference, total=8, n=4, subjects=2)


def scored(index, prediction, answer=0):
    logits = [-1.] * 4
    logits[prediction] = 1.
    choice = {'prediction': prediction, 'correct': prediction == answer, 'choice_logits': logits}
    return {'source_index': index, 'subject': '学科', 'answer': answer,
            'prefill': copy.deepcopy(choice), 'delayed': copy.deepcopy(choice)}


def test_paired_statistics_keep_original_questions_and_flip_direction():
    left = [scored(i, p) for i, p in enumerate((0, 0, 1, 1))]
    right = [scored(i, p) for i, p in enumerate((1, 1, 0, 1))]
    result = dc.paired_comparison(left, right)
    assert result['correct_to_wrong'] == 2 and result['wrong_to_correct'] == 1
    assert result['prediction_flips'] == 3 and result['loss_percentage_points'] == 25
    assert result['right_minus_left_percentage_points']['mean'] == -25
    assert result['right_minus_left_percentage_points']['n_questions'] == 4
    assert result['formal_gate_eligible'] is False
    right[0]['delayed'].pop('choice_logits')
    with pytest.raises(ValueError, match='缺少'):
        dc.paired_comparison(left, right)
    with pytest.raises(ValueError):
        dc.paired_comparison(left, list(reversed(left)))


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e0_delayed_capability')


@pytest.fixture
def run_inputs(script, tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    data_dir = root / 'data'
    data_dir.mkdir(parents=True)
    parquet = data_dir / 'test-00000-of-00001.parquet'
    parquet.write_bytes(b'local-fixture')
    source = root / 'reference'
    source.mkdir()
    data = [{'subject': f's{i%57}', 'question': str(i), 'choices': ['a', 'b', 'c', 'd'], 'answer': i%4}
            for i in range(14042)]
    indices = [{'source_index': i, 'subject': data[i]['subject']} for i in range(1400)]
    baseline = {'n': 1400, 'accuracy': 1., 'samples': [{**entry, 'answer': i%4, 'prediction': i%4, 'correct': True}
                                                    for i, entry in enumerate(indices)]}
    script.write_json(source / 'sample_indices.json', indices)
    script.write_json(source / 'baseline.json', baseline)
    hashes = {'sample_indices': script.sha256(source / 'sample_indices.json'),
              'reference_baseline': script.sha256(source / 'baseline.json'), 'parquet': script.sha256(parquet)}
    script.write_json(source / 'run_meta.json', {'status': 'completed', 'model': 'mock-model',
        'scoring': 'zero_shot_chat_ABCD_logits', 'n_questions': 1400,
        'artifact_hashes': {'sample_indices.json': hashes['sample_indices'], 'baseline.json': hashes['reference_baseline']},
        'data_hashes': {str(parquet): hashes['parquet']}})
    vectors = root / 'paired'
    (vectors / 'vectors').mkdir(parents=True)
    torch.save({'U': torch.eye(5, 8).repeat(22, 1, 1), 's': torch.ones(22), 'dims': script.DIMS},
               vectors / 'directions.pt')
    for poles in script.BIG_FIVE.values():
        for pole in poles:
            torch.save(torch.ones(22, 8), vectors / 'vectors' / f'{pole}.pt')
    (root / 'antler').mkdir()
    (root / 'antler' / 'synthetic.py').write_text('# 合成文件\n', encoding='utf-8')
    (root / 'experiments').mkdir()
    for name in ('_common.py', 'e0_diagnostics.py', 'e0_extract_coherence.py', 'e0_refilter.py'):
        (root / 'experiments' / name).write_text('# 合成依赖\n', encoding='utf-8')
    config = root / 'config.yaml'
    config.write_text('model: mock-model\n', encoding='utf-8')
    (root / 'mock-model').mkdir()
    (root / 'mock-model' / 'config.json').write_text('{}\n', encoding='utf-8')
    cfg = {'model': 'mock-model', 'out_path': root / 'out', 'api_key': 'OFFLINE_SECRET_SENTINEL'}
    args = SimpleNamespace(config=str(config), vectors_dir=str(vectors),
                           sample_indices=str(source / 'sample_indices.json'),
                           reference_baseline=str(source / 'baseline.json'), data=str(data_dir))
    monkeypatch.setattr(script, 'ROOT', root)
    monkeypatch.setattr(script, 'SOURCE_SHA256', hashes)
    monkeypatch.setattr(script, 'VECTOR_SHA256', {name: script.sha256(vectors / name) for name in script.VECTOR_SHA256})
    monkeypatch.setattr(script.pq, 'read_table', lambda path: SimpleNamespace(to_pylist=lambda: copy.deepcopy(data)))
    monkeypatch.setattr(script, 'setup', lambda *a, **kw: (cfg, None, None, None))
    monkeypatch.setattr(script, '_git', lambda *a: 'offline' if a[0] == 'rev-parse' else '')
    monkeypatch.setattr(script.sys, 'argv', ['delayed', '--config', str(config)])
    return args, cfg, source, parquet


def test_fixed_loader_checks_full_counts_and_registered_hashes(script, run_inputs):
    args, cfg, source, parquet = run_inputs
    selected, baseline, inputs = script.fixed_inputs(args, cfg)
    assert len(selected) == 1400 and len({r['subject'] for r in selected}) == 57
    assert baseline['n'] == 1400 and len(inputs) == 4
    parquet.write_bytes(b'changed')
    with pytest.raises(ValueError, match='哈希'):
        script.fixed_inputs(args, cfg)


@pytest.mark.parametrize('failure', [None, 'runtime_secret', 'vector_hash'])
def test_mock_run_is_independent_hashes_artifacts_and_saves_failure(script, run_inputs, monkeypatch, failure):
    args, cfg, source, parquet = run_inputs
    model_calls, measurement_calls = [], []
    model_stub = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=22, hidden_size=8),
                                  generation_config=SimpleNamespace(to_dict=lambda: {'eos_token_id': 2}))
    def fake_model(*a):
        model_calls.append(a)
        return model_stub, object()
    monkeypatch.setattr(script, 'load_model', fake_model)
    def fake_measure(model, tok, selected, **kwargs):
        measurement_calls.append(kwargs)
        if failure == 'runtime_secret':
            raise RuntimeError('Authorization: OFFLINE_SECRET_SENTINEL')
        samples = []
        for row in selected:
            result = scored(row['source_index'], row['answer'], row['answer'])
            result['subject'] = row['subject']
            samples.append(result)
        if len(measurement_calls) > 1:
            assert kwargs['reference_first'] == samples
            assert set(kwargs['steer']) == {21} and kwargs['steer'][21].norm() > 0
        else:
            assert kwargs['steer'] is None and kwargs['reference_first'] is None
        assert kwargs['batch_size'] == 16
        kwargs['on_batch'](samples)
        return {'n': len(samples), 'samples': samples, 'formal_gate_eligible': False}
    monkeypatch.setattr(script, 'delayed_mmlu', fake_measure)
    before = {p: p.read_bytes() for p in source.iterdir() if p.is_file()}
    if failure == 'vector_hash':
        (Path(args.vectors_dir) / 'vectors' / 'nervous.pt').write_bytes(b'changed')
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            script.run(args)
        out = next((cfg['out_path'] / 'delayed_capability').iterdir())
    else:
        out = script.run(args)
    meta_text = (out / 'run_meta.json').read_text(encoding='utf-8')
    meta = json.loads(meta_text)
    assert meta['status'] == ('failed' if failure else 'completed')
    assert meta['end_time'] and meta['formal_gate_eligible'] is False and meta['judge'] is None
    assert meta['generation_performed'] is False and 'OFFLINE_SECRET_SENTINEL' not in meta_text
    assert all(p.read_bytes() == raw for p, raw in before.items())
    if failure == 'vector_hash':
        assert not model_calls and not measurement_calls
    elif failure == 'runtime_secret':
        assert len(model_calls) == len(measurement_calls) == 1
    else:
        assert len(measurement_calls) == 3 and meta['completed_predictions'] == 4200
        assessment = script.read_json(out / 'assessment.json')
        assert assessment['formal_gate_eligible'] is False and set(assessment['decode_effects']) == {'C', 'N'}
        assert assessment['zero_prefix_effect']['n'] == 1400
    if failure:
        assert not (out / 'assessment.json').exists()
    hashes = script.read_json(out / 'artifact_hashes.json')
    assert all(script.sha256(out / name) == digest for name, digest in hashes.items())
    with zipfile.ZipFile(out / 'results.zip') as archive:
        assert archive.testzip() is None and 'run_meta.json' in archive.namelist()
        assert all(b'OFFLINE_SECRET_SENTINEL' not in archive.read(name) for name in archive.namelist())
