"""C10：D1数据边界与盲标防错测试，不调用模型。"""
from copy import deepcopy

import pytest

from antler.memory.episode_data import blind_packet, merge_and_split, validate_candidates


def candidate(identity='e1', intended=None):
    return dict(id=identity, intended={'O': 1} if intended is None else intended,
                text=f'A user asked me to help with a small project. I offered several steps and we tried them together over the next week. They sent me their results and described what happened to the project after our conversation ended. The record number was {identity}.',
                generator='generator-v1', generator_vendor='vendor-x')


def bundles(rows):
    public, mapping = blind_packet(rows)
    by_id = {r['id']: r for r in rows}
    a = dict(model='model-a', vendor='vendor-a', input_sha256='hash',
             records=[{'id': r['id'], 'labels': by_id[mapping[r['id']]]['intended']} for r in public])
    b = deepcopy(a); b.update(model='model-b', vendor='vendor-b')
    return public, mapping, a, b


def test_blind_packet_hides_intended_and_is_reproducible():
    rows = [candidate('e1'), candidate('e2', {'C': -1})]
    p, m = blind_packet(rows)
    assert all(set(r) == {'id', 'text'} and r['id'].startswith('D1B') for r in p)
    assert blind_packet(list(reversed(rows))) == (p, m)


@pytest.mark.parametrize('change', [dict(text='I was careful.'), dict(generator_vendor='DeepSeek'),
                                   dict(intended={'O': True}), dict(group_id='same-family')])
def test_invalid_candidates_rejected(change):
    r = candidate(); r.update(change)
    with pytest.raises(ValueError):
        validate_candidates([r])


def test_disagreement_preserved_and_no_premature_split():
    rows = [candidate('e1'), candidate('e2', {'C': -1})]
    p, m, a, b = bundles(rows)
    b['records'][0]['labels'] = {'N': 1}
    report = merge_and_split(rows, p, m, a, b, 'hash')
    assert len(report['accepted']) == len(report['rejected']) == 1
    assert not report['quota_complete'] and not report['formal_ready']
    assert 'split' not in report['accepted'][0]


def test_missing_duplicate_wrong_hash_or_same_model_fail():
    rows = [candidate()]
    for mutation in ('missing', 'hash', 'same'):
        p, m, a, b = bundles(rows)
        if mutation == 'missing': b['records'] = []
        elif mutation == 'hash': b['input_sha256'] = 'wrong'
        else: b = deepcopy(a)
        with pytest.raises(ValueError):
            merge_and_split(rows, p, m, a, b, 'hash')


def test_complete_quotas_split_and_human_gate():
    rows = []
    for dim in 'OCEAN':
        for sign in (-1, 1):
            for i in range(20):rows.append(candidate(f'{dim}{"minus" if sign == -1 else "plus"}x{i}', {dim: sign}))
    for i in range(40):
        rows.append(candidate(f'neutral{i}', {}))
        rows.append(candidate(f'multi{i}', {'O': 1, 'C': -1}))
    p, m, a, b = bundles(rows)
    report = merge_and_split(rows, p, m, a, b, 'hash')
    assert report['quota_complete'] is True and report['formal_ready'] is False
    assert [sum(r['split'] == s for r in report['accepted']) for s in ('train', 'dev', 'test')] == [168, 56, 56]
    assert all(r['labels']['human'] is None for r in report['accepted'])


def test_banned_word_on_valid_length_text_and_duplicate_punctuation():
    r = candidate()
    r['text'] = r['text'].replace('small', 'careful')
    with pytest.raises(ValueError, match='禁用'):
        validate_candidates([r])
    r = candidate()
    duplicate = {**r, 'id': 'other', 'text': r['text'].upper() + '!!!'}
    with pytest.raises(ValueError, match='重复'):
        validate_candidates([r, duplicate])


def test_builder_records_failed_and_completed_runs(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/build_episodes.py'
    spec = importlib.util.spec_from_file_location('build_episodes', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config = tmp_path / 'config.yaml'; config.write_text('use_judge: false\n')
    candidates = tmp_path / 'candidates.jsonl'; candidates.write_text(json.dumps(candidate()) + '\n')
    def setup(args, **kwargs):
        assert kwargs == {'need_model': False, 'need_judge': False}
        return {'out_path': tmp_path}, None, None, None
    monkeypatch.setattr(module, 'setup', setup)
    args = SimpleNamespace(config=str(config), candidates=str(candidates), prepared=None, labels_a=None, labels_b=None)
    out = module.run(args)
    assert json.loads((out / 'run_meta.json').read_text())['status'] == 'completed'
    args.labels_a = 'missing.json'
    with pytest.raises(ValueError, match='同时提供'):
        module.run(args)
    statuses = [json.loads(p.read_text(encoding='utf-8'))['status'] for p in tmp_path.glob('d1_builder/*/run_meta.json')]
    assert sorted(statuses) == ['completed', 'failed']
