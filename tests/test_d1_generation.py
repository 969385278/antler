"""C10：预先家族切分、正负配额、API归档与失败保留。"""
from collections import Counter
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.memory.d1_plan import make_plan, generation_prompt
from antler.memory.episode_data import validate_split_plan, blind_packet
from test_episode_data import candidate


def test_slots_are_balanced_and_families_do_not_cross_splits():
    plan = make_plan(); slots = plan['slots']
    assert len(slots) == len({r['id'] for r in slots}) == 280
    assert Counter(r['split'] for r in slots) == {'train': 168, 'dev': 56, 'test': 56}
    for group in {r['group_id'] for r in slots}:
        assert len({r['split'] for r in slots if r['group_id'] == group}) == 1
    for split, per_sign in [('train', 12), ('dev', 4), ('test', 4)]:
        single = Counter((next(iter(r['intended'])), next(iter(r['intended'].values())))
                         for r in slots if r['split'] == split and len(r['intended']) == 1)
        assert len(single) == 10 and set(single.values()) == {per_sign}
        multi = Counter((d, s) for r in slots if r['split'] == split and len(r['intended']) == 2 for d, s in r['intended'].items())
        assert len(multi) == 10
        assert all(multi[d, 1] == multi[d, -1] for d in 'OCEAN')


def test_group_plan_rejects_reassignment_and_blinds_metadata():
    plan = make_plan(); slot = plan['slots'][0]
    row = {**candidate(slot['id'], slot['intended']), 'group_id': slot['group_id'], 'split': slot['split']}
    validate_split_plan([row], plan)
    public, _ = blind_packet([row], allow_grouped=True)
    assert set(public[0]) == {'id', 'text'}
    row['split'] = 'dev' if slot['split'] != 'dev' else 'train'
    with pytest.raises(ValueError, match='split'):
        validate_split_plan([row], plan)
    assert slot['id'] not in generation_prompt(slot)


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('d1_generate')


def test_single_generation_keeps_payload_and_rejects_short_output(module, tmp_path):
    slot = make_plan()['slots'][0]
    conf = {'MODEL': 'MiniMax-M3', 'endpoint': 'https://example.org/v1/chat/completions', 'API_KEY': 'test-secret'}
    def post(url, **kw):
        assert kw['json']['thinking'] == {'type': 'disabled'}
        return SimpleNamespace(status_code=200, json=lambda: {'model': conf['MODEL'], 'choices': [
            {'finish_reason': 'stop', 'message': {'content': json.dumps({'text': candidate()['text']})}}]})
    result = module.generate_one(slot, conf, tmp_path, post)
    assert result['valid'] and result['candidate']['group_id'] == slot['group_id']
    assert 'test-secret' not in (tmp_path / slot['id'] / 'request.json').read_text()
    other = make_plan()['slots'][1]
    response = SimpleNamespace(status_code=200, json=lambda: {'model': conf['MODEL'], 'choices': [
        {'finish_reason': 'stop', 'message': {'content': '{"text":"I learned."}'}}]})
    assert not module.generate_one(other, conf, tmp_path, lambda *a, **k: response)['valid']


def test_assembly_keeps_failure_and_uses_plan_order(module):
    plan = make_plan(); results = {}
    for slot in reversed(plan['slots']):
        results[slot['id']] = {'id': slot['id'], 'valid': True, 'candidate': candidate(slot['id'])}
    results[plan['slots'][0]['id']] = {'id': plan['slots'][0]['id'], 'valid': False, 'reason': 'HTTP 429'}
    good, bad = module.assemble(results, plan)
    assert len(good) == 279 and len(bad) == 1
    assert good[0]['id'] == plan['slots'][1]['id']
    with pytest.raises(ValueError, match='不完整'):
        module.assemble({}, plan)
