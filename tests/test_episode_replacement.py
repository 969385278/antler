"""C10：补写限定缺额，保留旧文、情境和切分，接口失败不重发。"""
import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.memory.d1_plan import make_plan
from antler.memory.episode_replacement import replacement_slots
from test_episode_data import candidate


def data():
    slots = make_plan()['slots']
    accepted = [{**candidate(s['id'], s['intended']), **s} for s in slots[:2]]
    rejected = [{**candidate(slots[2]['id'], slots[2]['intended']), **slots[2]}]
    review = {'retained_count': 1, 'records': [{'id': r['id'], 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest(),
              'structural_decision': 'retain' if i == 0 else 'exclude'} for i, r in enumerate(accepted)]}
    return accepted, rejected, review


def test_slots_preserve_families_and_keep_immutable_old_text():
    args = data(); before = deepcopy(args)
    keep, slots = replacement_slots(*args)
    assert args == before
    assert len(keep) == 1 and len(slots) == 279
    assert keep[0]['id'] not in {s['id'] for s in slots}
    assert slots[0]['previous_text_sha256'] == args[2]['records'][1]['text_sha256']
    assert slots[2]['previous_text_sha256'] is None
    planned = {s['id']: s for s in make_plan()['slots']}
    for s in slots:
        assert all(s[k] == v for k, v in planned[s['id']].items())


@pytest.mark.parametrize('fault', ['hash', 'missing', 'count', 'split', 'duplicate'])
def test_corrupt_selection_rejected(fault):
    a, b, r = data()
    if fault == 'hash': r['records'][0]['text_sha256'] = 'wrong'
    elif fault == 'missing': r['records'].pop()
    elif fault == 'count': r['retained_count'] = 2
    elif fault == 'split': a[0]['split'] = 'train'
    else: b.append(a[0])
    with pytest.raises(ValueError): replacement_slots(a, b, r)


def test_dry_run_and_http_stop_preserve_claim(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    m = importlib.import_module('d1_replace')
    a, b, review = data()
    _, slots = replacement_slots(a, b, review)
    review_path = tmp_path / 'review.json'; review_path.write_text('{}')
    config = tmp_path / 'config.yaml'; config.write_text('out_dir: test')
    monkeypatch.setattr(m, 'setup', lambda *a, **k: ({'out_path': tmp_path}, None, None, None))
    monkeypatch.setattr(m, 'load', lambda *args: (a+b, a[:1], slots[:2], {}))
    conf = {'MODEL': 'MiniMax-M3', 'endpoint': 'https://example.invalid', 'API_KEY': 'secret'}
    calls = []
    monkeypatch.setattr(m, 'read_connection', lambda *args: conf)
    def gen(slot, conf, out, **kwargs):
        calls.append(slot['id']); (out / slot['id']).mkdir()
        assert kwargs['attempt'] == 3 and m.SUFFIX in kwargs['prompt_suffix']
        return {'id': slot['id'], 'valid': False, 'reason': 'HTTP 429'}
    monkeypatch.setattr(m, 'generate_one', gen)
    args = SimpleNamespace(config=str(config), source=str(tmp_path), review=str(review_path), env_file='.env', execute=False)
    assert m.run(args) is None and not calls
    args.execute = True
    out = m.run(args)
    assert len(calls) == 1
    assert json.loads((out/'run_meta.json').read_text())['status'] == 'stopped'
    assert len(json.loads((out/'not_completed.json').read_text())) == 1
    with pytest.raises(FileExistsError): m.run(args)
    assert len(calls) == 1


def remaining_data():
    from antler.memory.d1_plan import make_plan
    keep = [{**candidate(s['id'], s['intended']), **s} for s in make_plan()['slots'][:233]]
    ledger = []
    by_id = {r['id']: r for r in keep}
    for s in make_plan()['slots']:
        r = by_id.get(s['id'])
        ledger.append({'id': s['id'], 'status': 'old_retained' if r else 'new_label_rejected',
                       'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest() if r else 'a'*64})
    return {'selected': keep, 'ledger': ledger}


def test_remaining_round_preserves_233_and_binds_47():
    from antler.memory.episode_replacement import remaining_slots
    source = remaining_data(); before = deepcopy(source)
    keep, slots = remaining_slots(source)
    assert source == before and keep == source['selected']
    assert len(slots) == 47 and all(s['replacement_round'] == 2 for s in slots)
    assert all(s['previous_text_sha256'] == 'a'*64 for s in slots)
    assert not {s['id'] for s in slots} & {r['id'] for r in keep}


@pytest.mark.parametrize('fault', ['text', 'status', 'missing', 'duplicate', 'split'])
def test_remaining_rejects_changed_ledger(fault):
    from antler.memory.episode_replacement import remaining_slots
    source = remaining_data()
    if fault == 'text': source['ledger'][0]['text_sha256'] = 'b'*64
    elif fault == 'status': source['ledger'][0]['status'] = 'new_label_rejected'
    elif fault == 'missing': source['ledger'].pop()
    elif fault == 'duplicate': source['ledger'].append(source['ledger'][0])
    else: source['selected'][0]['split'] = 'train'
    with pytest.raises(ValueError): remaining_slots(source)
