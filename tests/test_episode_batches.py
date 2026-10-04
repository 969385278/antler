"""C10：跨批盲ID重复不混淆原槽位，拒绝替换已存在故事或挪动切分。"""
from copy import deepcopy
import pytest

from antler.memory.d1_plan import make_plan
from antler.memory.episode_data import blind_packet
from antler.memory.episode_batches import consolidate
from test_episode_data import candidate


def batch(index):
    slot = make_plan()['slots'][index]
    row = {**candidate(slot['id'], slot['intended']), **slot}
    public, mapping = blind_packet([row], allow_grouped=True)
    a = dict(vendor='minimax', model='MiniMax-M3', input_sha256='hash',
             records=[{'id': public[0]['id'], 'labels': slot['intended']}])
    b = deepcopy(a); b.update(vendor='glm', model='glm-5')
    return dict(id=f'batch{index}', rows=[row], public=public, mapping=mapping, a=a, b=b, packet_sha256='hash')


def test_repeated_blind_id_preserves_batch_and_sign():
    a, b = batch(0), batch(1)
    assert a['public'][0]['id'] == b['public'][0]['id']
    r = consolidate([a, b])
    assert len(r['label_accepted']) == 2
    assert {x['label_batch'] for x in r['label_accepted']} == {'batch0', 'batch1'}
    assert r['counts'] == {'O-1': 1, 'O1': 1}
    assert not r['formal_ready'] and not r['semantic_review_complete']


def test_disagreement_never_disappears_and_duplicate_slot_rejected():
    a, b = batch(0), batch(1)
    b['b']['records'][0]['labels'] = {}
    r = consolidate([a, b])
    assert len(r['label_rejected']) == len(r['label_accepted']) == 1
    c = deepcopy(a); c['id'] = 'another'
    with pytest.raises(ValueError, match='重复'):
        consolidate([a, c])


@pytest.mark.parametrize('fault', ['mapping', 'split', 'model', 'hash'])
def test_corrupt_batch_rejected(fault):
    a, b = batch(0), batch(1)
    if fault == 'mapping': b['mapping'][b['public'][0]['id']] = a['rows'][0]['id']
    elif fault == 'split': b['rows'][0]['split'] = 'train'
    elif fault == 'model': b['a']['model'] = 'other'
    else: b['a']['input_sha256'] = 'other'
    with pytest.raises(ValueError): consolidate([a, b])


def test_optional_extra_dimensions_preserve_labels_and_split():
    a = batch(0)
    a['a']['records'][0]['labels'] = {'O': -1, 'N': 1}
    a['b']['records'][0]['labels'] = {'O': -1, 'N': -1, 'A': 1}
    original = deepcopy(a)
    assert len(consolidate([a])['label_rejected']) == 1
    report = consolidate([a], allow_extra_dimensions=True)
    row = report['label_accepted'][0]
    assert row['extra_labels'] == {'llm_a': {'N': 1}, 'llm_b': {'N': -1, 'A': 1}}
    assert row['intended'] == {'O': -1}
    assert row['split'] == a['rows'][0]['split']
    assert report['accepted_with_extra_labels'] == 1
    assert not row['strict_label_match'] and not report['formal_ready']
    assert a == original


@pytest.mark.parametrize('labels', [{'N': 1}, {'O': 1, 'N': 1}])
def test_extra_policy_does_not_accept_missing_or_wrong_target(labels):
    a = batch(0)
    a['a']['records'][0]['labels'] = labels
    assert len(consolidate([a], allow_extra_dimensions=True)['label_rejected']) == 1


def test_extra_policy_keeps_neutral_requirement_and_all_multi_targets():
    slots = make_plan()['slots']
    neutral = batch(next(i for i, s in enumerate(slots) if not s['intended']))
    neutral['a']['records'][0]['labels'] = {'N': 1}
    assert len(consolidate([neutral], allow_extra_dimensions=True)['label_rejected']) == 1
    multi = batch(next(i for i, s in enumerate(slots) if len(s['intended']) > 1))
    multi['a']['records'][0]['labels'] = dict(multi['a']['records'][0]['labels'])
    multi['a']['records'][0]['labels'].pop(next(iter(multi['rows'][0]['intended'])))
    assert len(consolidate([multi], allow_extra_dimensions=True)['label_rejected']) == 1
