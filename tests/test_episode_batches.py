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
