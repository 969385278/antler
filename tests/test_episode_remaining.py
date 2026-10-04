"""C10：缺额验收保留失败，中性不利用空目标自动通过。"""
import hashlib
from copy import deepcopy

import pytest

from antler.memory.episode_remaining import select_remaining
from antler.memory.episode_replacement import remaining_slots
from test_episode_data import candidate
from test_episode_replacement import remaining_data


def inputs():
    selection = remaining_data()
    _, slots = remaining_slots(selection)
    rows, annotations, reviews = [], [], []
    for s in slots:
        row = {**candidate(s['id'], s['intended']), **s, 'attempt': 4}
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        rows.append(row)
        annotations.append({'id': s['id'], 'text_sha256': digest, 'llm_a': s['intended'], 'llm_b': s['intended']})
        reviews.append({'id': s['id'], 'text_sha256': digest, 'status': 'compatible'})
    return selection, rows, annotations, reviews


def test_all_retained_and_no_false_formal_acceptance():
    args = inputs(); before = deepcopy(args)
    result = select_remaining(*args)
    assert args == before and len(result['selected']) == 280
    retained = {r['id']: r for r in result['selected']}
    assert all(retained[r['id']] == r for r in args[0]['selected'])
    assert not result['formal_ready'] and not result['human_review_complete']


def test_neutral_and_structure_rejections_preserved():
    s, rows, annotations, reviews = inputs()
    index = next(i for i, r in enumerate(rows) if not r['intended'])
    annotations[index]['llm_a'] = {'O': 1}
    other = next(i for i in range(len(rows)) if i != index)
    reviews[other]['status'] = 'requires_review'
    result = select_remaining(s, rows, annotations, reviews)
    assert result['counts']['new_label_rejected'] == 1
    assert result['counts']['new_structure_rejected'] == 1
    assert len(result['selected']) == 278


@pytest.mark.parametrize('fault', ['missing', 'hash', 'round', 'previous', 'duplicate', 'label'])
def test_invalid_evidence_rejected(fault):
    s, rows, annotations, reviews = inputs()
    if fault == 'missing': annotations.pop()
    elif fault == 'hash': reviews[0]['text_sha256'] = 'a'*64
    elif fault == 'round': rows[0]['replacement_round'] = 1
    elif fault == 'previous': rows[0]['previous_text_sha256'] = None
    elif fault == 'duplicate': reviews.append(reviews[0])
    else: annotations[0]['llm_a'] = {'O': 2}
    with pytest.raises(ValueError): select_remaining(s, rows, annotations, reviews)
