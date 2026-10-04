"""C10：结构复核必须全覆盖并绑定原文，不伪造正式验收。"""
import hashlib
from copy import deepcopy

import pytest

from antler.memory.episode_review import join_structural_review


def inputs():
    rows = [{'id': str(i), 'text': f'episode {i}', 'intended': {'O': 1}} for i in range(3)]
    initial = [{'id': r['id'], 'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest(),
                'structural_status': 'requires_review' if i < 2 else 'compatible'} for i, r in enumerate(rows)]
    decisions = [{**r, 'decision': 'retain' if i == 0 else 'exclude', 'reason': '逐条理由'} for i, r in enumerate(initial[:2])]
    return rows, initial, decisions


def test_append_review_preserves_originals_and_formal_gate():
    args = inputs(); before = deepcopy(args)
    r = join_structural_review(*args)
    assert args == before
    assert r['retained_count'] == 2
    assert r['missing']['O1'] == 18
    assert r['review_counts'] == {'retain': 1, 'exclude': 1}
    assert not r['human_review'] and not r['blind'] and not r['formal_ready']
    assert all(not x['formal_eligible'] for x in r['records'])


@pytest.mark.parametrize('fault', ['omitted', 'duplicate', 'text', 'digest', 'reason', 'decision'])
def test_incomplete_or_changed_review_rejected(fault):
    rows, initial, decisions = inputs()
    if fault == 'omitted': decisions.pop()
    elif fault == 'duplicate': decisions.append(decisions[0])
    elif fault == 'text': rows[0]['text'] = 'changed'
    elif fault == 'digest': decisions[0]['text_sha256'] = 'wrong'
    elif fault == 'reason': decisions[0]['reason'] = ''
    else: decisions[0]['decision'] = 'pass'
    with pytest.raises(ValueError): join_structural_review(rows, initial, decisions)


def test_unresolved_is_not_retained():
    args = inputs()
    args[2][0]['decision'] = 'unresolved'
    assert join_structural_review(*args)['retained_count'] == 1
