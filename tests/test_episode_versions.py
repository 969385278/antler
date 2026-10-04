"""C10：新旧版本不得择优、混用或静默覆盖，机械候选审核必须齐全。"""
import hashlib
from copy import deepcopy

import pytest

from antler.memory.episode_versions import select_versions
from antler.memory.episode_replacement import replacement_slots
from test_episode_replacement import data
from test_episode_data import candidate


def inputs():
    a, b, old_review = data()
    _, slots = replacement_slots(a, b, old_review)
    new = [{**candidate('new-version', slots[0]['intended']), **slots[0]}]
    row = {**new[0], 'labels': {'llm_a': dict(slots[0]['intended']), 'llm_b': dict(slots[0]['intended']), 'human': None}}
    reviews = [{'id': row['id'], 'text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(), 'status': 'compatible'}]
    return a, b, old_review, new, [row], [], reviews


def test_version_ledger_preserves_inputs_and_never_declares_formal_ready():
    args = inputs(); before = deepcopy(args)
    result = select_versions(*args)
    assert args == before
    assert len(result['selected']) == 2 and len(result['ledger']) == 280
    assert result['counts'] == {'old_retained': 1, 'new_selected': 1, 'no_new_candidate': 278}
    assert not result['formal_ready'] and not result['human_review_complete']


def test_rejected_new_version_does_not_restore_rejected_old_version():
    args = list(inputs())
    row = args[4].pop()
    row['labels']['llm_a'] = {}
    args[5].append(row)
    result = select_versions(*args)
    assert len(result['selected']) == 1
    assert result['counts']['new_label_rejected'] == 1


@pytest.mark.parametrize('fault', ['missing_review', 'wrong_hash', 'changed_text', 'old_sha', 'wrong_labels', 'duplicate'])
def test_corrupt_or_incomplete_new_evidence_is_rejected(fault):
    args = list(inputs())
    if fault == 'missing_review': args[6].clear()
    elif fault == 'wrong_hash': args[6][0]['text_sha256'] = 'other'
    elif fault == 'changed_text': args[4][0]['text'] = 'other'
    elif fault == 'old_sha': args[3][0]['previous_text_sha256'] = 'other'
    elif fault == 'wrong_labels': args[4][0]['labels']['llm_b'] = {}
    else: args[5].append(args[4][0])
    with pytest.raises(ValueError): select_versions(*args)


def test_structural_doubt_keeps_slot_unfilled():
    args = inputs(); args[6][0]['status'] = 'requires_review'
    result = select_versions(*args)
    assert len(result['selected']) == 1
    assert result['counts']['new_structure_rejected'] == 1


def test_copied_rejected_old_text_cannot_be_a_new_version():
    args = inputs()
    args[3][0]['text'] = args[0][1]['text']
    with pytest.raises(ValueError, match='复制'):
        select_versions(*args)
