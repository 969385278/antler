"""C10：D1 60 段方向复核的抽样、盲包与一致性统计。"""
import pytest
from antler.memory.d1_review import (CELLS, cohen_kappa, make_packet, score_review, select_review_sample,
                                     validate_review_labels)


def rows(per=6):
    out = []
    for d, s in CELLS:
        for split, k in (('train', per), ('dev', 2), ('test', 2)):
            for i in range(k):
                eid = f'{d}{s}_{split}_{i}'
                out.append({'id': eid, 'intended': {d: s}, 'split': split, 'text': f'text {eid}',
                            'labels': {'llm_a': {d: s}, 'llm_b': {d: s, 'O': 1} if d != 'O' else {d: s}}})
    out.append({'id': 'neutral', 'intended': {}, 'split': 'dev', 'text': 'n', 'labels': {'llm_a': {}, 'llm_b': {}}})
    out.append({'id': 'multi', 'intended': {'O': 1, 'C': 1}, 'split': 'test', 'text': 'm',
                'labels': {'llm_a': {}, 'llm_b': {}}})
    return out


def test_selection_is_fixed_stratified_and_ignores_text():
    r = rows(); chosen = select_review_sample(r)
    assert len(chosen) == 60 and len(set(chosen)) == 60
    assert 'neutral' not in chosen and 'multi' not in chosen
    changed = [dict(x, text='other') for x in r]
    assert select_review_sample(changed) == chosen
    by = {x['id']: x for x in r}
    for d, s in CELLS:
        sub = [by[i] for i in chosen if by[i]['intended'] == {d: s}]
        assert [x['split'] for x in sub].count('train') == 4 and len(sub) == 6


def test_quota_shortfall_raises():
    r = [x for x in rows() if not (x['intended'] == {'A': -1} and x['split'] == 'test')]
    with pytest.raises(ValueError):
        select_review_sample(r)


def test_packet_hides_ids_and_intended():
    r = rows(); public, mapping = make_packet(r, select_review_sample(r))
    assert all(set(p) == {'id', 'text'} and p['id'].startswith('R') for p in public)
    assert set(mapping) == {p['id'] for p in public}


def test_label_validation():
    public = [{'id': 'R001', 'text': 't'}]
    assert validate_review_labels([{'id': 'R001', 'labels': {'O': 1}}], public) == {'R001': {'O': 1}}
    for bad in ({'O': 2}, {'X': 1}, {'O': True}, None):
        with pytest.raises(ValueError):
            validate_review_labels([{'id': 'R001', 'labels': bad}], public)
    with pytest.raises(ValueError):
        validate_review_labels([], public)


def test_kappa_values():
    assert cohen_kappa([1, -1, 0, 0], [1, -1, 0, 0]) == pytest.approx(1)
    assert cohen_kappa([0, 0], [0, 0]) is None
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) < 0


def test_score_counts_target_and_never_claims_human():
    r = rows(); public, mapping = make_packet(r, select_review_sample(r))
    by = {x['id']: x for x in r}
    review = {}
    for k, code in enumerate(sorted(mapping)):
        (d, s), = by[mapping[code]]['intended'].items()
        review[code] = {d: s} if k % 3 else {}
    res = score_review(r, mapping, review)
    assert res['n'] == 60 and res['target_correct'] == 40 and res['target_zero'] == 20
    assert res['human_review_complete'] is False and res['reviewer_kind'] == 'assistant'
    assert res['llm_a_vs_llm_b']['n_items'] == 60
