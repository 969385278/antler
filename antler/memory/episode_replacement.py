"""C10：从完整旧候选和封存复核导出缺额，保留原家族与切分。"""
import hashlib

from .d1_plan import make_plan
from .episode_data import validate_candidates, validate_split_plan


def replacement_slots(accepted, rejected, review):
    rows = accepted + rejected
    validate_candidates(rows, allow_grouped=True)
    plan = make_plan()
    validate_split_plan(rows, plan)
    records = review['records']
    indexed = {r['id']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != {r['id'] for r in accepted}:
        raise ValueError('复核必须精确覆盖全部标签符合候选')
    for row in accepted:
        record = indexed[row['id']]
        if record['text_sha256'] != hashlib.sha256(row['text'].encode('utf-8')).hexdigest():
            raise ValueError('复核与原文不一致')
        if record['structural_decision'] not in ('retain', 'exclude', 'unresolved'):
            raise ValueError('未知复核状态')
    keep = [r for r in accepted if indexed[r['id']]['structural_decision'] == 'retain']
    if review['retained_count'] != len(keep):
        raise ValueError('复核保留数不符')
    retained = {r['id'] for r in keep}
    old = {r['id']: r for r in rows}
    slots = []
    for slot in plan['slots']:
        if slot['id'] not in retained:
            prev = old.get(slot['id'])
            slots.append({**slot, 'replacement_round': 1,
                          'previous_text_sha256': hashlib.sha256(prev['text'].encode('utf-8')).hexdigest() if prev else None})
    return keep, slots
