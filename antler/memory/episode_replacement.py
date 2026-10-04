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


def remaining_slots(selection, *, round_id=2):
    """仅允许已登记的第二轮47槽位或第三轮26槽位，保留版本不变。"""
    if type(round_id) is not int or round_id not in (2, 3):
        raise ValueError('仅允许已授权的第二轮或第三轮')
    expected_keep = 233 if round_id == 2 else 254
    keep = selection['selected']
    validate_candidates(keep, allow_grouped=True)
    plan = make_plan()
    validate_split_plan(keep, plan)
    ledger = selection['ledger']
    indexed = {r['id']: r for r in ledger}
    planned = {r['id']: r for r in plan['slots']}
    retained = {r['id']: r for r in keep}
    if len(indexed) != len(ledger) or set(indexed) != set(planned) or len(retained) != expected_keep:
        raise ValueError('必须绑定280槽位和本轮登记的保留版本数')
    for identity, record in indexed.items():
        if (record['status'] in ('old_retained', 'new_selected')) != (identity in retained):
            raise ValueError('保留版本与台账状态不一致')
        if identity in retained and record['text_sha256'] != hashlib.sha256(retained[identity]['text'].encode()).hexdigest():
            raise ValueError('保留文本哈希不一致')
    slots = [{**s, 'replacement_round': round_id, 'previous_text_sha256': indexed[s['id']].get('text_sha256')}
             for s in plan['slots'] if s['id'] not in retained]
    if len(slots) != 280 - expected_keep:
        raise ValueError('本轮只允许登记的固定缺额')
    return keep, slots
