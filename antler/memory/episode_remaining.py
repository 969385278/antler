"""C10：第二轮固定缺额验收；旧233段不变，新候选不回选旧版。"""
import hashlib
from collections import Counter

from .d1_plan import make_plan
from .episode_data import label_valid, normalize, validate_candidates, validate_split_plan
from .episode_replacement import remaining_slots


def select_remaining(selection, candidates, annotations, reviews, *, invalid_annotations=()):
    """完整双标及先封存结构审核后选择唯一版本；缺失标签直接报错。"""
    keep, slots = remaining_slots(selection)
    allowed = {r['id']: r for r in slots}
    if candidates:
        validate_candidates(candidates, allow_grouped=True)
        validate_split_plan(candidates, make_plan())
    generated = {r['id']: r for r in candidates}
    labels = {r['id']: r for r in annotations}
    reviewed = {r['id']: r for r in reviews}
    invalid = {r['id']: r for r in invalid_annotations}
    if len(invalid) != len(invalid_annotations) or set(invalid) & set(labels):
        raise ValueError('无效响应重复或同时当作有效标签')
    for record in invalid.values():
        digest = record.get('response_sha256', '')
        if not record.get('reason') or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('无效标注需要原始响应哈希和原因')
    if len(labels) != len(annotations) or len(reviewed) != len(reviews):
        raise ValueError('双标或结构审核身份重复')
    if set(generated) - set(allowed) or set(labels) | set(invalid) != set(generated) or set(reviewed) != set(generated):
        raise ValueError('只能验收缺额，必须完整覆盖新候选')
    retained_texts = {normalize(r['text']) for r in keep}
    chosen = list(keep)
    ledger = [{'id': r['id'], 'status': 'old_retained',
               'text_sha256': hashlib.sha256(r['text'].encode()).hexdigest()} for r in keep]
    for identity, slot in allowed.items():
        if identity not in generated:
            ledger.append({'id': identity, 'status': 'no_new_candidate'})
            continue
        row, annotation, review = generated[identity], labels.get(identity), reviewed[identity]
        digest = hashlib.sha256(row['text'].encode()).hexdigest()
        if (row.get('replacement_round') != 2 or row.get('attempt') != 4
                or row.get('previous_text_sha256') != slot['previous_text_sha256']):
            raise ValueError('新版本轮次或前版绑定错误')
        if normalize(row['text']) in retained_texts:
            raise ValueError('新版本复制保留文本')
        if (annotation is not None and annotation.get('text_sha256') != digest) or review.get('text_sha256') != digest:
            raise ValueError('标签或结构审核未绑定原文')
        if review.get('status') not in ('compatible', 'requires_review', 'exclude'):
            raise ValueError('结构审核状态非法')
        if identity in invalid:
            ledger.append({'id': identity, 'status': 'new_annotation_invalid', 'text_sha256': digest,
                           'annotation_error': invalid[identity]})
            continue
        a, b = annotation['llm_a'], annotation['llm_b']
        if not label_valid(a) or not label_valid(b):
            raise ValueError('非法标签不可当中性')
        target = row['intended']
        ok = all(a.get(d) == b.get(d) == s for d, s in target.items()) if target else a == b == {}
        status = ('new_label_rejected' if not ok else
                  'new_selected' if review['status'] == 'compatible' else 'new_structure_rejected')
        ledger.append({'id': identity, 'status': status, 'text_sha256': digest,
                       'previous_text_sha256': slot['previous_text_sha256']})
        if status == 'new_selected':
            chosen.append({**row, 'labels': {'llm_a': a, 'llm_b': b, 'human': None},
                           'label_policy': 'target_signs_allow_extra_dimensions',
                           'strict_label_match': a == b == target,
                           'extra_labels': {'llm_a': {d: s for d, s in a.items() if d not in target},
                                            'llm_b': {d: s for d, s in b.items() if d not in target}}})
    validate_candidates(chosen, allow_grouped=True)
    validate_split_plan(chosen, make_plan())
    order = {r['id']: i for i, r in enumerate(make_plan()['slots'])}
    return {'selected': sorted(chosen, key=lambda r: order[r['id']]),
            'ledger': sorted(ledger, key=lambda r: order[r['id']]),
            'counts': dict(Counter(r['status'] for r in ledger)),
            'formal_ready': False, 'human_review_complete': False}
