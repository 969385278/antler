"""C10：按固定规则选定槽位版本，保留拒收轨迹，不回选旧失败文本。"""
import hashlib
from collections import Counter

from .d1_plan import make_plan
from .episode_data import label_valid, normalize, validate_candidates, validate_split_plan
from .episode_replacement import replacement_slots


def select_versions(old_accepted, old_rejected, old_review, new_candidates, new_accepted, new_rejected, new_reviews,
                    *, invalid_annotations=()):
    keep, slots = replacement_slots(old_accepted, old_rejected, old_review)
    allowed = {r['id']: r for r in slots}
    validate_candidates(new_candidates, allow_grouped=True)
    validate_split_plan(new_candidates, make_plan())
    old_texts = {normalize(r['text']) for r in old_accepted + old_rejected}
    if any(normalize(r['text']) in old_texts for r in new_candidates):
        raise ValueError('新版本不得复制任一旧候选文本')
    generated = {r['id']: r for r in new_candidates}
    labeled = new_accepted + new_rejected
    labels = {r['id']: r for r in labeled}
    reviewed = {r['id']: r for r in new_reviews}
    invalid = {r['id']: r for r in invalid_annotations}
    if len(invalid) != len(invalid_annotations) or set(invalid) & set(labels):
        raise ValueError('无效标注身份重复或同时被当作有效标签')
    for r in invalid.values():
        digest = r.get('response_sha256', '')
        if not r.get('reason') or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('无效标注必须绑定原始响应哈希和原因')
    if len(labels) != len(labeled) or len(reviewed) != len(new_reviews):
        raise ValueError('新版本标签或结构审核身份重复')
    if set(generated) != set(labels) | set(invalid) or set(generated) != set(reviewed):
        raise ValueError('必须完整覆盖本轮机械候选的双标与结构审核')
    label_ok_ids = {r['id'] for r in new_accepted}
    chosen = list(keep)
    ledger = {r['id']: {'id': r['id'], 'status': 'old_retained', 'text_sha256': hashlib.sha256(r['text'].encode('utf-8')).hexdigest()}
              for r in keep}
    for identity, slot in allowed.items():
        if identity not in generated:
            ledger[identity] = {'id': identity, 'status': 'no_new_candidate'}
            continue
        row, annotation, review = generated[identity], labels.get(identity), reviewed[identity]
        if row.get('replacement_round') != 1 or row.get('previous_text_sha256') != slot['previous_text_sha256']:
            raise ValueError('新版本未绑定登记的旧文本或补写轮次')
        if annotation is not None and any(annotation.get(k) != v for k, v in row.items() if k != 'labels'):
            raise ValueError('标注改变了生成候选')
        digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
        if review.get('text_sha256') != digest or review.get('status') not in ('compatible', 'requires_review', 'exclude'):
            raise ValueError('结构审核文本或状态错误')
        if identity in invalid:
            ledger[identity] = {'id': identity, 'status': 'new_annotation_invalid', 'text_sha256': digest,
                                'annotation_error': invalid[identity], 'previous_text_sha256': slot['previous_text_sha256']}
            continue
        a, b = annotation['labels']['llm_a'], annotation['labels']['llm_b']
        if not label_valid(a) or not label_valid(b):
            raise ValueError('非法模型标签')
        target = row['intended']
        label_ok = all(a.get(d) == b.get(d) == s for d, s in target.items()) if target else a == b == {}
        if label_ok != (identity in label_ok_ids):
            raise ValueError('新版本标签验收不符合目标符号规则')
        status = 'new_label_rejected' if not label_ok else ('new_selected' if review['status'] == 'compatible' else 'new_structure_rejected')
        ledger[identity] = {'id': identity, 'status': status, 'text_sha256': digest,
                            'previous_text_sha256': slot['previous_text_sha256']}
        if status == 'new_selected':
            chosen.append(annotation)
    if set(generated) - set(allowed):
        raise ValueError('禁止重写已保留的旧槽位')
    validate_candidates(chosen, allow_grouped=True)
    validate_split_plan(chosen, make_plan())
    order = [r['id'] for r in make_plan()['slots']]
    by_id = {r['id']: r for r in chosen}
    return {'selected': [by_id[i] for i in order if i in by_id],
            'ledger': [ledger[i] for i in order], 'counts': dict(Counter(r['status'] for r in ledger.values())),
            'formal_ready': False, 'human_review_complete': False,
            'note': '选择清单只解决版本唯一性与候选审核，不代表正式E1数据验收。'}
