"""C10：追加结构复核并按文本核验来源，不回改封存初审或模型标签。"""
import hashlib
from collections import Counter


def join_structural_review(accepted, initial_records, decisions):
    initial = {r['id']: r for r in initial_records}
    chosen = {r['id']: r for r in decisions}
    if len(initial) != len(initial_records) or len(chosen) != len(decisions):
        raise ValueError('审核身份重复')
    expected = {r['id'] for r in accepted if initial[r['id']]['structural_status'] == 'requires_review'}
    if set(chosen) != expected:
        raise ValueError('必须完整复核标签符合且初审存疑的候选，不多选或漏选')
    records = []
    for row in accepted:
        identity = row['id']
        digest = hashlib.sha256(row['text'].encode('utf-8')).hexdigest()
        if digest != initial[identity]['text_sha256']:
            raise ValueError('候选文本与封存初审不一致')
        decision = chosen.get(identity)
        if decision:
            if decision.get('text_sha256') != digest:
                raise ValueError('复核文本哈希不符')
            if decision.get('decision') not in ('retain', 'exclude', 'unresolved') or not decision.get('reason', '').strip():
                raise ValueError('复核结论或理由缺失')
            status = decision['decision']
        else:
            if initial[identity]['structural_status'] != 'compatible':
                raise ValueError('未知初审状态')
            status = 'retain'
        records.append({'id': identity, 'text_sha256': digest, 'initial_status': initial[identity]['structural_status'],
                        'structural_decision': status, 'review': decision, 'formal_eligible': False})
    retained_ids = {r['id'] for r in records if r['structural_decision'] == 'retain'}
    retained = [r for r in accepted if r['id'] in retained_ids]
    category = lambda r: next(iter(r['intended'])) + str(next(iter(r['intended'].values()))) if len(r['intended']) == 1 else ('multi' if r['intended'] else 'neutral')
    counts = Counter(category(r) for r in retained)
    quotas = {d+str(s): 20 for d in 'OCEAN' for s in (-1, 1)} | {'neutral': 40, 'multi': 40}
    return {'reviewer': 'Codex assistant', 'human_review': False, 'blind': False, 'formal_ready': False,
            'records': records, 'review_counts': dict(Counter(r['decision'] for r in decisions)),
            'retained_count': len(retained), 'retained_target_counts': dict(counts),
            'missing': {k: max(0, n-counts[k]) for k, n in quotas.items()}}
