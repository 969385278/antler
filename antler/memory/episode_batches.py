"""C10：按各批独立映射核验标签，再汇总固定槽位，绝不跨批按盲ID拼接。"""
from collections import Counter

from .d1_plan import make_plan
from .episode_data import agreement, merge_and_split, validate_candidates


def consolidate(batches, *, allow_extra_dimensions=False):
    if not batches or len({b['id'] for b in batches}) != len(batches):
        raise ValueError('批次为空或身份重复')
    rows = [r for b in batches for r in b['rows']]
    # 一次检查全部候选，包括标签被拒的故事，禁止用另一批覆盖旧槽位。
    validate_candidates(rows, allow_grouped=True)
    accepted, rejected, sources = [], [], []
    identities = None
    for b in batches:
        pair = [(b[k]['vendor'], b[k]['model']) for k in ('a', 'b')]
        if identities is not None and pair != identities:
            raise ValueError('不同批次的标注模型或顺序不一致')
        identities = pair
        report = merge_and_split(b['rows'], b['public'], b['mapping'], b['a'], b['b'],
                                 b['packet_sha256'], split_plan=make_plan(),
                                 allow_extra_dimensions=allow_extra_dimensions)
        sources.append({'batch_id': b['id'], 'packet_sha256': b['packet_sha256'], 'models': pair})
        for target, key in ((accepted, 'accepted'), (rejected, 'rejected')):
            for row in report[key]:
                target.append({**row, 'label_batch': b['id'], 'semantic_review': 'pending'})
    all_labeled = accepted + rejected
    count = lambda r: next(iter(r['intended'])) + str(next(iter(r['intended'].values()))) if len(r['intended']) == 1 else ('multi' if r['intended'] else 'neutral')
    counts = Counter(count(r) for r in accepted)
    quotas = {d + str(s): 20 for d in 'OCEAN' for s in (-1, 1)} | {'neutral': 40, 'multi': 40}
    split_counts = {split: dict(Counter(count(r) for r in accepted if r['split'] == split))
                    for split in ('train', 'dev', 'test')}
    multi = {split: {d: {str(s): sum(r['intended'].get(d) == s for r in accepted
                                   if r['split'] == split and len(r['intended']) > 1) for s in (-1, 1)}
                     for d in 'OCEAN'} for split in ('train', 'dev', 'test')}
    return {'label_accepted': accepted, 'label_rejected': rejected, 'batch_sources': sources,
            'label_policy': 'target_signs_with_extras_v1' if allow_extra_dimensions else 'strict_full_labels',
            'accepted_with_extra_labels': sum(not r['strict_label_match'] for r in accepted)
                                         if allow_extra_dimensions else 0,
            'counts': dict(counts), 'missing': {k: max(0, n - counts[k]) for k, n in quotas.items()},
            'split_counts': split_counts, 'multi_dim_counts': multi, 'quota_complete': counts == Counter(quotas),
            'agreement': agreement([r['labels']['llm_a'] for r in all_labeled], [r['labels']['llm_b'] for r in all_labeled]),
            'formal_ready': False, 'semantic_review_complete': False, 'human_review_complete': False}
