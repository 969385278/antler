"""C11：固定另一组提取探针，不用评测题或模型表现挑选复测题。"""
import hashlib


SALT = 'E1-probe-repeat-0:'


def probe_repeat_plan(extract_probes, reference):
    """原20拟合/10留出之外按固定哈希选20题；原留出题继续只评估。"""
    normalize = lambda q: ' '.join(q.casefold().split())
    if any(not isinstance(q, str) or not q.strip() for q in extract_probes):
        raise ValueError('探针必须为非空文本')
    pool = {normalize(q): q for q in extract_probes}
    if len(pool) != len(extract_probes):
        raise ValueError('提取探针存在规范化重复')
    if reference.get('source') != 'trait_extract':
        raise ValueError('参考探针必须来自提取集')
    fit, evaluation = reference['fit'], reference['evaluation']
    if len(fit) != 20 or len(evaluation) != 10:
        raise ValueError('参考必须为20拟合/10留出')
    if any(not isinstance(q, str) or not q.strip() for q in fit + evaluation):
        raise ValueError('参考探针为空或非文本')
    used = {normalize(q) for q in fit + evaluation}
    if len(used) != 30 or not used <= pool.keys():
        raise ValueError('参考题重复、跨组重叠或不属于提取集')
    # 不采用调用顺序或新随机种子，固定文本哈希排序决定全部复测题。
    remaining = sorted((q for q in extract_probes if normalize(q) not in used),
                       key=lambda q: hashlib.sha256((SALT + normalize(q)).encode('utf-8')).hexdigest())
    if len(remaining) < 20:
        raise ValueError('排除参考题后不足20题，不能降低配额')
    return {'scope': 'dev_probe_repeat_only', 'source': 'trait_extract', 'salt': SALT,
            'reference_fit': list(fit), 'fit': remaining[:20], 'evaluation': list(evaluation),
            'excluded_reference_count': 30, 'formal_e1_pass': False,
            'rule': 'fixed_hash_of_normalized_extract_text_excluding_all_reference_30'}
