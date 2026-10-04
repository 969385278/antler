"""C10：D1 60 段方向复核的固定抽样、盲包与一致性统计。

规格 5.3 原要求由作者本人盲标。2026-10-04 作者指定改由助手执行，因此本模块产生的
标签一律标记为非人工（reviewer_kind='assistant'），不能写成人工验收已完成。
抽样只依据 id、预定目标和 split 的哈希排序，不读取文本、模型标签或 Qwen 结果。
"""
import hashlib

DIMS = ['O', 'C', 'E', 'A', 'N']
SALT = 'd1_direction_review_60_v1'
SPLIT_QUOTA = {'train': 4, 'dev': 1, 'test': 1}
CELLS = [(d, s) for d in DIMS for s in (1, -1)]


def _rank(key):
    return hashlib.sha256(f'{SALT}|{key}'.encode()).hexdigest()


def cell_of(intended):
    """单目标经历返回 (维度, 符号)；中性或多目标返回 None。"""
    if not isinstance(intended, dict) or len(intended) != 1:
        return None
    (dim, sign), = intended.items()
    if dim not in DIMS or sign not in (1, -1):
        raise ValueError(f'非法预定目标：{intended}')
    return dim, sign


def select_review_sample(rows, split_quota=SPLIT_QUOTA):
    """每维每方向按 split 配额取哈希最小者；配额不足直接报错，不跨 split 补位。"""
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('id 重复')
    chosen = []
    for cell in CELLS:
        for split, quota in split_quota.items():
            pool = sorted((r['id'] for r in rows if cell_of(r['intended']) == cell and r['split'] == split), key=_rank)
            if len(pool) < quota:
                raise ValueError(f'{cell} {split} 只有 {len(pool)} 段，少于配额 {quota}')
            chosen += pool[:quota]
    return chosen


def make_packet(rows, chosen):
    """只含随机编号和文本的盲包；映射另存，不给复核者。"""
    by_id = {r['id']: r for r in rows}
    order = sorted(chosen, key=lambda i: _rank('order|' + i))
    public, mapping = [], {}
    for k, eid in enumerate(order, 1):
        code = f'R{k:03d}'
        public.append({'id': code, 'text': by_id[eid]['text']})
        mapping[code] = eid
    return public, mapping


def validate_review_labels(records, public):
    codes = [r.get('id') for r in records]
    if len(codes) != len(set(codes)) or set(codes) != {p['id'] for p in public}:
        raise ValueError('复核标签缺失、重复或含未知编号')
    out = {}
    for r in records:
        labels = r.get('labels')
        if not isinstance(labels, dict) or any(k not in DIMS or v not in (1, -1) or isinstance(v, bool)
                                               for k, v in labels.items()):
            raise ValueError(f'{r.get("id")} 标签非法：{labels}')
        out[r['id']] = labels
    return out


def _vec(labels):
    return [int((labels or {}).get(d, 0)) for d in DIMS]


def cohen_kappa(a, b, classes=(-1, 0, 1)):
    """名义 Cohen κ；期望一致为 1 时返回 None，避免除零伪造结果。"""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError('κ 输入长度不符')
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in classes)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _pair_stats(xs, ys):
    flat_x = [v for row in xs for v in row]
    flat_y = [v for row in ys for v in row]
    per_dim = {}
    for j, d in enumerate(DIMS):
        a = [row[j] for row in xs]; b = [row[j] for row in ys]
        per_dim[d] = {'agreement': sum(p == q for p, q in zip(a, b)) / len(a), 'kappa': cohen_kappa(a, b)}
    return {'n_items': len(xs), 'pooled_agreement': sum(p == q for p, q in zip(flat_x, flat_y)) / len(flat_x),
            'pooled_kappa': cohen_kappa(flat_x, flat_y), 'exact_vector_match': sum(p == q for p, q in zip(xs, ys)) / len(xs),
            'per_dim': per_dim}


def score_review(rows, mapping, review):
    """复核与预定目标、两模型标签的一致性；结果永远 human_review_complete=False。"""
    by_id = {r['id']: r for r in rows}
    items, cells = [], {f'{d}{"+" if s > 0 else "-"}': {'n': 0, 'target_correct': 0} for d, s in CELLS}
    rev, la, lb = [], [], []
    for code in sorted(mapping):
        row = by_id[mapping[code]]
        dim, sign = cell_of(row['intended'])
        lab = review[code]
        got = lab.get(dim, 0)
        key = f'{dim}{"+" if sign > 0 else "-"}'
        cells[key]['n'] += 1; cells[key]['target_correct'] += int(got == sign)
        rev.append(_vec(lab)); la.append(_vec(row['labels']['llm_a'])); lb.append(_vec(row['labels']['llm_b']))
        items.append({'code': code, 'id': row['id'], 'split': row['split'], 'intended': row['intended'],
                      'review': lab, 'llm_a': row['labels']['llm_a'], 'llm_b': row['labels']['llm_b'],
                      'target_correct': got == sign, 'target_review_value': got})
    n = len(items)
    correct = sum(i['target_correct'] for i in items)
    return {'reviewer_kind': 'assistant', 'human_review_complete': False, 'n': n,
            'target_direction_accuracy': correct / n, 'target_correct': correct,
            'target_zero': sum(i['target_review_value'] == 0 for i in items),
            'target_opposite': sum(i['target_review_value'] == -cell_of(i['intended'])[1] for i in items),
            'per_cell': cells, 'review_vs_llm_a': _pair_stats(rev, la), 'review_vs_llm_b': _pair_stats(rev, lb),
            'llm_a_vs_llm_b': _pair_stats(la, lb), 'items': items}
