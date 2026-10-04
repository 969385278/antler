"""C10：D1候选检查与盲标合并；不生成标签、不冒充人工复核。"""
from collections import Counter
import random
import re

from ..traits.data import DIMS


# 固定禁词表，覆盖五维名称及常见两极词形；不做宽泛子串匹配。
BANNED = frozenset('openness conscientiousness extraversion extroversion agreeableness neuroticism '
                   'careful carefully careless carelessly dependable conscientious unconscientious '
                   'outgoing introverted extroverted extraverted solitary compassionate selfish '
                   'anxious anxiously anxiety nervous nervously neurotic calm calmly inventive '
                   'agreeable disagreeable open-minded closed-minded self-interested'.split())


def normalize(text):
    return ' '.join(re.findall(r"[a-z]+(?:[-'][a-z]+)?|[0-9]+", text.casefold()))


def label_valid(value):
    return isinstance(value, dict) and set(value) <= set(DIMS) and all(type(v) is int and v in (-1, 1) for v in value.values())


def validate_candidates(rows, judge_vendor='deepseek', *, allow_grouped=False):
    """机械检查不代表故事语义合格，结构和人格方向仍由独立盲标复核。"""
    seen_ids, seen_texts = set(), set()
    if not rows:
        raise ValueError('候选经历不能为空')
    for row in rows:
        identity, text = row.get('id'), row.get('text')
        if not isinstance(identity, str) or not identity.strip() or identity in seen_ids:
            raise ValueError('候选id为空或重复')
        if not isinstance(text, str) or not text.strip() or normalize(text) in seen_texts:
            raise ValueError('候选文本为空或规范化后重复')
        words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)
        if not 40 <= len(words) <= 80:
            raise ValueError(f'{identity}不满足40–80词')
        if not re.search(r'\b(I|me|my|mine)\b', text, re.I):
            raise ValueError(f'{identity}缺少第一人称')
        hits = BANNED & {w.casefold() for w in words}
        if hits:
            raise ValueError(f'{identity}包含禁用特质词：{sorted(hits)}')
        if not label_valid(row.get('intended')):
            raise ValueError(f'{identity}预期方向非法')
        if not isinstance(row.get('generator'), str) or not row['generator'].strip():
            raise ValueError('缺少生成模型身份')
        vendor = row.get('generator_vendor')
        if not isinstance(vendor, str) or not vendor.strip() or vendor.strip().casefold() == judge_vendor.casefold():
            raise ValueError('生成厂商缺失或与主评委相同')
        # 本阶段只接收独立经历；语义变体须另行登记分组后再切分，不能跨split混入。
        if any(row.get(k) is not None for k in ('twin_id', 'paraphrase_of')) or (row.get('group_id') is not None and not allow_grouped):
            raise ValueError('有关联经历须另行分组划分，当前入口拒绝独立切分')
        seen_ids.add(identity); seen_texts.add(normalize(text))


def blind_packet(rows, *, allow_grouped=False):
    """公开id不编码维度或方向，映射留在独立文件；按固定seed打乱。"""
    validate_candidates(rows, allow_grouped=allow_grouped)
    shuffled = sorted(rows, key=lambda r: r['id'])
    random.Random(0).shuffle(shuffled)
    public, mapping = [], {}
    for index, row in enumerate(shuffled):
        identity = f'D1B{index + 1:04d}'
        public.append({'id': identity, 'text': row['text']})
        mapping[identity] = row['id']
    return public, mapping


def validate_labels(bundle, public, packet_sha256, judge_vendor='deepseek'):
    vendor, model = bundle.get('vendor'), bundle.get('model')
    if not isinstance(vendor, str) or not vendor.strip() or vendor.strip().casefold() == judge_vendor.casefold():
        raise ValueError('标注模型厂商缺失或与主评委相同')
    if not isinstance(model, str) or not model.strip() or bundle.get('input_sha256') != packet_sha256:
        raise ValueError('标注模型身份或盲标输入哈希不符')
    records = bundle.get('records', [])
    ids = [r.get('id') for r in records]
    if len(ids) != len(set(ids)) or set(ids) != {r['id'] for r in public}:
        raise ValueError('标签缺失、重复或包含未知id')
    if any(not label_valid(r.get('labels')) for r in records):
        raise ValueError('独立标签非法')
    return {r['id']: r['labels'] for r in records}


def agreement(a, b):
    """逐维包含0=未标注影响的一致率和kappa；不把0当真实中性证明。"""
    result = {}
    for dim in DIMS:
        x, y = [v.get(dim, 0) for v in a], [v.get(dim, 0) for v in b]
        n = len(x); ca, cb = Counter(x), Counter(y)
        po = sum(v == w for v, w in zip(x, y)) / n
        pe = sum(ca[v] * cb[v] for v in (-1, 0, 1)) / n**2
        result[dim] = {'n': n, 'agreement': po, 'kappa': (po - pe) / (1 - pe) if pe < 1 else None,
                       'labels_a': dict(ca), 'labels_b': dict(cb)}
    return result


def merge_and_split(rows, public, mapping, bundle_a, bundle_b, packet_sha256, *, split_plan=None):
    validate_candidates(rows, allow_grouped=split_plan is not None)
    if split_plan is not None:
        validate_split_plan(rows, split_plan)
    if (bundle_a.get('vendor', '').strip().casefold(), bundle_a.get('model', '').strip().casefold()) == (
            bundle_b.get('vendor', '').strip().casefold(), bundle_b.get('model', '').strip().casefold()):
        raise ValueError('必须提供两个不同模型的独立标签')
    a = validate_labels(bundle_a, public, packet_sha256)
    b = validate_labels(bundle_b, public, packet_sha256)
    # 映射必须来自相同候选，而非事后把标签换到另一段文本。
    expected_public, expected_mapping = blind_packet(rows, allow_grouped=split_plan is not None)
    if public != expected_public or mapping != expected_mapping:
        raise ValueError('盲标文本或映射已改变')
    original = {r['id']: r for r in rows}
    accepted, rejected = [], []
    for identity in mapping:
        row = original[mapping[identity]]
        entry = {**row, 'dataset': 'D1', 'labels': {'llm_a': a[identity], 'llm_b': b[identity], 'human': None}}
        if a[identity] == b[identity] == row['intended']:
            accepted.append(entry)
        else:
            rejected.append({**entry, 'reason': '独立标签未同时与预期完全一致'})
    stratum = lambda r: next(iter(r['intended'])) + str(next(iter(r['intended'].values()))) if len(r['intended']) == 1 else ('neutral' if not r['intended'] else 'multi')
    quotas = {dim + str(sign): 20 for dim in DIMS for sign in (-1, 1)} | {'neutral': 40, 'multi': 40}
    counts = Counter(stratum(r) for r in accepted)
    complete = counts == Counter(quotas)
    if split_plan is not None:
        planned = {r['id']: r['split'] for r in split_plan['slots']}
        for row in accepted:
            row['split'] = planned[row['id']]
    elif complete:
        for key, total in quotas.items():
            group = sorted((r for r in accepted if stratum(r) == key), key=lambda r: r['id'])
            random.Random('D1-v1-seed0-' + key).shuffle(group)
            for i, row in enumerate(group):
                row['split'] = 'train' if i < total * .6 else ('dev' if i < total * .8 else 'test')
    return {'accepted': accepted, 'rejected': rejected, 'counts': dict(counts), 'quotas': quotas,
            'quota_complete': complete, 'formal_ready': False, 'human_review_complete': False,
            'agreement': agreement(list(a.values()), [b[k] for k in a]),
            'note': '机械检查与两模型一致不等于人工盲审完成；多维逐维标签分布须另报。'}


def validate_split_plan(rows, plan):
    """分组版只接受生成前固定的计划，不在看到标签后移动样本。"""
    from .d1_plan import make_plan
    if plan != make_plan():
        raise ValueError('槽位计划与冻结的D1-v1不一致')
    slots = {s['id']: s for s in plan['slots']}
    for row in rows:
        slot = slots.get(row['id'])
        if slot is None or row.get('group_id') != slot['group_id'] or row['intended'] != slot['intended']:
            raise ValueError('候选身份、预期或情境家族不符合槽位计划')
        if row.get('split', slot['split']) != slot['split']:
            raise ValueError('候选split与预先计划冲突')
