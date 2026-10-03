"""C9：先封存助手评分，再揭盲比较；不创建作者标注或复标回执。"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re

from . import blind
from .calibration import spearman
from .human import sha256
from antler.traits.data import BIG_FIVE


IDS = {f'H{i:03d}' for i in range(1, 101)}
SEALED_FILES = ('ratings.json', 'public_snapshot.csv', 'rubric_snapshot.md', 'receipt.json')


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('JSON字段重复')
        result[key] = value
    return result


def _load(raw):
    return json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_unique_object)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _score(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('助手评分必须是0–100的有限数字，不能是布尔值或字符串')
    return value


def _public(raw):
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != blind.FIELDS:
        raise ValueError('公开盲表列必须与原始盲表一致，不能附带条件或评委分数')
    rows = list(reader)
    if len(rows) != 100 or {r['id'] for r in rows} != IDS:
        raise ValueError('公开盲表须恰含H001–H100，不能缺失或重复ID')
    targets = {p for poles in BIG_FIVE.values() for p in poles}
    for row in rows:
        if set(row) != set(blind.FIELDS) or any(row[m+'_human'] != '' for m in blind.METRICS):
            raise ValueError('公开盲表的人填评分列必须为空，且每行列数必须正确')
        if row['target'] not in targets or not all(_text(row[k]) for k in ('question', 'answer')):
            raise ValueError('公开盲表的目标、问题或回答无效')
    if len({tuple(r[k] for k in ('target', 'question', 'answer')) for r in rows}) != 100:
        raise ValueError('公开盲表含重复样本身份')
    return {r['id']: r for r in rows}


def _ratings(value, public_hash):
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1 or value.get('rating_source') != 'assistant'):
        raise ValueError('助手评分来源或版本无效')
    if value.get('author_standard_approved') is not True or not _text(value.get('approval_quote')):
        raise ValueError('需要作者明确批准量尺及非空批准原话')
    if value.get('public_table_sha256') != public_hash or not _hash(value.get('rubric_sha256')):
        raise ValueError('公开盲表或量尺哈希无效')
    raters = value.get('raters')
    if not isinstance(raters, list) or not raters:
        raise ValueError('评分者列表不能为空')
    rater_ids = set()
    for rater in raters:
        if (not isinstance(rater, dict) or not _text(rater.get('id'))
                or rater['id'] in rater_ids or rater.get('private_item_scores_read') is not False
                or not _text(rater.get('exposure_note'))):
            raise ValueError('评分者身份、逐项盲态声明或接触记录无效')
        rater_ids.add(rater['id'])
    rows = value.get('scores')
    if (not isinstance(rows, list) or len(rows) != 100
            or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in rows)
            or {r['id'] for r in rows} != IDS):
        raise ValueError('助手评分须恰含100个对应ID，不能缺失或重复')
    for row in rows:
        for metric in blind.METRICS:
            _score(row.get(metric))
        if not _text(row.get('reason')) or not isinstance(row.get('rater_id'), str) or row['rater_id'] not in rater_ids:
            raise ValueError('每条评分须有非空理由及有效评分者')
    return {r['id']: r for r in rows}


def _unchanged(hashes):
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('运行期间输入文件发生变化')


def seal(bundle, ratings_path, out):
    """只读公开材料，排他创建封存目录；不读取私有映射或源评分。"""
    public_path = Path(bundle) / 'blind.csv'
    rubric_path = Path(bundle) / 'assistant_rubric.md'
    ratings_path, out = Path(ratings_path), Path(out)
    if out.exists():
        raise FileExistsError('封存目录已存在，不能覆盖')
    public_raw, ratings_raw = public_path.read_bytes(), ratings_path.read_bytes()
    public_hash, ratings_hash = _digest(public_raw), _digest(ratings_raw)
    _public(public_raw)
    ratings = _load(ratings_raw)
    _ratings(ratings, public_hash)
    rubric_raw = rubric_path.read_bytes()
    if not _text(rubric_raw.decode('utf-8-sig')) or _digest(rubric_raw) != ratings['rubric_sha256']:
        raise ValueError('公开量尺正文与声明哈希不一致')
    inputs = {str(public_path.resolve()): public_hash, str(ratings_path.resolve()): ratings_hash,
              str(rubric_path.resolve()): _digest(rubric_raw)}
    _unchanged(inputs)
    receipt = {'schema_version': 1, 'rating_source': 'assistant',
               'sealed_at': blind.utcnow().isoformat(), 'n_samples': 100,
               'ratings_sha256': ratings_hash, 'public_table_sha256': public_hash,
               'rubric_sha256': ratings['rubric_sha256'], 'input_hashes': inputs,
               'formal_e0_pass': False}
    receipt['receipt_sha256'] = _digest(_json_bytes(receipt))
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in [('ratings.json', ratings_raw), ('public_snapshot.csv', public_raw),
                      ('rubric_snapshot.md', rubric_raw),
                      ('receipt.json', _json_bytes(receipt))]:
        with (out / name).open('xb') as stream:
            stream.write(raw)
    _unchanged(inputs)
    return receipt


def _sealed(bundle, sealed_dir):
    """所有公开封存检查通过之后，调用者才能打开私钥。"""
    sealed_dir = Path(sealed_dir)
    raw = {name: (sealed_dir / name).read_bytes() for name in SEALED_FILES}
    receipt = _load(raw['receipt.json'])
    if not isinstance(receipt, dict):
        raise ValueError('封存回执格式无效')
    digest = receipt.get('receipt_sha256')
    body = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    if (not _hash(digest) or _digest(_json_bytes(body)) != digest
            or receipt.get('schema_version') != 1 or receipt.get('rating_source') != 'assistant'
            or receipt.get('n_samples') != 100 or receipt.get('formal_e0_pass') is not False):
        raise ValueError('封存回执完整性校验失败')
    if (_digest(raw['ratings.json']) != receipt.get('ratings_sha256')
            or _digest(raw['public_snapshot.csv']) != receipt.get('public_table_sha256')
            or _digest(raw['rubric_snapshot.md']) != receipt.get('rubric_sha256')
            or not _text(raw['rubric_snapshot.md'].decode('utf-8-sig'))):
        raise ValueError('封存文件哈希变化')
    public = _public(raw['public_snapshot.csv'])
    ratings = _load(raw['ratings.json'])
    labels = _ratings(ratings, receipt['public_table_sha256'])
    if ratings['rubric_sha256'] != receipt.get('rubric_sha256'):
        raise ValueError('封存量尺哈希不一致')
    public_path = Path(bundle) / 'blind.csv'
    current = public_path.read_bytes()
    if _digest(current) != receipt['public_table_sha256'] or _public(current) != public:
        raise ValueError('公开盲表与封存原文不一致')
    hashes = {str((sealed_dir / name).resolve()): _digest(data) for name, data in raw.items()}
    hashes[str(public_path.resolve())] = _digest(current)
    return public, ratings, labels, hashes


def _summary(items):
    result = {}
    for metric in blind.METRICS:
        paired = [(r['assistant'][metric], r['judge'][metric]) for r in items
                  if r['judge'][metric] is not None]
        first, second = [p[0] for p in paired], [p[1] for p in paired]
        n = len(paired)
        result[metric] = {'n': n, 'n_total': len(items), 'spearman': spearman(first, second),
                          'mae': sum(abs(a-b) for a, b in paired)/n if n else None,
                          'mean_bias': sum(a-b for a, b in paired)/n if n else None,
                          'assistant_mean': sum(first)/n if n else None,
                          'judge_mean': sum(second)/n if n else None,
                          'assistant_range': [min(first), max(first)] if n else None,
                          'judge_range': [min(second), max(second)] if n else None}
    return result


def compare(bundle, sealed_dir):
    """封存验证后揭盲；差值均为助手减主评委，仅作测量诊断。"""
    bundle = Path(bundle)
    public, ratings, labels, inputs = _sealed(bundle, sealed_dir)
    key_path = bundle / 'private_key.json'
    key_raw = key_path.read_bytes()
    key = _load(key_raw)
    blind._validate_key(key)
    if set(key['samples']) != set(public):
        raise ValueError('私有映射与封存ID不一致')
    inputs[str(key_path.resolve())] = _digest(key_raw)
    inputs.update(key['source_hashes'])
    items = []
    for identity in sorted(public):
        sample, row, label = key['samples'][identity], public[identity], labels[identity]
        if any(sample[k] != row[k] for k in ('target', 'question', 'answer')):
            raise ValueError('私有映射与封存原文身份不一致')
        judge = {m: sample['judge'][m] for m in blind.METRICS}
        for score in judge.values():
            if score is not None:
                _score(score)
        assistant = {m: label[m] for m in blind.METRICS}
        items.append({'id': identity, 'target': row['target'], 'question': row['question'],
                      'answer': row['answer'], 'condition': sample['condition'],
                      'assistant': assistant, 'judge': judge, 'reason': label['reason'],
                      'rater_id': label['rater_id'],
                      'difference': {m: assistant[m]-judge[m] if judge[m] is not None else None
                                     for m in blind.METRICS}})
    by_target, by_condition = {}, {}
    for item in items:
        by_target.setdefault(item['target'], []).append(item)
        condition = item['condition']
        dim = next(d for d, poles in BIG_FIVE.items() if item['target'] in poles)
        group = (dim, condition.get('coef'), condition.get('pole'))
        by_condition.setdefault(group, []).append(item)
    agreement = _summary(items)
    _unchanged(inputs)
    return {'schema_version': 1, 'rating_source': 'assistant', 'n_samples': 100,
            'agreement': agreement,
            'per_target': {target: _summary(group) for target, group in sorted(by_target.items())},
            'per_dim_coef_pole': [{'dim': group[0], 'coef': group[1], 'pole': group[2],
                                  'agreement': _summary(rows)}
                                 for group, rows in sorted(by_condition.items(), key=lambda p: str(p[0]))],
            'items': items, 'raters': ratings['raters'], 'input_hashes': inputs,
            'assistant_judge_threshold': .7,
            'assistant_judge_threshold_met': all(m['n'] == 100 and m['spearman'] is not None
                                                 and m['spearman'] >= .7 for m in agreement.values()),
            'formal_e0_pass': False,
            'scope': '本批按作者授权由助手评分，作者确认尺度；报告助手与主评委一致性，未进行作者逐条独立评分或重测，不单独判定E0通过。均值和范围使用成对有效分，平均偏差为助手减主评委。'}
