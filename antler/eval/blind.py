"""C9：D5分层盲评及延迟复标；只处理已有回答，绝不生成作者评分。"""
import csv
from datetime import datetime, timedelta, timezone
import json
import random
from pathlib import Path

from .calibration import spearman
from .human import _score, sha256
from antler.traits.data import BIG_FIVE

FIELDS = ['id', 'target', 'question', 'answer', 'trait_human', 'coherence_human']
METRICS = ['trait', 'coherence']


def utcnow():
    return datetime.now(timezone.utc)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_pool(paths):
    """兼容calibration样本和C17已补连贯分的提取样本，保留原始来源。"""
    pool, hashes = [], {}
    for source in paths:
        path = Path(source).resolve()
        if str(path) in hashes:
            raise ValueError('输入文件重复')
        hashes[str(path)] = sha256(path)
        for line_index, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines()):
            if not line.strip(): continue
            record = json.loads(line)
            if 'result' in record:
                dim, pole, coef = record['dim'], record['pole'], record['coef']
                if dim not in BIG_FIVE or pole not in ('high', 'low'):
                    raise ValueError('无效维度或测量极点')
                target = BIG_FIVE[dim][pole == 'low']
                condition = {k: record.get(k) for k in ('layer', 'dim', 'pole', 'coef', 'axis', 'split', 'seed')}
                for index, s in enumerate(record['result']['samples']):
                    pool.append({'target': target, 'question': s['q'], 'answer': s['a'],
                                 'judge': {m: _score(s.get(m), optional=True) for m in METRICS},
                                 'stratum': [dim, str(coef)], 'condition': condition,
                                 'source': str(path), 'line_index': line_index, 'sample_index': index})
            elif {'pole', 'source_index', 'trait_score', 'coherence', 'question', 'answer'} <= set(record):
                target = record['pole']
                dims = [d for d, poles in BIG_FIVE.items() if target in poles]
                if len(dims) != 1 or record.get('side') not in ('pos', 'neg'):
                    raise ValueError('无效提取极点或组别')
                # 提取回答没有注入系数；分别保留pos/neg层，不伪造系数0。
                pool.append({'target': target, 'question': record['question'], 'answer': record['answer'],
                             'judge': {'trait': _score(record['trait_score'], optional=True),
                                       'coherence': _score(record['coherence'], optional=True)},
                             'stratum': [dims[0], 'extract_'+record['side']],
                             'condition': {'origin': 'extract', 'pole': target, 'side': record['side']},
                             'source': str(path), 'line_index': line_index, 'sample_index': None})
            else:
                raise ValueError('输入须为calibration样本或C17逐条补分JSONL')
    for entry in pool:
        if any(not isinstance(entry[k], str) or not entry[k] for k in ('question', 'answer')):
            raise ValueError('问题及回答必须是非空文本')
    if any(sha256(p) != h for p, h in hashes.items()):
        raise RuntimeError('读取时输入发生变化')
    return pool, hashes


def _write_table(path, mapping):
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for identity, entry in mapping.items():
            writer.writerow({'id': identity, **{k: entry[k] for k in FIELDS[1:4]},
                             'trait_human': '', 'coherence_human': ''})


def export_first(paths, out, seed=0):
    """固定100条；轮流从维度×系数层抽取，随机化后产生与条件无关的ID。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    pool, hashes = read_pool(paths)
    groups = {}
    for entry in pool: groups.setdefault(tuple(entry['stratum']), []).append(entry)
    if not groups or len(groups) > 100 or {s[0] for s in groups} != set(BIG_FIVE):
        raise ValueError('D5需要五维且最多100个分层；每层至少一条')
    rng = random.Random(seed)
    strata = sorted(groups)
    rng.shuffle(strata)
    for group in groups.values(): rng.shuffle(group)
    selected, used, counts = [], set(), {s: 0 for s in strata}
    while len(selected) < 100:
        added = False
        for stratum in strata:
            group = groups[stratum]
            while group:
                entry = group.pop()
                identity = (entry['target'], entry['question'], entry['answer'])
                if identity in used: continue
                used.add(identity); selected.append(entry); counts[stratum] += 1
                added = True
                break
            if len(selected) == 100: break
        if not added: raise ValueError('去除相同目标/问题/回答后不足100条')
    if any(n == 0 for n in counts.values()):
        raise ValueError('去重后有分层无样本，不能静默省略该层')
    rng.shuffle(selected)
    mapping = {f'H{i+1:03}': entry for i, entry in enumerate(selected)}
    # 在作者开始标注前固定20条复标身份；只存私有映射，7天后才导出表。
    repeat_ids = rng.sample(list(mapping), 20)
    key = {'schema_version': 1, 'seed': seed, 'created_at': utcnow().isoformat(),
           'source_hashes': hashes, 'samples': mapping, 'repeat_ids': repeat_ids,
           'strata_counts': [{'stratum': list(s), 'n': n} for s,n in counts.items()]}
    write_json(out/'private_key.json', key)
    _write_table(out/'blind.csv', mapping)
    (out/'instructions.md').write_text(
        '# 人格和连贯性盲评\n\n仅打开blind.csv；完成首次和复标前均不要打开private_key.json或评分汇总。\n\n'
        'target列指定要评价的特质，不表示实验注入方向。人格分：0表示完全不体现该特质，100表示强烈体现。'
        '连贯性分：评价表达能否理解、逻辑是否一致、是否出现明显胡言乱语；不要因为人格不讨喜而扣连贯分。'
        '两项都填0–100，保留ID、target、问题和回答，可添加备注。\n\n'
        '首次标注导入完整后开始计时，至少7天后导出20条复标。复标不显示旧分及原ID。\n',
        encoding='utf-8', newline='\n')
    return key


def _labels(path, mapping):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not set(FIELDS) <= set(reader.fieldnames or []): raise ValueError('盲表缺列')
        rows = list(reader)
    if len(rows) != len(mapping) or {r['id'] for r in rows} != set(mapping):
        raise ValueError('盲表ID重复、缺失或变化')
    result = {}
    for row in rows:
        expected = mapping[row['id']]
        if any(row[k] != expected[k] for k in FIELDS[1:4]): raise ValueError('盲表内容身份变化')
        result[row['id']] = {m: _score(row[m+'_human'], optional=True) for m in METRICS}
    return result


def _agreement(first, second):
    result = {}
    for metric in METRICS:
        pairs = [(first[i][metric], second[i][metric]) for i in first
                 if first[i][metric] is not None and second[i][metric] is not None]
        result[metric] = {'n_paired': len(pairs), 'n_total': len(first),
                          'spearman': spearman([p[0] for p in pairs], [p[1] for p in pairs])}
    return result


def import_first(bundle, csv_path):
    bundle = Path(bundle); key_path = bundle/'private_key.json'; key = read_json(key_path)
    _validate_key(key)
    human = _labels(csv_path, key['samples'])
    judge = {i: s['judge'] for i,s in key['samples'].items()}
    result = {'agreement': _agreement(human, judge), 'formal_e0_pass': False}
    complete = all(v is not None for scores in human.values() for v in scores.values())
    result['complete_human'] = complete
    result['human_judge_threshold_met'] = complete and all(
        r['n_paired'] == 100 and r['spearman'] is not None and r['spearman'] >= .7
        for r in result['agreement'].values())
    if complete:
        receipt_path = bundle/'private_first_receipt.json'
        receipt = {'completed_at': utcnow().isoformat(), 'key_sha256': sha256(key_path),
                   'csv_sha256': sha256(csv_path), 'labels': human}
        if receipt_path.exists():
            prior = read_json(receipt_path)
            if prior['key_sha256'] != receipt['key_sha256'] or prior['labels'] != human:
                raise ValueError('已封存首次标注；不能覆盖或沿用旧复标计时')
        else: write_json(receipt_path, receipt)
    return result


def _validate_key(key):
    if key.get('schema_version') != 1 or len(key.get('samples', {})) != 100:
        raise ValueError('无效D5映射')
    if len(key.get('repeat_ids', [])) != 20 or len(set(key['repeat_ids'])) != 20 or not set(key['repeat_ids']) <= set(key['samples']):
        raise ValueError('复标样本身份无效')
    identities={(s['target'],s['question'],s['answer']) for s in key['samples'].values()}
    if len(identities)!=100:raise ValueError('首次盲评含重复样本身份')
    if any(sha256(p) != h for p,h in key['source_hashes'].items()):
        raise ValueError('原始来源哈希已改变')
    pool, _ = read_pool(key['source_hashes'])
    originals = {(s['source'],s['line_index'],s['sample_index']):s for s in pool}
    for sample in key['samples'].values():
        locator=(sample['source'],sample['line_index'],sample['sample_index'])
        if originals.get(locator) != sample: raise ValueError('映射样本与原始来源不一致')


def export_repeat(bundle, out):
    bundle = Path(bundle); key_path=bundle/'private_key.json'
    key, receipt = read_json(key_path), read_json(bundle/'private_first_receipt.json')
    if sha256(key_path) != receipt['key_sha256']: raise ValueError('首次标注后映射已改变')
    due = datetime.fromisoformat(receipt['completed_at']) + timedelta(days=7)
    if utcnow() < due: raise ValueError('距离完整首次标注导入尚不足7天')
    identities = list(key['repeat_ids'])
    random.Random(key['seed']+1).shuffle(identities)
    mapping = {f'R{i+1:03}': {**key['samples'][original], 'first_id': original}
               for i, original in enumerate(identities)}
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    _write_table(out/'repeat.csv', mapping)
    write_json(out/'private_repeat_key.json', {'samples': mapping, 'bundle': str(bundle.resolve()),
               'first_receipt_sha256': sha256(bundle/'private_first_receipt.json'),
               'exported_at': utcnow().isoformat()})
    return {'n_samples': 20, 'not_before': due.isoformat()}


def import_repeat(repeat_key, csv_path):
    key = read_json(repeat_key); bundle=Path(key['bundle'])
    receipt_path=bundle/'private_first_receipt.json'
    if sha256(receipt_path) != key['first_receipt_sha256']: raise ValueError('首次标注封存记录变化')
    receipt=read_json(receipt_path)
    first_key=read_json(bundle/'private_key.json')
    if sha256(bundle/'private_key.json') != receipt['key_sha256']:
        raise ValueError('首次映射变化')
    samples=key['samples']
    if len(samples)!=20 or {s['first_id'] for s in samples.values()}!=set(first_key['repeat_ids']):
        raise ValueError('复标抽样身份变化')
    for sample in samples.values():
        if {k:v for k,v in sample.items() if k!='first_id'}!=first_key['samples'][sample['first_id']]:
            raise ValueError('复标内容与首次样本不一致')
    due=datetime.fromisoformat(receipt['completed_at'])+timedelta(days=7)
    if datetime.fromisoformat(key['exported_at']) < due or utcnow() < due:
        raise ValueError('复标间隔不足7天')
    repeat=_labels(csv_path,key['samples'])
    first={i:receipt['labels'][s['first_id']] for i,s in key['samples'].items()}
    return {'test_retest':_agreement(first,repeat),
            'complete_human':all(v is not None for s in repeat.values() for v in s.values()),
            'interval_days':(datetime.fromisoformat(key['exported_at'])-datetime.fromisoformat(receipt['completed_at'])).total_seconds()/86400,
            'formal_e0_pass':False}
