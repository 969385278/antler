"""C9：核验任务盲标身份并计算作者与评委的一致性，不生成任何人工分数。"""
import csv
import hashlib
import json
import math
from pathlib import Path

from .calibration import spearman


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _score(value, *, optional=False):
    if optional and (value is None or value == ''):
        return None
    if isinstance(value, bool):
        raise ValueError('评分必须是0–100的有限数字')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('评分必须是0–100的有限数字') from exc
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError('评分必须是0–100的有限数字')
    return number


def import_task_labels(csv_path, key_path, source_root=None):
    """只读导入C18导出的30条task盲表；缺分与不一致均不能验证评委。"""
    csv_path, key_path = Path(csv_path).resolve(), Path(key_path).resolve()
    root = Path(source_root).resolve() if source_root else key_path.parent
    hashes = {str(p): sha256(p) for p in (csv_path, key_path)}
    key = json.loads(key_path.read_text(encoding='utf-8-sig'))
    mapping = key.get('samples')
    if key.get('schema_version') != 1 or not isinstance(mapping, dict) or not mapping:
        raise ValueError('不支持的盲标映射格式')
    if key.get('n_samples') != len(mapping):
        raise ValueError('映射样本数与声明不一致')
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'question', 'answer', 'task_human'}.issubset(reader.fieldnames or []):
            raise ValueError('盲表缺少必要列')
        rows = list(reader)
    identities = [row['id'] for row in rows]
    if len(set(identities)) != len(identities) or set(identities) != set(mapping):
        raise ValueError('盲表ID重复、缺失或与映射不一致')
    sources, pairs, missing_human, missing_judge, qa = {}, [], [], [], set()
    for row in rows:
        identity = row['id']
        entry = mapping[identity]
        path = (root / entry['source']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('原始回答路径越出指定根目录')
        if path not in sources:
            hashes[str(path)] = sha256(path)
            sources[path] = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        condition = entry['condition']
        matches = [record for record in sources[path]
                   if all(record.get(k) == condition[k] for k in ('layer', 'dim', 'pole', 'coef'))]
        if len(matches) != 1:
            raise ValueError('原始回答中没有唯一对应的实验条件')
        index = entry['sample_index']
        samples = matches[0]['result']['samples']
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(samples):
            raise ValueError('映射中的样本下标无效')
        original = samples[index]
        if (row['question'], row['answer']) != (original['q'], original['a']):
            raise ValueError('盲表的问题或回答已改变，拒绝关联评分')
        judge = _score(original.get('task'), optional=True)
        if _score(entry.get('task_judge'), optional=True) != judge:
            raise ValueError('映射评委分与原始样本不一致')
        human = _score(row['task_human'], optional=True)
        qa.add((row['question'], row['answer']))
        if human is None: missing_human.append(identity)
        if judge is None: missing_judge.append(identity)
        if human is not None and judge is not None:
            pairs.append((human, judge))
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    complete = not missing_human and not missing_judge
    # 仅部分已填样本的相关性不能验证评委，原定30条必须全部有效。
    eligible = complete and len(rows) == 30
    if any(sha256(p) != digest for p, digest in hashes.items()):
        raise RuntimeError('导入期间输入文件发生变化')
    return {'n_samples': len(rows), 'n_paired': len(pairs), 'n_unique_qa': len(qa),
            'missing_human_ids': missing_human, 'missing_judge_ids': missing_judge,
            'complete': complete, 'spearman': rho, 'threshold': .6,
            'task_judge_validated': eligible and rho is not None and rho >= .6,
            'formal_e0_pass': False, 'input_hashes': hashes,
            'scope': '仅30条任务理解盲评；不替代D5人格/连贯性100条及7天复标'}
