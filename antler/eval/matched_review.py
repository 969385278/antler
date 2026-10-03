"""C9：固定200条配对助手盲审，先公开封存再揭盲，不生成或调用评委。"""
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import random
import re
import subprocess

from antler.config import ROOT
from antler.traits.data import BIG_FIVE, load_trait
from .assistant_review import _digest, _json_bytes, _load, _text, _hash, _score, _unchanged, _summary
from .human import sha256
from .stats import paired_bootstrap


IDS = {f'M{i:03d}' for i in range(1, 201)}
METRICS = ('trait', 'coherence')
ARMS = ('zero', 'all', 'decode_base', 'decode_mid', 'decode_upper')
SEEDS = (1, 2, 3)
COEFS = {'C': dict(zip(ARMS, (0., -.175, -.175, -.225, -.300))),
         'N': dict(zip(ARMS, (0., .165, .165, .200, .250)))}
RUN_NAME = '2026-10-03T182340.163154_0000_e41713cc'
RUN_COMMIT = '988e8e22aefc4fd8417b10271c303a559a491011'
RUN_START = '2026-10-03T18:23:40.163349+00:00'
APPROVED_RUBRIC_SHA256 = '3f090db634aa02fddff92d854c9ff947a71bdefc27d196eccbce03ed0fcfabd0'
VECTOR_ROOT = 'outputs/main/refilter_diagnostics/20261003T022259Z_099e5cd2/variants/paired/'
PINNED_INPUTS = {
    'experiments/e0_decode_strength.py': 'd49dca39dbab0647ed80adb4c709fe6c90d37b345287484f166ed899a2a98668',
    'antler/core/generate.py': '0fd2b1d6ae81fd5d20eea4245264c643bb0a9200fe023bda30f5c1c5a363bd04',
    'antler/core/hooks.py': 'c1a8b9f318ea6e516f1136e0d2b920384c3bd58d647beecf4635241ff97a922e',
    VECTOR_ROOT + 'directions.pt': '5e99e6e40abbdcd6f5c3d3833f0d0c24c4d41ef676fc0b0c923afabc4db7cfaa',
    VECTOR_ROOT + 'vectors/careless.pt': 'cc6c27f273e0e78d3ed069ab51ea360f00ea1e76d8ec0bfcf749d003c4e6b539',
    VECTOR_ROOT + 'vectors/nervous.pt': '4caa81abb73a3ee01b6bde595dab19a7f879057d7190d7325468d955b4f9f507',
}
SOURCE_ARTIFACTS = {'run_meta.json', 'samples.jsonl', 'generation_settings.json', 'preflight.json',
                    'generation_batches.jsonl', 'phase_traces.json', 'score_plan.json',
                    'judge_cache.jsonl', 'judge_responses.jsonl', 'assessment.json'}
PUBLIC_FILES = ('public.json', 'assistant_rubric.md', 'trait_definitions.json', 'export_manifest.json')
SEALED_FILES = ('ratings.json', *PUBLIC_FILES, 'receipt.json')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_exclusive(out, files):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        with (out / name).open('xb') as stream:
            stream.write(raw)


def _condition(row):
    return {k: row[k] for k in ('dim', 'pole', 'arm', 'layer', 'axis', 'coef', 'seed',
                              'question_index', 'sample_index', 'split')} | {'phase': row['steer_phase']}


def _identity(row):
    return tuple(row[k] for k in ('dim', 'pole', 'arm', 'seed', 'question_index', 'sample_index'))


def validate_rows(rows, traits):
    """验证全1800条，拒绝缺分、错误极点、test题与身份类型混淆。"""
    expected = {(d, p, a, s, q, i) for d in COEFS for p in ('high', 'low') for a in ARMS
                for s in SEEDS for q in range(10) for i in range(3)}
    actual = []
    for row in rows:
        if not isinstance(row, dict) or any(type(row.get(k)) is not int for k in
                                           ('seed', 'question_index', 'sample_index', 'layer')):
            raise ValueError('源回答的身份字段必须为整数')
        identity = _identity(row)
        if identity not in expected:
            raise ValueError('源回答越过固定矩阵或calib范围')
        actual.append(identity)
        target = BIG_FIVE[row['dim']][row['pole'] == 'low']
        if (row.get('target') != target or row.get('q') != traits[target]['questions'][row['question_index']]
                or row['layer'] != 21 or row.get('axis') != 'pole' or row.get('split') != 'calib'
                or type(row.get('coef')) not in (int, float) or row['coef'] != COEFS[row['dim']][row['arm']]
                or row.get('steer_phase') != ('decode' if row['arm'].startswith('decode') else 'all')
                or not _text(row.get('a'))):
            raise ValueError('源回答条件、题目或原文与登记不符')
        for metric in (*METRICS, 'task'):
            _score(row.get(metric))
    if len(rows) != 1800 or len(set(actual)) != 1800 or set(actual) != expected:
        raise ValueError('源回答须完整保留1800条唯一身份')


def select_rows(rows):
    """只访问身份，使用预登记的两个独立随机数生成器，不读取任何分数。"""
    indexed = {_identity(r): (i, r) for i, r in enumerate(rows)}
    rng, selected = random.Random(20261004), []
    for g, (dim, pole) in enumerate(itertools.product(('C', 'N'), ('high', 'low'))):
        questions = list(range(10))
        rng.shuffle(questions)
        for rank, q in enumerate(questions):
            seed, sample = SEEDS[(rank + g) % 3], rng.randrange(3)
            selected.extend(indexed[dim, pole, arm, seed, q, sample] for arm in ARMS)
    random.Random(20261005).shuffle(selected)
    return [(f'M{i:03d}', index, row) for i, (index, row) in enumerate(selected, 1)]


def _verify_recorded_inputs(recorded):
    """绑定登记的核心哈希；能本地取得的文件用当前文件或锁定commit对象核验。"""
    if not isinstance(recorded, dict) or not recorded or any(not _hash(h) for h in recorded.values()):
        raise ValueError('源运行输入哈希清单无效')
    normalized = {p.replace('\\', '/'): h for p, h in recorded.items()}
    for relative, expected in PINNED_INPUTS.items():
        matches = [digest for name, digest in normalized.items() if name.endswith('/' + relative)]
        if matches != [expected]:
            raise ValueError('源运行的固定源码或向量哈希不符')
    evidence, local_hashes = [], {}
    for original, digest in normalized.items():
        relative = None
        for marker in ('experiments/', 'configs/', 'docs/', 'data/', 'models/', 'outputs/', 'antler/'):
            token = '/' + marker
            if token in original:
                relative = marker + original.rsplit(token, 1)[1]
                break
        if relative is None or '..' in Path(relative).parts:
            evidence.append({'source': original, 'status': 'unavailable_locally'})
            continue
        path = ROOT / relative
        if path.is_file() and sha256(path) == digest:
            evidence.append({'source': original, 'status': 'local_file', 'local': str(path)})
            local_hashes[str(path.resolve())] = digest
            continue
        if relative.startswith(('configs/', 'models/', 'outputs/')) and relative not in PINNED_INPUTS:
            # 服务器配置可含已授权的暂存差异；不存在同一副本时只保留源记录，不冒称本地核验。
            evidence.append({'source': original, 'status': 'source_recorded_no_matching_local_copy',
                             'relative': relative})
            continue
        raw = None
        try:
            raw = subprocess.check_output(['git', 'show', RUN_COMMIT + ':' + relative], cwd=ROOT,
                                          stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            pass
        if raw is not None:
            if _digest(raw) != digest:
                raise ValueError('源运行输入与锁定commit文件不一致')
            evidence.append({'source': original, 'status': 'locked_git_object', 'relative': relative})
        elif path.is_file():
            raise ValueError('可取得的本地输入与源运行哈希不一致')
        else:
            evidence.append({'source': original, 'status': 'unavailable_locally'})
    return evidence, local_hashes


def verify_source(source_dir):
    """只接受指定完成轮；聚合assessment仅核验字节哈希，不读取其内容来选样。"""
    source = Path(source_dir).resolve()
    if source.name != RUN_NAME or source.parent.name != 'decode_strength':
        raise ValueError('本轮只允许登记的唯一源目录')
    manifest_path = source / 'artifact_hashes.json'
    manifest_raw = manifest_path.read_bytes()
    manifest = _load(manifest_raw)
    if not isinstance(manifest, dict) or set(manifest) != SOURCE_ARTIFACTS:
        raise ValueError('源产物清单不完整')
    inputs = {str(manifest_path): _digest(manifest_raw)}
    for name, digest in manifest.items():
        if not _hash(digest) or sha256(source / name) != digest:
            raise ValueError('源产物哈希校验失败')
        inputs[str(source / name)] = digest
    meta = _load((source / 'run_meta.json').read_bytes())
    expected = {'task': 'C19', 'status': 'completed', 'stage': 'completed', 'git_commit': RUN_COMMIT,
                'start_time': RUN_START, 'layer': 21, 'seeds': list(SEEDS), 'arms': list(ARMS),
                'coefs': COEFS, 'split': 'calib', 'expected_samples': 1800, 'generated': 1800,
                'n_samples_per_question': 3, 'max_new_tokens': 320, 'batch_size': 16,
                'temperature': 1., 'top_p': 1., 'preflight_samples': 8, 'preflight_tokens': 16,
                'max_score_inputs': 5400, 'score_shuffle_seed': 0, 'formal_e0_pass': False}
    if (any(type(meta.get(k)) is not type(v) or meta[k] != v for k, v in expected.items())
            or not _text(meta.get('end_time')) or meta.get('judge', {}).get('model') != 'deepseek-flash'):
        raise ValueError('源运行身份或完成状态与固定登记不符')
    evidence, local_hashes = _verify_recorded_inputs(meta.get('data_hashes'))
    inputs.update(local_hashes)
    traits = {p: load_trait(p, 'eval') for dim in COEFS for p in BIG_FIVE[dim]}
    raw = (source / 'samples.jsonl').read_bytes()
    rows = [_load(line) for line in raw.splitlines() if line.strip()]
    validate_rows(rows, traits)
    _unchanged(inputs)
    return {'rows': rows, 'traits': traits, 'input_hashes': inputs, 'input_verification': evidence,
            'source_samples_sha256': _digest(raw), 'source_manifest_sha256': _digest(manifest_raw),
            'source_provenance': {'directory': str(source), 'git_commit': RUN_COMMIT, 'start_time': RUN_START}}


def export(source_dir, rubric_path, out):
    """排他导出公开表和私有映射，原量尺逐字节复制，不覆盖旧批。"""
    if Path(out).exists():
        raise FileExistsError('导出目录已存在，不能覆盖')
    source = verify_source(source_dir)
    rubric_path = Path(rubric_path).resolve()
    rubric = rubric_path.read_bytes()
    if not _text(rubric.decode('utf-8-sig')) or _digest(rubric) != APPROVED_RUBRIC_SHA256:
        raise ValueError('量尺须保留作者已批准历史附件的原字节')
    inputs = {**source['input_hashes'], str(rubric_path): _digest(rubric)}
    public, mapping = [], {}
    for identity, index, row in select_rows(source['rows']):
        item = {'id': identity, 'target': row['target'], 'question': row['q'], 'answer': row['a']}
        public.append(item)
        mapping[identity] = {**item, 'condition': _condition(row),
            'source': {'path': str(Path(source_dir).resolve() / 'samples.jsonl'), 'row_index': index,
                       'sha256': source['source_samples_sha256'], 'row_sha256': _digest(_json_bytes(row))}}
    key = {'schema_version': 1, 'n_samples': 200, 'source_provenance': source['source_provenance'],
           'source_samples_sha256': source['source_samples_sha256'],
           'source_manifest_sha256': source['source_manifest_sha256'], 'samples': mapping,
           'input_hashes': inputs, 'input_verification': source['input_verification']}
    files = {'public.json': _json_bytes(public), 'assistant_rubric.md': rubric,
             'trait_definitions.json': _json_bytes({p: t['eval_prompt'] for p, t in source['traits'].items()}),
             'private_key.json': _json_bytes(key)}
    manifest = {'schema_version': 1, 'n_samples': 200, 'rating_source': 'assistant', 'exported_at': _now(),
                'source_samples_sha256': source['source_samples_sha256'],
                'source_manifest_sha256': source['source_manifest_sha256'],
                'hashes': {name: _digest(raw) for name, raw in files.items()}, 'formal_e0_pass': False}
    manifest['manifest_sha256'] = _digest(_json_bytes(manifest))
    files['export_manifest.json'] = _json_bytes(manifest)
    _unchanged(inputs)
    _write_exclusive(out, files)
    _unchanged(inputs)
    return {'n_samples': 200, 'bundle': str(Path(out).resolve()), 'input_hashes': inputs,
            'source_provenance': source['source_provenance'], 'source_samples_sha256': source['source_samples_sha256'],
            'selection_seed': 20261004, 'shuffle_seed': 20261005, 'formal_e0_pass': False}


def _public(raw):
    rows = _load(raw)
    if (not isinstance(rows, list) or len(rows) != 200 or any(not isinstance(r, dict) for r in rows)
            or any(set(r) != {'id', 'target', 'question', 'answer'} for r in rows)
            or {r['id'] for r in rows} != IDS):
        raise ValueError('公开表须恰含M001–M200及四个允许字段')
    targets = {p for d in COEFS for p in BIG_FIVE[d]}
    if any(r['target'] not in targets or not all(_text(r[k]) for k in ('question', 'answer')) for r in rows):
        raise ValueError('公开目标、问题或回答无效')
    # 同文不同条件保留不同ID，此处有意不按文本去重。
    return {r['id']: r for r in rows}


def _public_bundle(raw):
    public = _public(raw['public.json'])
    manifest = _load(raw['export_manifest.json'])
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if (manifest.get('schema_version') != 1 or manifest.get('n_samples') != 200
            or manifest.get('rating_source') != 'assistant' or manifest.get('formal_e0_pass') is not False
            or _digest(_json_bytes(body)) != manifest.get('manifest_sha256')
            or not _hash(manifest.get('source_samples_sha256')) or not _hash(manifest.get('source_manifest_sha256'))
            or set(manifest.get('hashes', {})) != {'public.json', 'assistant_rubric.md', 'trait_definitions.json', 'private_key.json'}
            or not _hash(manifest['hashes']['private_key.json'])):
        raise ValueError('公开导出清单完整性校验失败')
    for name in ('public.json', 'assistant_rubric.md', 'trait_definitions.json'):
        if _digest(raw[name]) != manifest['hashes'][name]:
            raise ValueError('公开材料与导出哈希不一致')
    if _digest(raw['assistant_rubric.md']) != APPROVED_RUBRIC_SHA256:
        raise ValueError('公开量尺不是已批准历史附件的原字节')
    definitions = _load(raw['trait_definitions.json'])
    if (not isinstance(definitions, dict) or set(definitions) != {r['target'] for r in public.values()}
            or not all(_text(v) for v in definitions.values()) or not _text(raw['assistant_rubric.md'].decode('utf-8-sig'))):
        raise ValueError('公开定义或量尺无效')
    return public, manifest


def _ratings(value, public_hash, rubric_hash):
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('rating_source') != 'assistant' or value.get('author_standard_approved') is not True
            or not _text(value.get('approval_quote'))):
        raise ValueError('必须记录助手来源、作者批准尺度及批准原话')
    if value.get('public_table_sha256') != public_hash or value.get('rubric_sha256') != rubric_hash:
        raise ValueError('评分声明与公开材料哈希不一致')
    raters = value.get('raters')
    if not isinstance(raters, list) or not raters:
        raise ValueError('评分者及实际曝光记录不能为空')
    ids = set()
    for rater in raters:
        if (not isinstance(rater, dict) or not _text(rater.get('id')) or rater['id'] in ids
                or rater.get('fresh_context') is not True or rater.get('private_item_scores_read') is not False
                or rater.get('private_item_conditions_read') is not False or not _text(rater.get('exposure_note'))):
            raise ValueError('评分者须记录新上下文、逐条条件/评分盲态及实际曝光')
        ids.add(rater['id'])
    rows = value.get('scores')
    if (not isinstance(rows, list) or len(rows) != 200 or any(not isinstance(r, dict) for r in rows)
            or any(set(r) != {'id', 'trait', 'coherence', 'reason', 'rater_id'} for r in rows)
            or {r['id'] for r in rows} != IDS):
        raise ValueError('评分须完整包含200个对应ID及规定字段')
    for row in rows:
        for metric in METRICS:
            _score(row[metric])
        if (not _text(row['reason']) or re.search('[\u3400-\u9fff]', row['reason']) is None
                or not isinstance(row['rater_id'], str) or row['rater_id'] not in ids):
            raise ValueError('每条评分须有中文理由及有效评分者')
    return {r['id']: r for r in rows}


def seal(bundle, ratings_path, out):
    """只读取公开材料和助手评分；在读取任何私有文件前排他封存。"""
    if Path(out).exists():
        raise FileExistsError('封存目录已存在，不能覆盖')
    bundle, ratings_path = Path(bundle), Path(ratings_path)
    raw = {name: (bundle / name).read_bytes() for name in PUBLIC_FILES}
    _, manifest = _public_bundle(raw)
    ratings_raw = ratings_path.read_bytes()
    _ratings(_load(ratings_raw), manifest['hashes']['public.json'], manifest['hashes']['assistant_rubric.md'])
    inputs = {str((bundle / name).resolve()): _digest(value) for name, value in raw.items()}
    inputs[str(ratings_path.resolve())] = _digest(ratings_raw)
    files = {**raw, 'ratings.json': ratings_raw}
    receipt = {'schema_version': 1, 'rating_source': 'assistant', 'n_samples': 200, 'sealed_at': _now(),
               'hashes': {name: _digest(value) for name, value in files.items()},
               'input_hashes': inputs, 'formal_e0_pass': False}
    receipt['receipt_sha256'] = _digest(_json_bytes(receipt))
    files['receipt.json'] = _json_bytes(receipt)
    _unchanged(inputs)
    _write_exclusive(out, files)
    _unchanged(inputs)
    return receipt


def _sealed(bundle, sealed_dir):
    """公开封存与当前公开材料全部验证之后，compare才可读取私有映射。"""
    sealed_dir, bundle = Path(sealed_dir), Path(bundle)
    raw = {name: (sealed_dir / name).read_bytes() for name in SEALED_FILES}
    receipt = _load(raw['receipt.json'])
    body = {k: v for k, v in receipt.items() if k != 'receipt_sha256'}
    if (receipt.get('schema_version') != 1 or receipt.get('rating_source') != 'assistant'
            or receipt.get('n_samples') != 200 or receipt.get('formal_e0_pass') is not False
            or _digest(_json_bytes(body)) != receipt.get('receipt_sha256')
            or set(receipt.get('hashes', {})) != set(SEALED_FILES) - {'receipt.json'}):
        raise ValueError('封存回执完整性校验失败')
    for name, digest in receipt['hashes'].items():
        if _digest(raw[name]) != digest:
            raise ValueError('封存文件哈希发生变化')
    public, manifest = _public_bundle(raw)
    ratings = _load(raw['ratings.json'])
    labels = _ratings(ratings, manifest['hashes']['public.json'], manifest['hashes']['assistant_rubric.md'])
    inputs = {str((sealed_dir / name).resolve()): _digest(value) for name, value in raw.items()}
    for name in PUBLIC_FILES:
        path = bundle / name
        if sha256(path) != _digest(raw[name]):
            raise ValueError('当前公开材料与封存原文不一致')
        inputs[str(path.resolve())] = _digest(raw[name])
    return public, manifest, ratings, labels, inputs


def _agreement(items):
    summary = _summary(items)
    for metric in METRICS:
        summary[metric]['distributions'] = {side: [{'score': score, 'count': count}
            for score, count in sorted(Counter(r[side][metric] for r in items).items())]
            for side in ('assistant', 'judge', 'difference')}
    return summary


def _paired(items):
    reports = []
    for target in sorted({r['target'] for r in items}):
        groups = {arm: sorted([r for r in items if r['target'] == target and r['condition']['arm'] == arm],
                              key=lambda r: r['condition']['question_index']) for arm in ARMS}
        for arm in ARMS[1:]:
            for reference in (('zero',) if arm == 'all' else ('zero', 'all')):
                a, b = groups[reference], groups[arm]
                identities = lambda rows: [(r['condition']['question_index'], r['condition']['seed'],
                                             r['condition']['sample_index']) for r in rows]
                if len(a) != 10 or len(b) != 10 or identities(a) != identities(b):
                    raise ValueError('配对条件未保持相同的十道题、种子和采样')
                ids = [r['condition']['question_index'] for r in a]
                reports.append({'target': target, 'arm': arm, 'reference': reference, 'n_questions': 10,
                    'n_samples_per_question': 1, 'formal_e0_pass': False,
                    'scores': {side: {metric: paired_bootstrap([r[side][metric] for r in a], ids,
                        [r[side][metric] for r in b], ids, n_boot=10000, seed=0)
                        for metric in METRICS} for side in ('assistant', 'judge')},
                    'difference_in_differences': {metric: paired_bootstrap(
                        [r['assistant'][metric] - r['judge'][metric] for r in a], ids,
                        [r['assistant'][metric] - r['judge'][metric] for r in b], ids,
                        n_boot=10000, seed=0) for metric in METRICS},
                    'interval_note': '均为逐点题目配对区间，非同时区间；差中差为助手条件效应减主评委条件效应，不设新门槛。'})
    return reports


def compare(bundle, sealed_dir):
    """验证封存后关联唯一源记录；保留全部条件、同文差异和两套题目配对结果。"""
    bundle = Path(bundle)
    public, manifest, ratings, labels, inputs = _sealed(bundle, sealed_dir)
    key_path = bundle / 'private_key.json'
    key_raw = key_path.read_bytes()
    if _digest(key_raw) != manifest['hashes']['private_key.json']:
        raise ValueError('私有映射哈希发生变化')
    key = _load(key_raw)
    if (key.get('schema_version') != 1 or key.get('n_samples') != 200 or set(key.get('samples', {})) != IDS
            or key.get('source_samples_sha256') != manifest['source_samples_sha256']
            or key.get('source_manifest_sha256') != manifest['source_manifest_sha256']):
        raise ValueError('私有映射身份或源绑定无效')
    inputs[str(key_path.resolve())] = _digest(key_raw)
    source = verify_source(key['source_provenance']['directory'])
    if (source['source_provenance'] != key['source_provenance']
            or source['source_samples_sha256'] != manifest['source_samples_sha256']
            or source['source_manifest_sha256'] != manifest['source_manifest_sha256']):
        raise ValueError('最终源样本或源清单已改变')
    inputs.update(source['input_hashes'])
    items = []
    for identity, index, row in select_rows(source['rows']):
        entry, shown, label = key['samples'][identity], public[identity], labels[identity]
        expected = {'id': identity, 'target': row['target'], 'question': row['q'], 'answer': row['a']}
        locator = {'path': str(Path(key['source_provenance']['directory']) / 'samples.jsonl'),
                   'row_index': index, 'sha256': source['source_samples_sha256'], 'row_sha256': _digest(_json_bytes(row))}
        if shown != expected or entry != {**expected, 'condition': _condition(row), 'source': locator}:
            raise ValueError('私有映射、条件身份或原文与固定选样不一致')
        assistant = {m: label[m] for m in METRICS}
        judge = {m: row[m] for m in METRICS}
        items.append({**expected, 'condition': _condition(row), 'source': locator, 'assistant': assistant,
                      'judge': judge, 'difference': {m: assistant[m] - judge[m] for m in METRICS},
                      'reason': label['reason'], 'rater_id': label['rater_id']})
    duplicates = []
    same_text = {}
    for item in items:
        same_text.setdefault(tuple(item[k] for k in ('target', 'question', 'answer')), []).append(item)
    for group in same_text.values():
        if len(group) < 2:
            continue
        duplicates.append({'ids': [r['id'] for r in group], 'target': group[0]['target'],
            'question': group[0]['question'], 'answer': group[0]['answer'],
            'pairs': [{'left': a['id'], 'right': b['id'],
                       'assistant_difference': {m: a['assistant'][m] - b['assistant'][m] for m in METRICS},
                       'judge_difference': {m: a['judge'][m] - b['judge'][m] for m in METRICS}}
                      for a, b in itertools.combinations(group, 2)]})
    agreement = _agreement(items)
    result = {'schema_version': 1, 'rating_source': 'assistant', 'n_samples': 200, 'n_question_groups': 40,
        'source_provenance': source['source_provenance'], 'source_samples_sha256': source['source_samples_sha256'],
        'agreement': agreement,
        'per_target': {t: _agreement([r for r in items if r['target'] == t]) for t in sorted({r['target'] for r in items})},
        'per_condition': [{'target': t, 'arm': arm, 'agreement': _agreement([r for r in items
                           if r['target'] == t and r['condition']['arm'] == arm])}
                          for t in sorted({r['target'] for r in items}) for arm in ARMS],
        'paired_comparisons': _paired(items), 'identical_text_groups': duplicates, 'items': items,
        'raters': ratings['raters'], 'author_standard_approved': ratings['author_standard_approved'],
        'approval_quote': ratings['approval_quote'], 'input_hashes': inputs,
        'source_input_verification': source['input_verification'], 'formal_e0_pass': False,
        'assistant_judge_threshold': .7,
        'assistant_judge_threshold_met': all(v['spearman'] is not None and v['spearman'] >= .7 for v in agreement.values()),
        'scope': '仅助手按作者批准尺度评分，不是独立人工标注。每题每臂一条，覆盖种子不等于每题三个种子；'
                 '200条仅40个题目组，条件比较每套10题。全1800条生成统计保持原轮。旧D5不合并，'
                 '不新增评分调用或test，不修改正式门槛，不单独判定E0通过。'}
    _unchanged(inputs)
    return result
