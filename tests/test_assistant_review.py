"""C9：只用合成数据验证助手评分封存、揭盲顺序及诊断统计。"""
import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.eval import assistant_review as review, blind
from antler.eval.human import sha256


@pytest.fixture
def case(tmp_path):
    records = []
    for dim in ['O', 'C', 'E', 'A', 'N']:
        for pole in ['high', 'low']:
            for coef in [0, .2]:
                records.append({'dim': dim, 'pole': pole, 'coef': coef, 'layer': 16,
                                'split': 'calib', 'seed': 0,
                                'result': {'samples': [
                                    {'q': f'{dim}/{pole}/{coef}/q{i}', 'a': f'合成回答{i}',
                                     'trait': i*8, 'coherence': 100-i*8} for i in range(10)]}})
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    bundle = tmp_path / 'bundle'
    key = blind.export_first([source], bundle, seed=0)
    rubric = bundle / 'assistant_rubric.md'
    rubric.write_text('助手按指定特质与逻辑连贯性分别给0至100分。\n', encoding='utf-8')
    ratings = {'schema_version': 1, 'rating_source': 'assistant', 'author_standard_approved': True,
               'approval_quote': '我确认这个评分尺度，继续。',
               'public_table_sha256': sha256(bundle / 'blind.csv'), 'rubric_sha256': sha256(rubric),
               'raters': [{'id': 'assistant-1', 'private_item_scores_read': False,
                           'exposure_note': '仅阅读公开文本和评分量尺。'}],
               'scores': [{'id': identity, 'trait': row['judge']['trait'],
                           'coherence': row['judge']['coherence'], 'reason': '合成理由',
                           'rater_id': 'assistant-1'} for identity, row in key['samples'].items()]}
    path = tmp_path / 'ratings.json'
    blind.write_json(path, ratings)
    return SimpleNamespace(bundle=bundle, key=key, source=source, ratings=ratings,
                           path=path, sealed=tmp_path / 'sealed', root=tmp_path)


def save(case):
    # 特意保留非法NaN，交给被测入口拒绝。
    case.path.write_text(json.dumps(case.ratings, ensure_ascii=False), encoding='utf-8')


def guard_private(case, monkeypatch):
    original = Path.open
    forbidden = {(case.bundle / 'private_key.json').resolve(), case.source.resolve()}

    def guarded(path, *args, **kwargs):
        assert path.resolve() not in forbidden, '封存校验前不应读取私钥或源评分'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)


def test_seal_blind_and_immutable(case, monkeypatch):
    guard_private(case, monkeypatch)
    receipt = review.seal(case.bundle, case.path, case.sealed)
    assert receipt['rating_source'] == 'assistant' and receipt['n_samples'] == 100
    assert not receipt['formal_e0_pass']
    assert (case.sealed / 'ratings.json').read_bytes() == case.path.read_bytes()
    assert (case.sealed / 'public_snapshot.csv').read_bytes() == (case.bundle / 'blind.csv').read_bytes()
    assert (case.sealed / 'rubric_snapshot.md').read_bytes() == (case.bundle / 'assistant_rubric.md').read_bytes()
    original = (case.sealed / 'receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        review.seal(case.bundle, case.path, case.sealed)
    assert (case.sealed / 'receipt.json').read_bytes() == original
    assert not (case.bundle / 'private_first_receipt.json').exists()


@pytest.mark.parametrize('value', [True, False, None, '50', float('nan'), float('inf'), -1, 101])
def test_invalid_score_rejected(case, value):
    case.ratings['scores'][0]['trait'] = value
    save(case)
    with pytest.raises(ValueError, match='有限数字'):
        review.seal(case.bundle, case.path, case.sealed)
    assert not case.sealed.exists()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'unknown'])
def test_invalid_score_ids(case, change):
    rows = case.ratings['scores']
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows[1]['id'] = rows[0]['id']
    elif change == 'extra': rows.append(dict(rows[0]))
    else: rows[0]['id'] = 'H999'
    save(case)
    with pytest.raises(ValueError, match='对应ID'):
        review.seal(case.bundle, case.path, case.sealed)


@pytest.mark.parametrize('change', ['approval', 'quote', 'source', 'public_hash', 'rubric_hash',
                                   'rater', 'rater_duplicate', 'private_read', 'exposure', 'reason'])
def test_provenance_validation(case, change):
    ratings = case.ratings
    if change == 'approval': ratings['author_standard_approved'] = 1
    elif change == 'quote': ratings['approval_quote'] = ' '
    elif change == 'source': ratings['rating_source'] = 'human'
    elif change == 'public_hash': ratings['public_table_sha256'] = '0'*64
    elif change == 'rubric_hash': ratings['rubric_sha256'] = '0'*64
    elif change == 'rater': ratings['scores'][0]['rater_id'] = 'unknown'
    elif change == 'rater_duplicate': ratings['raters'].append(dict(ratings['raters'][0]))
    elif change == 'private_read': ratings['raters'][0]['private_item_scores_read'] = True
    elif change == 'exposure': ratings['raters'][0]['exposure_note'] = ''
    else: ratings['scores'][0]['reason'] = ''
    save(case)
    with pytest.raises(ValueError):
        review.seal(case.bundle, case.path, case.sealed)


@pytest.mark.parametrize('change', ['human', 'missing', 'duplicate_id', 'duplicate_text', 'extra_column'])
def test_public_validation(case, change):
    path = case.bundle / 'blind.csv'
    with path.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    fields = list(blind.FIELDS)
    if change == 'human': rows[0]['trait_human'] = '50'
    elif change == 'missing': rows.pop()
    elif change == 'duplicate_id': rows[1]['id'] = rows[0]['id']
    elif change == 'duplicate_text':
        for key in ('target', 'question', 'answer'): rows[1][key] = rows[0][key]
    else:
        fields.append('judge')
        for row in rows: row['judge'] = ''
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    case.ratings['public_table_sha256'] = sha256(path)
    save(case)
    with pytest.raises(ValueError):
        review.seal(case.bundle, case.path, case.sealed)


@pytest.mark.parametrize('name', review.SEALED_FILES)
def test_sealed_tamper_fails_before_private_read(case, monkeypatch, name):
    review.seal(case.bundle, case.path, case.sealed)
    path = case.sealed / name
    if name == 'receipt.json':
        receipt = blind.read_json(path)
        receipt['n_samples'] = 99
        blind.write_json(path, receipt)
    else:
        path.write_bytes(path.read_bytes()+b' ')
    guard_private(case, monkeypatch)
    with pytest.raises(ValueError, match='封存'):
        review.compare(case.bundle, case.sealed)


def test_current_public_tamper_fails_before_private_read(case, monkeypatch):
    review.seal(case.bundle, case.path, case.sealed)
    path = case.bundle / 'blind.csv'
    path.write_bytes(path.read_bytes()+b'\n')
    guard_private(case, monkeypatch)
    with pytest.raises(ValueError, match='原文'):
        review.compare(case.bundle, case.sealed)


def test_text_altered_before_seal_rejected_on_reveal(case):
    path = case.bundle / 'blind.csv'
    path.write_text(path.read_text(encoding='utf-8').replace('合成回答', '修改回答'), encoding='utf-8')
    case.ratings['public_table_sha256'] = sha256(path)
    save(case)
    review.seal(case.bundle, case.path, case.sealed)
    with pytest.raises(ValueError, match='原文身份'):
        review.compare(case.bundle, case.sealed)


@pytest.mark.parametrize('change', ['source', 'key'])
def test_source_and_private_mapping_tamper(case, change):
    review.seal(case.bundle, case.path, case.sealed)
    if change == 'source':
        case.source.write_bytes(case.source.read_bytes()+b'\n')
    else:
        case.key['samples']['H001']['judge']['trait'] = 99
        blind.write_json(case.bundle / 'private_key.json', case.key)
    with pytest.raises(ValueError, match='来源'):
        review.compare(case.bundle, case.sealed)


def test_correct_comparison_and_rubric_snapshot(case):
    for row in case.ratings['scores']:
        row['trait'] += 5
        row['coherence'] -= 5
    case.ratings['scores'].reverse()
    save(case)
    review.seal(case.bundle, case.path, case.sealed)
    (case.bundle / 'assistant_rubric.md').unlink()
    result = review.compare(case.bundle, case.sealed)
    assert result['assistant_judge_threshold_met'] and not result['formal_e0_pass']
    assert len(result['items']) == 100 and len(result['per_target']) == 10
    for metric, bias in [('trait', 5), ('coherence', -5)]:
        scores = result['agreement'][metric]
        assert scores['n'] == scores['n_total'] == 100
        assert scores['spearman'] == pytest.approx(1)
        assert scores['mae'] == 5 and scores['mean_bias'] == bias
        assert scores['assistant_mean'] - scores['judge_mean'] == pytest.approx(bias)
        assert len(scores['assistant_range']) == len(scores['judge_range']) == 2
    assert sum(group['agreement']['trait']['n'] for group in result['per_dim_coef_pole']) == 100
    for item in result['items']:
        assert item['reason'] == '合成理由' and item['difference'] == {'trait': 5, 'coherence': -5}
        assert item['answer'] == case.key['samples'][item['id']]['answer']
    assert not any('human' in key for key in result)
    assert not (case.bundle / 'private_first_receipt.json').exists()


def test_constant_scores_have_no_correlation(case):
    for row in case.ratings['scores']:
        row['trait'] = row['coherence'] = 50
    save(case)
    review.seal(case.bundle, case.path, case.sealed)
    result = review.compare(case.bundle, case.sealed)
    assert all(metric['spearman'] is None for metric in result['agreement'].values())
    assert not result['assistant_judge_threshold_met']


def test_missing_judge_is_reported_without_fabrication():
    result = review._summary([
        {'assistant': {'trait': 40, 'coherence': 80}, 'judge': {'trait': None, 'coherence': 75}},
        {'assistant': {'trait': 50, 'coherence': 90}, 'judge': {'trait': 45, 'coherence': None}}])
    assert result['trait']['n'] == 1 and result['trait']['n_total'] == 2
    assert result['trait']['spearman'] is None and result['trait']['mean_bias'] == 5
    assert result['coherence']['n'] == 1 and result['coherence']['spearman'] is None


def test_cli_metadata_success_and_failure(case, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    module = importlib.import_module('e0_assistant_review')

    def setup(args, need_judge, need_model):
        assert not need_judge and not need_model
        return {'out_path': case.root / 'runs', 'api_key': 'SECRET'}, None, None, None

    monkeypatch.setattr(module, 'setup', setup)
    config = case.root / 'config.yaml'
    config.write_text('out_dir: unused\n', encoding='utf-8')
    args = SimpleNamespace(config=str(config), action='seal', bundle=str(case.bundle),
                           ratings=str(case.path), sealed_dir=None)
    with monkeypatch.context() as guarded:
        guard_private(case, guarded)
        out = module.run(args)
    meta = blind.read_json(out / 'run_meta.json')
    assert meta['status'] == 'completed' and meta['input_hashes'] == meta['data_hashes']
    assert not meta['model_loaded'] and not meta['judge_called'] and not meta['formal_e0_pass']
    assert meta['config']['api_key'] == '[已隐藏]' and meta['end_time']
    assert all('private_key' not in key for key in meta['input_hashes'])
    args.action, args.ratings, args.sealed_dir = 'compare', None, str(out / 'sealed')
    compared = module.run(args)
    meta = blind.read_json(compared / 'run_meta.json')
    assert meta['status'] == 'completed' and str(case.source.resolve()) in meta['input_hashes']
    assert str((case.bundle / 'private_key.json').resolve()) in meta['input_hashes']
    args.action, args.ratings, args.sealed_dir = 'seal', str(case.path), None
    case.ratings['scores'][0]['trait'] = True
    save(case)
    with pytest.raises(ValueError): module.run(args)
    metas = [blind.read_json(p) for p in (case.root / 'runs').glob('assistant_d5/*/run_meta.json')]
    failed = [m for m in metas if m['status'] == 'failed']
    assert len(failed) == 1 and failed[0]['error']['type'] == 'ValueError'
    assert failed[0]['input_hashes'] and failed[0]['end_time']
