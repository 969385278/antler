"""C9：用合成1800条验证固定配对盲审，不读取真实评分、不调用模型。"""
from collections import Counter, defaultdict
from copy import deepcopy
import importlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.eval import matched_review as review
from antler.eval.blind import read_json, write_json
from antler.eval.human import sha256


def save_source(case):
    write_json(case.source / 'run_meta.json', case.meta)
    (case.source / 'samples.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in case.rows),
        encoding='utf-8', newline='\n')
    write_json(case.source / 'artifact_hashes.json', {
        name: sha256(case.source / name) for name in review.SOURCE_ARTIFACTS})


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    traits = {pole: {'questions': [f'{pole}的合成题{i}' for i in range(20)],
                     'eval_prompt': f'{pole}的固定特质定义：问题{{question}}；回答{{answer}}'}
              for dim in ('C', 'N') for pole in review.BIG_FIVE[dim]}
    source = tmp_path / 'outputs' / 'main' / 'decode_strength' / review.RUN_NAME
    source.mkdir(parents=True)
    code = tmp_path / 'experiments' / 'e0_decode_strength.py'
    code.parent.mkdir()
    code.write_text('# 合成锁定源码\n', encoding='utf-8', newline='\n')
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    monkeypatch.setattr(review, 'PINNED_INPUTS', {'experiments/e0_decode_strength.py': sha256(code)})

    def load_trait(pole, split):
        assert split == 'eval'
        return traits[pole]

    monkeypatch.setattr(review, 'load_trait', load_trait)
    rows = []
    for dim, pole, arm, seed, q, sample in itertools.product(
            ('C', 'N'), ('high', 'low'), review.ARMS, (1, 2, 3), range(10), range(3)):
        target, level = review.BIG_FIVE[dim][pole == 'low'], review.ARMS.index(arm)
        # zero与all故意同文，保留不同条件身份和分数，检查不重抽与差异报告。
        text_arm = 'same' if arm in ('zero', 'all') else arm
        rows.append({'dim': dim, 'pole': pole, 'arm': arm, 'seed': seed,
                     'question_index': q, 'sample_index': sample, 'target': target,
                     'layer': 21, 'axis': 'pole', 'coef': review.COEFS[dim][arm], 'split': 'calib',
                     'steer_phase': 'decode' if arm.startswith('decode') else 'all',
                     'q': traits[target]['questions'][q],
                     'a': f'合成回答{target}/{q}/{seed}/{sample}/{text_arm}',
                     'trait': 20 + q + seed + sample + level * 5,
                     'coherence': 80 + q - level * 2, 'task': 90 - level})
    meta = {'task': 'C19', 'status': 'completed', 'stage': 'completed',
            'git_commit': review.RUN_COMMIT, 'start_time': review.RUN_START,
            'end_time': '2026-10-04T00:00:00+00:00', 'layer': 21,
            'seeds': [1, 2, 3], 'arms': list(review.ARMS), 'coefs': review.COEFS,
            'split': 'calib', 'expected_samples': 1800, 'generated': 1800,
            'n_samples_per_question': 3, 'max_new_tokens': 320, 'batch_size': 16,
            'temperature': 1., 'top_p': 1., 'preflight_samples': 8, 'preflight_tokens': 16,
            'max_score_inputs': 5400, 'score_shuffle_seed': 0, 'formal_e0_pass': False,
            'judge': {'model': 'deepseek-flash'}, 'data_hashes': {str(code): sha256(code)}}
    for name in review.SOURCE_ARTIFACTS:
        (source / name).write_text('{}\n', encoding='utf-8', newline='\n')
    rubric = tmp_path / 'approved_rubric.md'
    rubric.write_bytes(b'\xef\xbb\xbf' + '历史批准量尺；旧批曝光说明仅指旧批。\r\n'.encode('utf-8'))
    monkeypatch.setattr(review, 'APPROVED_RUBRIC_SHA256', sha256(rubric))
    case = SimpleNamespace(root=tmp_path, source=source, code=code, rows=rows, meta=meta,
                           traits=traits, rubric=rubric, bundle=tmp_path / 'bundle',
                           path=tmp_path / 'ratings.json', sealed=tmp_path / 'sealed')
    save_source(case)
    return case


def save_ratings(case):
    # 保留NaN等非法输入供被测函数拒绝。
    case.path.write_text(json.dumps(case.ratings, ensure_ascii=False), encoding='utf-8', newline='\n')


@pytest.fixture
def case(source_case):
    case = source_case
    case.export_result = review.export(case.source, case.rubric, case.bundle)
    case.key = read_json(case.bundle / 'private_key.json')
    case.ratings = {'schema_version': 1, 'rating_source': 'assistant',
        'author_standard_approved': True, 'approval_quote': '认可这个尺度，请继续。',
        'public_table_sha256': sha256(case.bundle / 'public.json'),
        'rubric_sha256': sha256(case.bundle / 'assistant_rubric.md'),
        'raters': [{'id': 'synthetic-assistant', 'fresh_context': True,
                    'private_item_scores_read': False, 'private_item_conditions_read': False,
                    'exposure_note': '新上下文仅接触本批公开文本、特质定义与批准量尺。'}],
        'scores': []}
    for identity, entry in case.key['samples'].items():
        row = case.rows[entry['source']['row_index']]
        level = review.ARMS.index(row['arm'])
        case.ratings['scores'].append({'id': identity, 'trait': row['trait'] + 5 + 2 * level,
            'coherence': row['coherence'] - 3 + level, 'reason': '合成中文理由，供测试使用。',
            'rater_id': 'synthetic-assistant'})
    save_ratings(case)
    return case


def guard_private(case, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        resolved = path.resolve()
        assert resolved != (case.bundle / 'private_key.json').resolve(), '封存前不应读私有映射'
        assert not resolved.is_relative_to(case.source.resolve()), '封存前不应读源评分'
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)


def test_fixed_selection_and_complete_groups(source_case):
    selected = review.select_rows(source_case.rows)
    assert len(selected) == 200 and {x[0] for x in selected} == review.IDS
    assert [x[0] for x in selected] == [f'M{i:03d}' for i in range(1, 201)]
    groups = defaultdict(list)
    for identity, index, row in selected:
        assert source_case.rows[index] is row
        groups[row['dim'], row['pole'], row['question_index']].append(row)
    assert len(groups) == 40
    seeds = Counter()
    for group in groups.values():
        assert {r['arm'] for r in group} == set(review.ARMS)
        assert len({(r['seed'], r['sample_index']) for r in group}) == 1
        seeds[group[0]['seed']] += 1
    assert seeds == {1: 14, 2: 13, 3: 13}
    for dim, pole in itertools.product(('C', 'N'), ('high', 'low')):
        counts = Counter(group[0]['seed'] for key, group in groups.items() if key[:2] == (dim, pole))
        assert sorted(counts.values()) == [3, 3, 4]
    assert review.select_rows(source_case.rows) == selected


def test_selection_never_reads_scores_or_answers(source_case):
    class IdentityOnly(dict):
        def __getitem__(self, key):
            assert key in {'dim', 'pole', 'arm', 'seed', 'question_index', 'sample_index'}
            return super().__getitem__(key)

        def get(self, *args, **kwargs):
            raise AssertionError('选样不能额外读取内容或分数')

    selected = review.select_rows([IdentityOnly(row) for row in source_case.rows])
    assert len(selected) == 200
    assert [i for _, i, _ in selected] == [i for _, i, _ in review.select_rows(source_case.rows)]


def test_export_public_boundary_duplicates_and_provenance(case):
    public = read_json(case.bundle / 'public.json')
    assert len(public) == 200
    assert all(set(row) == {'id', 'target', 'question', 'answer'} for row in public)
    texts = Counter((r['target'], r['question'], r['answer']) for r in public)
    assert Counter(texts.values()) == {1: 120, 2: 40}
    assert (case.bundle / 'assistant_rubric.md').read_bytes() == case.rubric.read_bytes()
    assert read_json(case.bundle / 'trait_definitions.json') == {
        pole: trait['eval_prompt'] for pole, trait in case.traits.items()}
    manifest = read_json(case.bundle / 'export_manifest.json')
    assert all(sha256(case.bundle / name) == digest for name, digest in manifest['hashes'].items())
    assert case.export_result['n_samples'] == 200 and not case.export_result['formal_e0_pass']
    assert case.export_result['source_provenance']['git_commit'] == review.RUN_COMMIT
    assert case.export_result['source_samples_sha256'] == sha256(case.source / 'samples.jsonl')
    assert case.key['source_manifest_sha256'] == sha256(case.source / 'artifact_hashes.json')
    for entry in case.key['samples'].values():
        row = case.rows[entry['source']['row_index']]
        assert entry['source']['row_sha256'] == review._digest(review._json_bytes(row))
        assert entry['condition']['phase'] == row['steer_phase']
    old = (case.bundle / 'public.json').read_bytes()
    with pytest.raises(FileExistsError):
        review.export(case.source, case.rubric, case.bundle)
    assert (case.bundle / 'public.json').read_bytes() == old


@pytest.mark.parametrize('key,value', [
    ('seed', True), ('seed', 0), ('question_index', 10), ('sample_index', 3),
    ('layer', 20), ('layer', True), ('target', 'calm'), ('q', 'test题'), ('axis', 'global'),
    ('split', 'test'), ('coef', True), ('coef', .001), ('steer_phase', 'prefill'),
    ('arm', 'prefill'), ('trait', None), ('coherence', float('nan')), ('task', 101)])
def test_source_identity_and_score_validation(source_case, key, value):
    source_case.rows[0][key] = value
    save_source(source_case)
    with pytest.raises(ValueError):
        review.export(source_case.source, source_case.rubric, source_case.bundle)
    assert not source_case.bundle.exists()


@pytest.mark.parametrize('change', ['missing', 'duplicate'])
def test_source_requires_all_1800(source_case, change):
    if change == 'missing':
        source_case.rows.pop()
    else:
        source_case.rows[-1] = dict(source_case.rows[0])
    save_source(source_case)
    with pytest.raises(ValueError, match='1800'):
        review.verify_source(source_case.source)


@pytest.mark.parametrize('key,value', [
    ('status', 'running'), ('stage', 'generation'), ('git_commit', 'wrong'),
    ('start_time', '2026-10-05'), ('generated', 1799), ('layer', True),
    ('seeds', [0, 1, 2]), ('max_new_tokens', 321), ('temperature', 1),
    ('judge', {'model': 'changed-judge'}), ('formal_e0_pass', True), ('end_time', '')])
def test_source_fixed_protocol(source_case, key, value):
    source_case.meta[key] = value
    save_source(source_case)
    with pytest.raises(ValueError, match='登记'):
        review.verify_source(source_case.source)


@pytest.mark.parametrize('change', ['artifact', 'manifest_missing', 'input_hash', 'wrong_directory'])
def test_source_provenance_rejects_tamper(source_case, change):
    case = source_case
    directory = case.source
    if change == 'artifact':
        (case.source / 'assessment.json').write_text('changed', encoding='utf-8')
    elif change == 'manifest_missing':
        manifest = read_json(case.source / 'artifact_hashes.json')
        manifest.pop('judge_responses.jsonl')
        write_json(case.source / 'artifact_hashes.json', manifest)
    elif change == 'input_hash':
        case.meta['data_hashes'][str(case.code)] = 'a' * 64
        save_source(case)
    else:
        directory = case.source.parent / 'other_round'
    with pytest.raises(ValueError):
        review.verify_source(directory)


def test_seal_reads_only_public_material_and_keeps_bytes(case, monkeypatch):
    guard_private(case, monkeypatch)
    receipt = review.seal(case.bundle, case.path, case.sealed)
    assert receipt['n_samples'] == 200 and receipt['rating_source'] == 'assistant'
    assert not receipt['formal_e0_pass']
    assert (case.sealed / 'ratings.json').read_bytes() == case.path.read_bytes()
    for name in review.PUBLIC_FILES:
        assert (case.sealed / name).read_bytes() == (case.bundle / name).read_bytes()
    assert not any('private_key' in name for name in receipt['input_hashes'])
    old = (case.sealed / 'receipt.json').read_bytes()
    with pytest.raises(FileExistsError):
        review.seal(case.bundle, case.path, case.sealed)
    assert (case.sealed / 'receipt.json').read_bytes() == old


@pytest.mark.parametrize('score', [True, False, None, '50', float('nan'), float('inf'), -1, 101])
def test_invalid_assistant_scores(case, score):
    case.ratings['scores'][0]['coherence'] = score
    save_ratings(case)
    with pytest.raises(ValueError, match='有限数字'):
        review.seal(case.bundle, case.path, case.sealed)
    assert not case.sealed.exists()


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'extra', 'unknown', 'extra_field'])
def test_ratings_require_exact_ids_and_fields(case, change):
    rows = case.ratings['scores']
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[-1]['id'] = rows[0]['id']
    elif change == 'extra':
        rows.append(dict(rows[0]))
    elif change == 'unknown':
        rows[0]['id'] = 'M999'
    else:
        rows[0]['judge'] = 10
    save_ratings(case)
    with pytest.raises(ValueError, match='200'):
        review.seal(case.bundle, case.path, case.sealed)


@pytest.mark.parametrize('change', ['unapproved', 'quote', 'source', 'public_hash', 'rubric_hash',
    'fresh', 'scores_read', 'conditions_read', 'exposure', 'rater_id', 'reason_english'])
def test_ratings_require_provenance_and_chinese_reasons(case, change):
    data, rater = case.ratings, case.ratings['raters'][0]
    if change == 'unapproved': data['author_standard_approved'] = False
    elif change == 'quote': data['approval_quote'] = ''
    elif change == 'source': data['rating_source'] = 'human'
    elif change == 'public_hash': data['public_table_sha256'] = '0' * 64
    elif change == 'rubric_hash': data['rubric_sha256'] = '0' * 64
    elif change == 'fresh': rater['fresh_context'] = False
    elif change == 'scores_read': rater['private_item_scores_read'] = True
    elif change == 'conditions_read': rater['private_item_conditions_read'] = True
    elif change == 'exposure': rater['exposure_note'] = ''
    elif change == 'rater_id': data['scores'][0]['rater_id'] = 'unknown'
    else: data['scores'][0]['reason'] = 'English only'
    save_ratings(case)
    with pytest.raises(ValueError):
        review.seal(case.bundle, case.path, case.sealed)


@pytest.mark.parametrize('name', review.SEALED_FILES)
def test_damaged_seal_rejected_before_private_read(case, monkeypatch, name):
    review.seal(case.bundle, case.path, case.sealed)
    path = case.sealed / name
    if name == 'receipt.json':
        body = read_json(path)
        body['n_samples'] = 199
        write_json(path, body)
    else:
        path.write_bytes(path.read_bytes() + b' ')
    guard_private(case, monkeypatch)
    with pytest.raises(ValueError, match='封存'):
        review.compare(case.bundle, case.sealed)


@pytest.mark.parametrize('name', review.PUBLIC_FILES)
def test_changed_public_material_rejected_before_reveal(case, monkeypatch, name):
    review.seal(case.bundle, case.path, case.sealed)
    path = case.bundle / name
    path.write_bytes(path.read_bytes() + b' ')
    guard_private(case, monkeypatch)
    with pytest.raises(ValueError, match='公开材料'):
        review.compare(case.bundle, case.sealed)


@pytest.mark.parametrize('change', ['source', 'private_key'])
def test_compare_rejects_source_and_mapping_tamper(case, change):
    review.seal(case.bundle, case.path, case.sealed)
    if change == 'source':
        path = case.source / 'samples.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
    else:
        key = deepcopy(case.key)
        key['samples']['M001']['condition']['seed'] = 0
        write_json(case.bundle / 'private_key.json', key)
    with pytest.raises(ValueError, match='哈希'):
        review.compare(case.bundle, case.sealed)


def test_approved_rubric_must_keep_original_bytes(source_case):
    source_case.rubric.write_bytes(source_case.rubric.read_bytes() + b' ')
    with pytest.raises(ValueError):
        review.export(source_case.source, source_case.rubric, source_case.bundle)
    assert not source_case.bundle.exists()


def test_registered_server_config_can_differ_from_local_copy(source_case):
    path = source_case.root / 'configs' / 'main.yaml'
    path.parent.mkdir()
    path.write_text('server_setting: local_copy\n', encoding='utf-8')
    recorded = review._digest(b'server_setting: approved_server_override\n')
    source_case.meta['data_hashes']['/server/repository/configs/main.yaml'] = recorded
    save_source(source_case)
    result = review.verify_source(source_case.source)
    evidence = [r for r in result['input_verification'] if r['source'].endswith('/configs/main.yaml')]
    assert len(evidence) == 1
    assert evidence[0]['status'] == 'source_recorded_no_matching_local_copy'
    assert str(path.resolve()) not in result['input_hashes']


def test_full_comparison_question_pairs_deltas_and_distributions(case):
    review.seal(case.bundle, case.path, case.sealed)
    result = review.compare(case.bundle, case.sealed)
    assert result['n_samples'] == 200 and result['n_question_groups'] == 40
    assert result['rating_source'] == 'assistant' and not result['formal_e0_pass']
    assert len(result['items']) == 200 and len(result['per_target']) == 4
    assert len(result['per_condition']) == 20 and len(result['paired_comparisons']) == 28
    expected_pairs = {(target, arm, reference) for target in case.traits for arm in review.ARMS[1:]
                      for reference in (('zero',) if arm == 'all' else ('zero', 'all'))}
    assert {(r['target'], r['arm'], r['reference']) for r in result['paired_comparisons']} == expected_pairs
    for report in result['paired_comparisons']:
        level = review.ARMS.index(report['arm']) - review.ARMS.index(report['reference'])
        assert report['n_questions'] == 10 and report['n_samples_per_question'] == 1
        assert not report['formal_e0_pass']
        for side, trait_delta, coherence_delta in [('judge', 5 * level, -2 * level),
                                                  ('assistant', 7 * level, -level)]:
            for metric, expected in [('trait', trait_delta), ('coherence', coherence_delta)]:
                value = report['scores'][side][metric]
                assert value['n_questions'] == 10 and value['n_boot'] == 10000 and value['seed'] == 0
                assert value['mean'] == value['ci_low'] == value['ci_high'] == expected
        for metric, expected in [('trait', 2 * level), ('coherence', level)]:
            value = report['difference_in_differences'][metric]
            assert value['mean'] == value['ci_low'] == value['ci_high'] == expected
            assert value['n_questions'] == 10 and value['n_boot'] == 10000 and value['seed'] == 0
    for metric, bias in [('trait', 9), ('coherence', -1)]:
        summary = result['agreement'][metric]
        assert summary['n'] == summary['n_total'] == 200
        assert summary['mean_bias'] == bias
        for counts in summary['distributions'].values():
            assert sum(r['count'] for r in counts) == 200
    assert all(group['agreement']['trait']['n'] == 10 for group in result['per_condition'])
    assert all(group['trait']['n'] == 50 for group in result['per_target'].values())
    assert str((case.source / 'samples.jsonl').resolve()) in result['input_hashes']
    assert str((case.bundle / 'private_key.json').resolve()) in result['input_hashes']


def test_same_text_different_ids_keeps_original_rating_differences(case):
    before = case.path.read_bytes()
    review.seal(case.bundle, case.path, case.sealed)
    result = review.compare(case.bundle, case.sealed)
    assert case.path.read_bytes() == before
    assert (case.sealed / 'ratings.json').read_bytes() == before
    duplicates = result['identical_text_groups']
    assert len(duplicates) == 40
    assert all(len(group['ids']) == 2 and len(group['pairs']) == 1 for group in duplicates)
    for group in duplicates:
        pair = group['pairs'][0]
        assert abs(pair['assistant_difference']['trait']) == 7
        assert abs(pair['assistant_difference']['coherence']) == 1
        assert abs(pair['judge_difference']['trait']) == 5
        assert abs(pair['judge_difference']['coherence']) == 2


def test_mapping_identity_tamper_rejected_even_with_updated_manifest(case):
    key = deepcopy(case.key)
    key['samples']['M001']['condition']['phase'] = 'prefill'
    write_json(case.bundle / 'private_key.json', key)
    manifest = read_json(case.bundle / 'export_manifest.json')
    manifest['hashes']['private_key.json'] = sha256(case.bundle / 'private_key.json')
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    manifest['manifest_sha256'] = review._digest(review._json_bytes(body))
    write_json(case.bundle / 'export_manifest.json', manifest)
    review.seal(case.bundle, case.path, case.sealed)
    with pytest.raises(ValueError, match='条件身份'):
        review.compare(case.bundle, case.sealed)


@pytest.fixture
def cli(case, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    module = importlib.import_module('e0_matched_review')

    def setup(args, need_judge, need_model):
        assert need_judge is False and need_model is False
        return {'out_path': case.root / 'runs', 'api_key': 'PRIVATE_TEST_TOKEN'}, None, None, None

    monkeypatch.setattr(module, 'setup', setup)
    monkeypatch.setattr(module, '_git', lambda *args: None)
    config = case.root / 'config.yaml'
    config.write_text('model: unused\n', encoding='utf-8')
    args = SimpleNamespace(config=str(config), action='seal', source_dir=None, rubric=None,
                           bundle=str(case.bundle), ratings=str(case.path), sealed_dir=None)
    return module, args


def test_cli_success_manifest_and_no_model_or_judge(case, cli, monkeypatch):
    module, args = cli
    with monkeypatch.context() as guarded:
        guard_private(case, guarded)
        out = module.run(args)
    meta = read_json(out / 'run_meta.json')
    assert meta['status'] == 'completed' and meta['end_time']
    assert not meta['model_loaded'] and not meta['judge_called'] and not meta['generation_performed']
    assert not meta['formal_e0_pass'] and meta['rating_source'] == 'assistant'
    assert meta['config']['api_key'] == '[已隐藏]'
    assert meta['code_hashes'] and meta['protocol_hashes']
    assert meta['config_sha256'] == sha256(args.config)
    assert all('private_key' not in name for name in meta['input_hashes'])
    manifest = read_json(out / 'artifact_hashes.json')
    assert all(sha256(out / name) == digest for name, digest in manifest.items())
    assert 'run_meta.json' in manifest and 'sealed/ratings.json' in manifest
    args.action, args.ratings, args.sealed_dir = 'compare', None, str(out / 'sealed')
    compared = module.run(args)
    meta = read_json(compared / 'run_meta.json')
    assert meta['status'] == 'completed'
    assert str((case.bundle / 'private_key.json').resolve()) in meta['input_hashes']
    assert str((case.source / 'samples.jsonl').resolve()) in meta['input_hashes']


def test_cli_export_uses_own_bundle_and_records_source(case, cli):
    module, args = cli
    args.action, args.bundle, args.ratings = 'export', None, None
    args.source_dir, args.rubric = str(case.source), str(case.rubric)
    out = module.run(args)
    assert (out / 'bundle' / 'public.json').exists()
    meta = read_json(out / 'run_meta.json')
    assert meta['source_provenance']['source_dir'] == str(case.source.resolve())
    assert meta['source_provenance']['library']['git_commit'] == review.RUN_COMMIT
    assert meta['status'] == 'completed'


def test_cli_failure_retains_generic_error_and_manifest(case, cli, monkeypatch):
    module, args = cli

    def fail(*args):
        raise ValueError('PRIVATE_TEST_TOKEN authentication failed')

    monkeypatch.setattr(module.matched_review, 'seal', fail)
    with pytest.raises(RuntimeError) as error:
        module.run(args)
    assert 'PRIVATE_TEST_TOKEN' not in str(error.value)
    paths = list((case.root / 'runs' / 'matched_review').glob('*/run_meta.json'))
    assert len(paths) == 1
    meta = read_json(paths[0])
    assert meta['status'] == 'failed' and meta['error_type'] == 'ValueError'
    assert meta['end_time'] and meta['input_hashes']
    assert 'PRIVATE_TEST_TOKEN' not in paths[0].read_text(encoding='utf-8')
    assert (paths[0].parent / 'artifact_hashes.json').exists()


@pytest.mark.parametrize('action,changes', [
    ('export', {'source_dir': 'source', 'rubric': None, 'bundle': None, 'ratings': None}),
    ('seal', {'source_dir': 'source'}), ('seal', {'sealed_dir': 'sealed'}),
    ('compare', {'sealed_dir': 'sealed'}), ('compare', {'sealed_dir': None, 'ratings': None})])
def test_cli_rejects_action_inappropriate_paths_before_library(case, cli, monkeypatch, action, changes):
    module, args = cli
    args.action = action
    vars(args).update(changes)

    def forbidden(*args):
        raise AssertionError('参数边界失败时不能访问库数据')

    for method in ('export', 'seal', 'compare'):
        monkeypatch.setattr(module.matched_review, method, forbidden)
    with pytest.raises(RuntimeError):
        module.run(args)
    meta = read_json(next((case.root / 'runs' / 'matched_review').glob('*/run_meta.json')))
    assert meta['error_type'] == 'ValueError' and meta['stage'] == 'validation'
