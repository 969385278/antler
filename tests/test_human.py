"""C9：纯CPU身份核验，不下载模型，不调用评委，不代填真实人工评分。"""
import csv
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from antler.eval.human import import_task_labels


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'question', 'answer', 'task_human', 'notes'])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def bundle(tmp_path):
    condition = {'layer': 16, 'dim': 'C', 'pole': 'low', 'coef': -.2}
    samples = [{'q': f'测试问题{i}', 'a': f'测试回答{i}', 'task': i * 3} for i in range(30)]
    source = tmp_path / 'samples.jsonl'
    source.write_text(json.dumps({**condition, 'result': {'samples': samples}}), encoding='utf-8')
    mapping = {f'T{i:03}': {'condition': condition, 'source': source.name,
                          'sample_index': i, 'task_judge': s['task']} for i, s in enumerate(samples)}
    key = tmp_path / 'key.json'
    key.write_text(json.dumps({'schema_version': 1, 'n_samples': 30, 'samples': mapping}), encoding='utf-8')
    rows = [{'id': f'T{i:03}', 'question': s['q'], 'answer': s['a'], 'task_human': s['task'], 'notes': ''}
            for i, s in enumerate(samples)]
    table = tmp_path / 'blind.csv'
    write_csv(table, rows)
    return table, key, source, rows


def test_identity_pairing_reordering_and_read_only(bundle):
    table, key, source, rows = bundle
    write_csv(table, list(reversed(rows)))
    before = [p.read_bytes() for p in (table, key, source)]
    result = import_task_labels(table, key)
    assert result['task_judge_validated'] and result['spearman'] == pytest.approx(1)
    assert not result['formal_e0_pass']
    assert before == [p.read_bytes() for p in (table, key, source)]


def test_missing_scores_not_zero_or_passing(bundle):
    table, key, _, rows = bundle
    rows[1]['task_human'] = ''
    write_csv(table, rows)
    result = import_task_labels(table, key)
    assert result['n_paired'] == 29 and result['missing_human_ids'] == ['T001']
    assert not result['task_judge_validated'] and not result['complete']
    for row in rows: row['task_human'] = ''
    write_csv(table, rows)
    assert import_task_labels(table, key)['spearman'] is None


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '101', '无'])
def test_invalid_scores_rejected(bundle, value):
    table, key, _, rows = bundle
    rows[0]['task_human'] = value
    write_csv(table, rows)
    with pytest.raises(ValueError, match='有限数字'): import_task_labels(table, key)


@pytest.mark.parametrize('field', ['id', 'question', 'answer'])
def test_tampered_identity_rejected(bundle, field):
    table, key, _, rows = bundle
    rows[0][field] = 'changed'
    write_csv(table, rows)
    with pytest.raises(ValueError): import_task_labels(table, key)


def test_duplicate_and_missing_ids_rejected(bundle):
    table, key, _, rows = bundle
    write_csv(table, rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match='ID'): import_task_labels(table, key)
    write_csv(table, rows[:-1])
    with pytest.raises(ValueError, match='ID'): import_task_labels(table, key)


def test_constant_and_reversed_ratings_fail(bundle):
    table, key, _, rows = bundle
    for row in rows: row['task_human'] = 50
    write_csv(table, rows)
    result = import_task_labels(table, key)
    assert result['spearman'] is None and not result['task_judge_validated']
    for i, row in enumerate(rows): row['task_human'] = 100 - i * 3
    write_csv(table, rows)
    result = import_task_labels(table, key)
    assert result['spearman'] == pytest.approx(-1) and not result['task_judge_validated']


def test_private_mapping_score_and_path_checked(bundle):
    table, key, _, _ = bundle
    data = json.loads(key.read_text())
    data['samples']['T000']['task_judge'] = 99
    key.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='评委分'): import_task_labels(table, key)
    data['samples']['T000']['source'] = '../elsewhere.jsonl'
    key.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='越出'): import_task_labels(table, key)


def test_missing_judge_not_counted(bundle):
    table, key, source, _ = bundle
    data = json.loads(key.read_text()); data['samples']['T000']['task_judge'] = None
    key.write_text(json.dumps(data))
    original = json.loads(source.read_text()); original['result']['samples'][0]['task'] = None
    source.write_text(json.dumps(original))
    result = import_task_labels(table, key)
    assert result['n_paired'] == 29 and result['missing_judge_ids'] == ['T000']
    assert not result['task_judge_validated']


def test_entrypoint_metadata_without_model_or_judge(bundle, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    module = importlib.import_module('e0_import_task_labels')
    def setup(args, need_judge, need_model):
        assert need_judge is False and need_model is False
        return {'out_path': tmp_path}, None, None, None
    monkeypatch.setattr(module, 'setup', setup)
    cfg = tmp_path / 'config.yaml'; cfg.write_text('out_dir: outputs/test\n')
    table, key, _, _ = bundle
    out = module.run(SimpleNamespace(csv=str(table), key=str(key), source_root=None, config=str(cfg)))
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and not meta['model_loaded'] and not meta['judge_called']
    assert meta['input_hashes'] and (out / 'agreement.json').exists()
