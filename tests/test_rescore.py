"""C8：复评只能改变独立产物的分数，保留原文、原分和输入证据。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import e0_rescore as rescore


def row():
    return {'layer': 19, 'dim': 'O', 'coef': -.125, 'pole': 'low', 'axis': 'pole',
            'split': 'calib', 'seed': 0, 'max_new_tokens': 320, 'trait_name': 'consistent',
            'result': {'trait': 40, 'coherence': 95, 'task': 100, 'samples': [
                {'q': '问题', 'a': '原始回答', 'question_index': 0, 'sample_index': 0,
                 'trait': 40, 'coherence': 95, 'task': 100}]}}


def source(tmp_path, record=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path/'calibration_judge_samples.jsonl'
    path.write_text(json.dumps(record or row(), ensure_ascii=False)+'\n', encoding='utf-8')
    meta = {'status': 'completed', 'model': 'frozen',
            'judge': {'model':'fixed','base_url':'https://example.invalid','max_tokens':16,'extra_body':{}},
            'prompt_hashes': {k:hashlib.sha256(getattr(rescore,k).encode()).hexdigest()
                              for k in ('COHERENCE_PROMPT','TASK_PROMPT')},
            'data_hashes': {str(rescore.KIT/'trait_eval/consistent.json'):
                            rescore.sha256(rescore.KIT/'trait_eval/consistent.json')}}
    (tmp_path/'run_meta.json').write_text(json.dumps(meta), encoding='utf-8')
    return path


class FakeJudge:
    model = 'fixed'
    base_url = 'https://example.invalid'
    max_tokens = 16
    extra_body = {}

    def __init__(self, cache_path=None):
        self.cache_path = cache_path

    def trait_scores(self, prompt, questions, answers):
        assert questions == ['问题'] and answers == ['原始回答']
        assert isinstance(prompt, str)
        return [50]

    def coherence_scores(self, questions, answers):
        return [None]

    def task_scores(self, questions, answers):
        return [90]


def test_rescore_preserves_text_old_scores_and_missing():
    original = row()
    before = copy.deepcopy(original)
    result = rescore.rescore_records([original], FakeJudge(), {'consistent':'固定提示'})[0]
    assert original == before
    sample = result['result']['samples'][0]
    assert sample['a'] == '原始回答'
    assert sample['original_scores'] == {'trait':40,'coherence':95,'task':100}
    assert sample['trait'] == 50 and sample['coherence'] is None
    assert result['result']['n_valid_coherence'] == 0
    assert result['result']['coherence'] is None
    assert result['replaces_original_scores'] is False


def test_duplicates_must_be_identical_and_are_not_rerated_twice(tmp_path):
    one, two = source(tmp_path/'one'), source(tmp_path/'two')
    records, hashes, _ = rescore.load_inputs([one,two], ['O'], [-.125])
    assert len(records) == 1 and len(hashes) == 4
    other = row(); other['result']['samples'][0]['trait'] = 41
    source(tmp_path/'two', other)
    with pytest.raises(ValueError, match='重复条件'):
        rescore.load_inputs([one,two], ['O'], [-.125])


def test_model_identity_in_legacy_config_is_checked(tmp_path):
    one, two = source(tmp_path/'one'), source(tmp_path/'two')
    for path, model in [(one,'first'),(two,'different')]:
        meta_path=path.parent/'run_meta.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        meta.pop('model')
        meta['config']={'model':model}
        meta_path.write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ValueError, match='源实验的模型'):
        rescore.load_inputs([one,two], ['O'], [-.125])


@pytest.mark.parametrize('change', ['test','pole','duplicate','score'])
def test_rejects_invalid_inputs_before_scoring(tmp_path, change):
    record = row()
    if change == 'test': record['split'] = 'test'
    if change == 'pole': record['trait_name'] = 'inventive'
    if change == 'duplicate': record['result']['samples'] *= 2
    if change == 'score': record['result']['samples'][0]['task'] = float('nan')
    path = source(tmp_path,record)
    with pytest.raises(ValueError):
        rescore.load_inputs([path], ['O'], [-.125])


def test_run_uses_fresh_cache_without_model_and_preserves_source(tmp_path, monkeypatch):
    path = source(tmp_path/'input')
    before = path.read_bytes()
    cfgpath = tmp_path/'config.yaml'; cfgpath.write_text('use_judge: true\n')
    cfg = {'out_path':tmp_path/'output', 'model':'frozen'}
    calls = []
    def setup(args, **kwargs):
        assert kwargs == {'need_model':False,'need_judge':False}
        return cfg,None,None,None
    def judge(cache_path):
        assert not Path(cache_path).exists()
        calls.append(cache_path)
        return FakeJudge(cache_path)
    monkeypatch.setattr(rescore, 'setup', setup)
    monkeypatch.setattr(rescore, 'Judge', judge)
    args = SimpleNamespace(config=str(cfgpath),sources=[str(path)],dims=['O'],coefs=[-.125])
    first, second = rescore.run(args), rescore.run(args)
    assert first != second and calls[0] != calls[1]
    assert path.read_bytes() == before
    meta = json.loads((first/'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed' and meta['n_answers'] == 1
    assert meta['model_loaded'] is False and meta['generation_performed'] is False
    assert meta['cache_initially_empty'] is True
    assert meta['replaces_original_scores'] is False


def test_rejects_changed_judge_and_prompts(tmp_path, monkeypatch):
    path = source(tmp_path/'input')
    cfgpath = tmp_path/'config.yaml'; cfgpath.write_text('use_judge: true\n')
    monkeypatch.setattr(rescore,'setup',lambda *a,**k:({'out_path':tmp_path/'out'},None,None,None))
    args = SimpleNamespace(config=str(cfgpath),sources=[str(path)],dims=['O'],coefs=[-.125])
    changed = FakeJudge(); changed.model = 'different'
    monkeypatch.setattr(rescore,'Judge',lambda **k:changed)
    with pytest.raises(ValueError, match='评委配置'):
        rescore.run(args)
    meta_path=path.parent/'run_meta.json'
    meta=json.loads(meta_path.read_text(encoding='utf-8'))
    meta['prompt_hashes']['TASK_PROMPT']='changed'
    meta_path.write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises(ValueError, match='固定评分提示'):
        rescore.run(args)
