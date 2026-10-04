"""C11：未完成网格拒绝选参、老师变化停止、只从新A初始化B。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_core import model, D, L
from test_e1_kl_pilot import Tokenizer


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e1_probe_repeat')


def test_running_reference_cannot_trigger_selection(module, tmp_path):
    (tmp_path/'run_meta.json').write_text(json.dumps({'status':'running','preflight':False,'episodes_completed':9}))
    with pytest.raises(ValueError, match='尚未完整结束'):
        module.verify_archive(tmp_path)


def test_archive_checks_real_files_not_only_completion_claim(module, tmp_path):
    meta = tmp_path/'run_meta.json'
    meta.write_text(json.dumps({'status':'completed','preflight':False,'episodes_completed':10}))
    data = tmp_path/'results.json'; data.write_text('{}')
    manifest = {'run_meta.json':module.sha256(meta),'results.json':module.sha256(data)}
    (tmp_path/'artifact_hashes.json').write_text(json.dumps(manifest))
    assert module.verify_archive(tmp_path)['status'] == 'completed'
    data.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='哈希'):
        module.verify_archive(tmp_path)


def test_heldout_compares_actual_tokens_and_prompts(module):
    record = {'probe':'q','teacher_prompt':'p','teacher_prompt_ids':[1], 'neutral_prompts':['n'],
              'text':'answer','token_ids':[2], 'n_generated_tokens':1,'finish_reason':'eos'}
    assert module.heldout_differences([record], [dict(record)], ['q']) == []
    changed = dict(record, token_ids=[3])
    assert module.heldout_differences([record], [changed], ['q'])[0]['different_fields'] == ['token_ids']
    assert module.heldout_differences([], [record], ['q'])
    assert module.heldout_differences([{'probe':'q'}], [{'probe':'q'}], ['q'])


def configured(module, model, monkeypatch, tmp_path):
    from antler.memory.kl_validation import KLItem
    root = tmp_path/'model'; root.mkdir(); (root/'config.json').write_text('{}')
    reference = tmp_path/'reference'; reference.mkdir()
    (reference/'artifact_hashes.json').write_text('{}')
    paths = {}
    for name in ['config','directions','experiences','probe_plan','model_fingerprint']:
        p=tmp_path/name; p.write_text('{}'); paths[name]=str(p)
    args = SimpleNamespace(**paths, reference_run=str(reference), execute=True)
    plan={'fit':[f'new{i}' for i in range(20)],'reference_fit':[f'old{i}' for i in range(20)],
          'evaluation':[f'eval{i}' for i in range(10)]}
    episodes=[{'id':f'ep{i}','text':'story','intended':{'O':1},'group_id':f'family{i//2}'} for i in range(10)]
    original={'episodes':[{'id':r['id'],'delta_A':[0.]*5,'arms':{'personality_lr0':{'coordinates':[9.]*5}}} for r in episodes]}
    selection={'spaces':{'personality':{'selected_arm':'personality_lr0','selected_lr':.005}}}
    monkeypatch.setattr(module,'LAYER',2)
    monkeypatch.setattr(module,'validate_reference',lambda a:({'config':{'dtype':'bfloat16'}},plan,
        torch.eye(5,D).repeat(L,1,1),torch.ones(L),episodes,original,selection))
    monkeypatch.setattr(module,'setup',lambda *a,**k:({'out_path':tmp_path/'out','model':str(root)},None,None,None))
    monkeypatch.setattr(module,'verify_model',lambda *a:{})
    loads=[]
    def load(*a):
        loads.append(a); return model,Tokenizer()
    monkeypatch.setattr(module,'load_model',load)
    calls=[]
    def corpus(m,t,experience,probes,max_tokens):
        items=[KLItem(q,torch.tensor([1]),torch.log_softmax(torch.zeros(1,64),dim=-1),
                      tuple(torch.tensor([2]) for _ in range(5))) for q in probes]
        records=[{'probe':q,'teacher_prompt':'p','teacher_prompt_ids':[1],'neutral_prompts':['n']*5,
                  'text':'answer','token_ids':[3],'n_generated_tokens':1,'finish_reason':'eos'} for q in probes]
        return items,records
    monkeypatch.setattr(module,'prepare_corpus',corpus)
    for episode in episodes:
        folder=reference/episode['id'];folder.mkdir()
        records=corpus(None,None,None,plan['evaluation'],64)[1]
        (folder/'teacher_answers.json').write_text(json.dumps(records))
    difference=torch.arange(D,dtype=torch.float32)
    monkeypatch.setattr(module,'fit_activation_difference',lambda *a:difference)
    def fit(m,training,evaluation,**kwargs):
        calls.append((kwargs,[i.probe for i in training],[i.probe for i in evaluation]))
        return {'coordinates':kwargs['init']+.1,'vector':torch.zeros(D),'baseline':{'kl':1.},'final':{'kl':.5}}
    monkeypatch.setattr(module,'fit_and_evaluate',fit)
    monkeypatch.setattr(module,'evaluate_kl',lambda *a,**k:{'kl':.7})
    return args,plan,calls,loads,difference


def test_repeat_is_disjoint_frozen_and_uses_new_A_not_old_B(module, model, monkeypatch, tmp_path):
    args,plan,calls,loads,difference=configured(module,model,monkeypatch,tmp_path)
    args.execute=False
    assert module.run(args) is None and not loads
    args.execute=True
    out=module.run(args)
    assert len(calls)==10 and len(loads)==1
    for kw,fit,evaluation in calls:
        assert fit==plan['fit'] and evaluation==plan['evaluation']
        assert kw['lr']==.005 and kw['steps']==60 and kw['diagnostics']
        assert torch.equal(kw['init'],difference[:5])
        assert not torch.equal(kw['init'],torch.full((5,),9.))
    meta=json.loads((out/'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status']=='completed' and meta['episodes_completed']==10
    assert {'git_dirty','config_path','torch','transformers','cuda','gpu','judge','seed'} <= meta.keys()
    assert meta['all_parameters_frozen'] and meta['judge'] is None
    assert not meta['formal_e1_pass'] and not meta['behavior_validated']
    for method in ['A','B']:
        result=json.loads((out/f'{method}_stability.json').read_text(encoding='utf-8'))
        assert result['n_episodes']==10 and result['n_groups']==5
    for name,digest in json.loads((out/'artifact_hashes.json').read_text(encoding='utf-8')).items():
        assert module.sha256(out/name)==digest
    with pytest.raises(FileExistsError): module.run(args)
    assert len(loads)==1


def test_teacher_changed_stops_before_fit_and_keeps_failure(module, model, monkeypatch, tmp_path):
    args,plan,calls,loads,_=configured(module,model,monkeypatch,tmp_path)
    folder=Path(args.reference_run)/'ep0'
    records=json.loads((folder/'teacher_answers.json').read_text(encoding='utf-8'));records[0]['token_ids']=[9]
    (folder/'teacher_answers.json').write_text(json.dumps(records))
    with pytest.raises(ValueError,match='共享留出题老师回答变化'):
        module.run(args)
    assert calls==[]
    out=next((tmp_path/'out/e1_probe_repeat').glob('*/run_meta.json')).parent
    assert json.loads((out/'run_meta.json').read_text(encoding='utf-8'))['status']=='failed'
    differences=json.loads((out/'ep0/heldout_teacher_comparison.json').read_text(encoding='utf-8'))['differences']
    assert differences[0]['different_fields']==['token_ids']
