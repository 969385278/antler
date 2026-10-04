"""C11：开发入口九臂共享语料并开启优化诊断，预检不运行test数据。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from test_core import model, D, L
from test_e1_kl_pilot import Tokenizer


def test_dev_preflight_uses_all_nine_rates_and_diagnostics(model, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    m = importlib.import_module('e1_kl_dev')
    config = tmp_path/'config.yaml';config.write_text('model: unused')
    directions = tmp_path/'directions.pt';directions.write_bytes(b'fixture')
    episodes = tmp_path/'episodes.json';episodes.write_text('{}')
    model_dir = tmp_path/'model';model_dir.mkdir();(model_dir/'config.json').write_text('{}')
    monkeypatch.setattr(m,'setup',lambda *a,**k:({'out_path':tmp_path/'out','model':str(model_dir)},None,None,None))
    monkeypatch.setattr(m,'LAYER',2)
    monkeypatch.setattr(m,'load_inputs',lambda *a:(torch.eye(5,D).repeat(L,1,1),torch.ones(L),
        [{'id':f'dev{i}','text':'story','intended':{'O':1}} for i in range(10)]))
    monkeypatch.setattr(m,'load_model',lambda *a:(model,Tokenizer()))
    monkeypatch.setattr(m,'probe_questions',lambda *a,**k:[f'q{i}' for i in range(30)])
    import e1_kl_pilot as pilot
    monkeypatch.setattr(pilot,'chat_prompt',lambda tok,q,system:system+q)
    monkeypatch.setattr(pilot,'generate',lambda model,tok,prompts,**k:[dict(text='answer',token_ids=[12,2],n_generated_tokens=2,finish_reason='eos') for p in prompts])
    monkeypatch.setattr(m,'fit_activation_difference',lambda *a:torch.zeros(D))
    calls=[]
    def fit(model,fit_items,eval_items,**kw):
        calls.append((kw,[i.probe for i in fit_items],[i.probe for i in eval_items]))
        return {'coordinates':torch.zeros(5),'vector':torch.zeros(D),'baseline':{'kl':.2},'final':{'kl':.1},'diagnostics':{}}
    monkeypatch.setattr(m,'fit_and_evaluate',fit)
    out=m.run(SimpleNamespace(config=str(config),directions=str(directions),experiences=str(episodes),preflight=True))
    assert len(calls)==9 and all(c[0]['diagnostics'] is True for c in calls)
    assert all(c[1:]==(['q0','q1'],['q20','q21']) for c in calls)
    assert [c[0]['lr'] for c in calls]==sum(m.learning_rates(D).values(),[])
    meta=json.loads((out/'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status']=='completed' and meta['episodes_completed']==1
    assert not meta['formal_e1_pass'] and not meta['behavior_validated']
