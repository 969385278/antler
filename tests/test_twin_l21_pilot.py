"""第21层固定小样本：对照语义、配对单位、阶段暴露和停止。"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from antler.eval.twin_l21 import make_vectors, make_jobs, summarize, pair_jobs, pair_result
from test_core import model


def fixture():
    acts=torch.randn(9,60,32,generator=torch.Generator().manual_seed(1))
    vectors=make_vectors(acts,torch.eye(5,32),2.)
    materials=json.loads(Path('data/diagnostics/twin_outcome_dev_v1.json').read_text(encoding='utf8'))
    return vectors,materials


def test_random_control_matches_full_norm_and_feedback_is_shared():
    vectors,_=fixture();lookup={v['id']:v for v in vectors}
    assert len(vectors)==16
    for a,r in ((-1,-1),(-1,1),(1,-1),(1,1)):
        key=f'a{a}_r{r}_'
        assert lookup[key+'random_norm']['norm_s']==pytest.approx(lookup[key+'full_state']['norm_s'])
    assert torch.equal(lookup['a-1_r1_feedback_only']['vector'],lookup['a1_r1_feedback_only']['vector'])
    assert torch.equal(lookup['a1_r1_feedback_only']['vector'],-lookup['a1_r-1_feedback_only']['vector'])


def test_budget_and_capability_failure_skip_only_failed_arm():
    vectors,materials=fixture();ids={v['id'] for v in vectors}
    jobs=make_jobs(materials,vectors,ids)
    assert len(jobs)==250
    assert len([j for j in jobs if j['upstream_id'] is None])==160
    assert len(make_jobs(materials,vectors,ids-{vectors[0]['id']}))==240
    assert all(j['split']=='calib' and j['question_index']<5 and j['seed']==0 for j in jobs)
    assert all(j['memory_index']==0 for j in jobs if j['arm']=='injected')


def answers():
    vectors,materials=fixture()
    jobs=make_jobs(materials,vectors,{v['id'] for v in vectors})
    return [{**j,'a':f'回答{j["id"]}','trait':50.,'coherence':90.,'task':90.} for j in jobs]


def test_matching_is_closeness_not_highest_trait():
    rows=answers()
    for row in rows:
        if row['arm'] in ('context','injected'):
            row['trait']=20. if row['outcome_sign']==-1 else 80.
    report=summarize(rows)['reports'][0]['cells'][0]['metrics']['trait']
    assert report['matched_error']['mean']==0
    assert report['mismatch_minus_match_error']['mean']==60


def test_missing_scores_and_controls_do_not_pass():
    rows=answers(); rows[-1]['trait']=None
    assert summarize(rows)['missing_scores']==1
    assert summarize(rows)['formal_e1_pass'] is False
    with pytest.raises(ValueError):summarize(rows+[rows[0]])


def test_pairs_reverse_content_and_preserve_reference():
    jobs=pair_jobs(answers())
    assert len(jobs)==160
    for key in {j['pair_id'] for j in jobs}:
        pair=[j for j in jobs if j['pair_id']==key]
        assert len(pair)==2 and pair[0]['a_id']==pair[1]['b_id']
        assert pair[0]['reference_id']==pair[1]['reference_id']


@pytest.mark.parametrize('ab,ba,value',[('A','B',1),('B','A',0),('A','A',.5),('T','B',.5),(None,'B',None)])
def test_pair_position_bias_not_a_win(ab,ba,value):
    assert pair_result(ab,ba)==value


def test_default_never_loads_model(monkeypatch):
    import e1_twin_l21_pilot as entry
    monkeypatch.setattr(entry,'setup',lambda *a,**kw:({},None,None,None))
    monkeypatch.setattr(entry,'validate',lambda a:([],[],{}, {}, []))
    monkeypatch.setattr(entry,'load_model',lambda *a,**k:pytest.fail('预检不加载模型'))
    assert entry.run(SimpleNamespace(execute=False))['layer']==21


def test_real_model_decode_only_and_cleanup(monkeypatch,model):
    import e1_twin_l21_pilot as entry
    from test_twin_raw_behavior import Tokenizer
    from antler.core.hooks import Recorder
    vectors,materials=fixture();jobs=[j for j in make_jobs(materials,vectors,{v['id'] for v in vectors}) if j['arm']=='injected'][:5]
    vector=vectors[0]['vector'];tok=Tokenizer()
    monkeypatch.setattr(model.generation_config,'eos_token_id',2)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    def fake_generate(**kw):
        with Recorder(model,[2]) as rec,torch.no_grad():
            first=model(input_ids=kw['input_ids'],attention_mask=kw['attention_mask'],use_cache=True)
            assert first.past_key_values.get_seq_length()==4
            model(input_ids=torch.full((5,1),4,dtype=torch.long),past_key_values=first.past_key_values,use_cache=True)
        return torch.cat((kw['input_ids'],torch.tensor([[4,2]]).expand(5,2)),dim=1)
    monkeypatch.setattr(model,'generate',fake_generate)
    result=entry.collect(model,tok,jobs,[{'id':jobs[0]['vector_id'],'vector':vector}],[],layer=2)
    assert len(result)==5
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())
    assert all(not layer._forward_hooks for layer in model.model.layers)
