"""C21：匹配不看质量，Pareto保留多指标，完整配对与来源拒绝。"""
import sys
from pathlib import Path

import pytest

from antler.eval.prompt_persona import ARMS,PROMPTS,SEEDS,choose_prompt,frontier,validate,analyze

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import e0_prompt_persona as runner


def fixture_rows():
    traits={t:{'questions':[f'{t}-{q}' for q in range(20)]} for t in ('careless','dependable')}
    gains=dict(zero=0,original=40,projected=34,prompt_mild=10,prompt_moderate=34,prompt_strong=60)
    quality=dict(zero=90,original=75,projected=80,prompt_mild=90,prompt_moderate=82,prompt_strong=65)
    rows=[]
    for a in ARMS:
        for p in ('high','low'):
            target='dependable' if p=='high' else 'careless'
            for s in SEEDS:
                for q in range(10):
                    for i in range(3):
                        rows.append(dict(arm=a,pole=p,seed=s,question_index=q,sample_index=i,dim='C',
                                         target=target,q=traits[target]['questions'][q],layer=21,steer_phase='decode',
                                         coef=-.225 if a in ('original','projected') else 0,split='calib',
                                         trait=10+gains[a] if p=='low' else 80-gains[a],
                                         coherence=quality[a],task=quality[a],finish_reason='length'))
    return rows,traits


def test_match_uses_only_effect_and_fixed_tie_break():
    name,matched=choose_prompt(dict(projected=34,prompt_mild=30,prompt_moderate=38,prompt_strong=70))
    assert name=='prompt_mild' and matched
    assert not choose_prompt(dict(projected=34,prompt_mild=20,prompt_moderate=50,prompt_strong=70))[1]


def test_frontier_all_four_quality_axes_matter():
    result=frontier({'a':[34,-10,-10,-10,-10],'b':[34,-9,-9,-9,-9],
                     'c':[40,-20,-20,-20,-20],'d':[34,-8,-8,-8,-11]})
    assert result['dominated_by']['a']==['b']
    assert set(result['frontier'])=={'b','c','d'}


def test_equal_points_do_not_dominate_each_other():
    assert len(frontier({'a':[0]*5,'b':[0]*5})['frontier'])==2


def test_complete_analysis_rejects_practical_shortcut():
    rows,traits=fixture_rows()
    result=analyze(rows,traits)
    assert result['selected_prompt']=='prompt_moderate'
    assert result['selected_mean_matched']
    assert result['comparisons']['prompt_moderate']['ci_supports_matching']
    assert not result['comparisons']['prompt_moderate']['all_quality_bounds_ge0']
    assert result['pareto']['dominated_by']['projected']==['prompt_moderate']
    assert not result['strict_e0_pass'] and not result['practical_e0_pass']
    assert result['effects']['projected']['low']['trait']['mean']==34


@pytest.mark.parametrize('change',['missing','duplicate','test','coef','question','seed_bool'])
def test_identity_rejection(change):
    rows,traits=fixture_rows()
    if change=='missing':rows.pop()
    elif change=='duplicate':rows[-1]=rows[0]
    elif change=='test':rows[0]['split']='test'
    elif change=='coef':rows[0]['coef']=.1
    elif change=='question':rows[0]['q']=traits['dependable']['questions'][10]
    else:rows[0]['seed']=True
    with pytest.raises(ValueError):validate(rows,traits)


@pytest.mark.parametrize('score',[None,float('nan'),float('inf'),True,101])
def test_bad_scores_not_dropped(score):
    rows,traits=fixture_rows();rows[0]['task']=score
    with pytest.raises(ValueError):analyze(rows,traits)


def test_source_hash_mismatch_before_model(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'SOURCE',tmp_path)
    (tmp_path/'injection_vectors.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):runner.load_sources({'model':'unused'})


def test_all_prompt_texts_frozen_unique():
    assert len(PROMPTS)==3 and len(set(PROMPTS.values()))==3
    assert all('{question}' not in p for p in PROMPTS.values())


def test_runner_completes_all_arms_before_scoring(tmp_path,monkeypatch):
    import hashlib
    import json
    from types import SimpleNamespace
    import torch
    _,traits=fixture_rows()
    for t in traits.values():t['eval_prompt']='Trait {question} {answer}'
    templates={'coherence':runner.COHERENCE_PROMPT,'task':runner.TASK_PROMPT,
               **{p:t['eval_prompt'] for p,t in traits.items()}}
    identity=dict(model='deepseek-flash',base_url='https://example.test',max_tokens=16,extra_body={})
    old={'prompt_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in templates.items()},'judge':identity}
    config=tmp_path/'config.yaml';config.write_text('model: fake\n',encoding='utf-8')
    monkeypatch.setattr(runner,'setup',lambda *a,**kw:({'out_path':tmp_path,'model':'fake'},None,None,None))
    monkeypatch.setattr(runner,'load_sources',lambda cfg:({'original':torch.zeros(2560),'projected':torch.ones(2560)},old,[],tmp_path))
    monkeypatch.setattr(runner,'load_trait',lambda p,*a:traits[p])
    monkeypatch.setattr(runner,'Judge',lambda **kw:SimpleNamespace(**identity))
    class Tok:
        def apply_chat_template(self,messages,**kw):return json.dumps(messages)
        def batch_decode(self,*a,**kw):return ['token']
    class Model:
        config=SimpleNamespace(num_hidden_layers=36,hidden_size=2560)
        generation_config=SimpleNamespace(eos_token_id=2,to_dict=lambda:{'eos_token_id':2})
        def parameters(self):return iter([torch.zeros(1)])
        def register_forward_hook(self,*a,**kw):return SimpleNamespace(remove=lambda:None)
    monkeypatch.setattr(runner,'load_model',lambda *a,**kw:(Model(),Tok()))
    calls=[]
    def fake_generate(model,tok,prompts,**kw):
        calls.append((len(prompts),kw['max_new_tokens']))
        n=kw['max_new_tokens']
        if 'phase_trace_out' in kw:
            for start in range(0,len(prompts),16):
                trace=[dict(phase='prefill',applied=False)]+[dict(phase='decode',applied=kw['steer'] is not None)]*(n-1)
                kw['phase_trace_out'].append(dict(batch_start=start,trace=trace))
        return [dict(text='token',token_ids=[3]*n,n_generated_tokens=n,finish_reason='length') for _ in prompts]
    monkeypatch.setattr(runner,'generate',fake_generate)
    def score(judge,rows,actual_traits,out):
        assert len(rows)==1080 and len(calls)==38
        for r in rows:r.update(trait=20,coherence=90,task=90)
        return True
    monkeypatch.setattr(runner,'score_rows',score)
    out=runner.run(SimpleNamespace(config=str(config)))
    assert json.loads((out/'run_meta.json').read_text())['status']=='completed'
    assessment=json.loads((out/'assessment.json').read_text())
    assert not assessment['practical_e0_pass']
