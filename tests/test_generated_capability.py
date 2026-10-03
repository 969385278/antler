"""C5：严格解析、固定分母与真实微型模型decode暴露。"""
import sys
from pathlib import Path

import pytest
import torch

from antler.core.generate import generate
from antler.eval.generated_capability import parse_answer, exposure, summarize
from test_core import model
from test_generate import Tokenizer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import e0_generated_capability as runner


@pytest.mark.parametrize('text,pred,status',[
    ('Reason.\nFinal answer: B',1,'valid'),
    ('Reason.\n final answer: c \n\n',2,'valid'),
    ('B',None,'invalid_final_line'),
    ('Final answer: B. ',None,'invalid_final_line'),
    ('Final answer: B\nMore text',None,'invalid_final_line'),
    ('Final answer: A\nFinal answer: B',None,'conflicting_claims'),
    ('The correct answer is D.\nFinal answer: B',None,'conflicting_claims'),
    ('The answer is B.\nFinal answer: B',1,'valid'),
    ("I cannot answer this.\nFinal answer: C",None,'refusal'),
    ('',None,'invalid_final_line'),
])
def test_parser(text,pred,status):
    result=parse_answer(text)
    assert (result['prediction'],result['parse_status'])==(pred,status)


class CharTokenizer:
    def batch_decode(self,rows,**kwargs):
        return [''.join(chr(t) for t in row if t) for row in rows]


def test_exposure_excludes_eos_padding_and_later_forwards():
    text='Reason\nFinal answer: A'
    tokens=list(map(ord,text))+[0]
    trace=[dict(phase='prefill',applied=False)]+[dict(phase='decode',applied=True)]*100
    item=dict(text=text,token_ids=tokens,n_generated_tokens=len(tokens))
    result=exposure(CharTokenizer(),item,trace,parse_answer(text))
    assert result['choice_token_index']==len(text)-1
    assert result['applied_before_choice']==len(text)-1
    assert result['applied_through_end']==len(text)


def test_exposure_rejects_bad_trace():
    item=dict(text='A',token_ids=[65],n_generated_tokens=1)
    with pytest.raises(ValueError):
        exposure(CharTokenizer(),item,[dict(phase='decode',applied=True)],parse_answer('A'))


def test_real_tiny_decode_exposure_and_zero_equivalence(model,monkeypatch):
    monkeypatch.setattr(model.generation_config,'eos_token_id',None)
    kw=dict(steer_phase='decode',max_new_tokens=4,temperature=0,batch_size=2,
            return_details=True,return_token_ids=True)
    runs=[]
    for steer in (None,{2:torch.zeros(32)},{2:torch.arange(32).float()/10}):
        traces=[]
        items=generate(model,Tokenizer(),['4 5','7'],steer=steer,phase_trace_out=traces,**kw)
        runs.append(items)
        for item in items:
            value=exposure(Tokenizer(),item,traces[0]['trace'],parse_answer(item['text']))
            assert value['applied_through_end']==(0 if steer is None else 3)
            assert value['applied_before_choice'] is None
    assert runs[0]==runs[1]
    assert all(not p.requires_grad for p in model.parameters())


def paired():
    return [dict(arm=arm,source_index=i,subject='a',answer=0,prediction=pred,
                 correct=pred==0,finish_reason='length',decode_steps_before_choice=5 if pred is not None else None,
                 applied_before_choice=(0 if arm=='zero' else 5) if pred is not None else None)
            for arm,preds in [('zero',[0,0]),('N_projected',[0,None])] for i,pred in enumerate(preds)]


def test_fixed_denominator_counts_parse_failure_wrong():
    result=summarize(paired(),expected_n=2)
    assert result['loss_percentage_points']==50
    assert result['right_to_wrong']==1
    assert result['arms']['N_projected']['invalid']==1
    assert not result['within_2pp_diagnostic'] and not result['formal_gate_eligible']


@pytest.mark.parametrize('mode',['duplicate','missing','incorrect','order'])
def test_summary_rejects_bad_pairing(mode):
    data=paired()
    if mode=='duplicate': data[-1]=data[-2]
    if mode=='missing': data.pop()
    if mode=='incorrect': data[-1]['correct']=True
    if mode=='order': data[-2:]=reversed(data[-2:])
    with pytest.raises(ValueError): summarize(data,expected_n=2)


def test_candidate_rejects_unregistered_source(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'SOURCE',tmp_path)
    (tmp_path/'injection_vectors.pt').write_bytes(b'unregistered')
    with pytest.raises(ValueError): runner.candidate()


def test_nonfinite_logits_abort():
    from types import SimpleNamespace
    runner.finite_logits(None,None,SimpleNamespace(logits=torch.zeros(1,2,4)))
    with pytest.raises(ValueError):
        runner.finite_logits(None,None,SimpleNamespace(logits=torch.tensor([float('nan')])))
