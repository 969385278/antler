"""C5：固定分层、答案读取及零注入的CPU回归。"""
import pytest
import torch
from transformers import BatchEncoding

from test_core import model
from antler.eval.capability import stratified_sample, mmlu_accuracy, capability_summary, rotate_options


def rows():
    return [{'subject': subject, 'question': str(i), 'choices': ['a','b','c','d'], 'answer':i%4}
            for subject in ['math','physics','history'] for i in range(10)]


def test_stratified_fixed_balanced_without_replacement():
    sample = stratified_sample(rows(), 14)
    assert sample == stratified_sample(rows(),14)
    assert len({r['source_index'] for r in sample}) == 14
    counts = [sum(r['subject']==s for r in sample) for s in ['math','physics','history']]
    assert max(counts)-min(counts) == 1


class Tokenizer:
    def encode(self, letter, **kw):
        return [1+'ABCD'.index(letter)]
    def apply_chat_template(self, messages, **kw):
        return messages[-1]['content']
    def __call__(self, prompts, **kw):
        return BatchEncoding({'input_ids':torch.tensor([[5,6,7]]*len(prompts)),
                              'attention_mask':torch.ones(len(prompts),3,dtype=torch.long)})


def test_cpu_logits_and_zero_hook(model):
    sample = stratified_sample(rows(),6)
    baseline = mmlu_accuracy(model,Tokenizer(),sample,batch_size=2)
    injected = mmlu_accuracy(model,Tokenizer(),sample,steer={1:torch.zeros(model.config.hidden_size)},batch_size=2)
    with torch.no_grad():
        expected = int(model(input_ids=torch.tensor([[5,6,7]])).logits[0,-1,[1,2,3,4]].argmax())
    assert baseline == injected and baseline['n'] == 6
    assert all(r['prediction']==expected for r in baseline['samples'])
    assert all(not layer._forward_hooks for layer in model.model.layers)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert capability_summary(baseline,injected)['within_2pp']


def test_reject_invalid_sampling_and_letter_tokenization(model):
    with pytest.raises(ValueError):
        stratified_sample(rows(), 2)
    with pytest.raises(ValueError):
        stratified_sample(rows(), 31)
    tok=Tokenizer(); tok.encode=lambda *a,**kw:[1,2]
    with pytest.raises(ValueError):
        mmlu_accuracy(model,tok,stratified_sample(rows(),3))


def _scored(n_correct, total=1400):
    return {'n':total, 'accuracy':n_correct/total, 'samples':[
        {'source_index':i,'subject':'test','answer':0,'prediction':0 if i<n_correct else 1,
         'correct':i<n_correct} for i in range(total)]}


@pytest.mark.parametrize('correct,passes', [(933,True),(932,True),(931,False),(961,True)])
def test_exact_two_percentage_point_boundary(correct,passes):
    summary=capability_summary(_scored(960),_scored(correct))
    assert summary['within_2pp'] is passes
    assert summary['loss_percentage_points']==100*(960-correct)/1400
    assert summary['baseline_correct']==960 and summary['injected_correct']==correct
    if correct==932:
        assert 100*(960/1400-932/1400)>2  # 原实现会误拒绝恰好2点的条件。
        assert summary['loss_percentage_points']==2


@pytest.mark.parametrize('change',['duplicate','answer','prediction','order','count'])
def test_summary_rejects_incomparable_or_invalid_samples(change):
    left,right=_scored(960),_scored(932)
    if change=='duplicate':right['samples'][1]=right['samples'][0]
    elif change=='answer':right['samples'][0].update(answer=2,prediction=2)
    elif change=='prediction':right['samples'][0]['prediction']=1
    elif change=='order':right['samples'].reverse()
    else:right['n']=1401
    with pytest.raises(ValueError):capability_summary(left,right)


def test_rotation_preserves_correct_content_and_balances_positions():
    import copy
    original = stratified_sample(rows(), 14)
    saved = copy.deepcopy(original)
    for rotation in range(4):
        shifted = rotate_options(original, rotation)
        for source, target in zip(original, shifted):
            assert target['choices'][target['answer']] == source['choices'][source['answer']]
            assert target['source_index'] == source['source_index']
            assert target['question'] == source['question']
        assert rotate_options(shifted, (4-rotation) % 4) == original
    for index in range(len(original)):
        assert {rotate_options(original, r)[index]['answer'] for r in range(4)} == set(range(4))
    assert original == saved and rotate_options(original, 0) == original
    rotated = rotate_options(original, 1)
    rotated[0]['choices'][0] = '修改副本'
    assert original == saved


@pytest.mark.parametrize('rotation', [-1, 4, True, 1.5, '1'])
def test_rotation_rejects_invalid_rotation(rotation):
    with pytest.raises(ValueError):rotate_options(rows(), rotation)


def test_rotation_cpu_zero_hook_and_content_in_prompts(model):
    class RecordingTokenizer(Tokenizer):
        def __call__(self, prompts, **kw):
            self.prompts = list(prompts)
            return super().__call__(prompts, **kw)
    sample = rotate_options(stratified_sample(rows(), 3), 1)
    tok = RecordingTokenizer()
    baseline = mmlu_accuracy(model, tok, sample)
    injected = mmlu_accuracy(model, tok, sample,
                            steer={1:torch.zeros(model.config.hidden_size)})
    assert all('A. b\nB. c\nC. d\nD. a' in p for p in tok.prompts)
    assert baseline == injected
    assert all(r['answer'] == s['answer'] for r,s in zip(baseline['samples'],sample))
    assert capability_summary(baseline,injected)['loss_percentage_points'] == 0
    assert all(not layer._forward_hooks for layer in model.model.layers)
