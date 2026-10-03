"""C5：固定分层、答案读取及零注入的CPU回归。"""
import pytest
import torch
from transformers import BatchEncoding

from test_core import model
from antler.eval.capability import stratified_sample, mmlu_accuracy


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


def test_reject_invalid_sampling_and_letter_tokenization(model):
    with pytest.raises(ValueError):
        stratified_sample(rows(), 2)
    with pytest.raises(ValueError):
        stratified_sample(rows(), 31)
    tok=Tokenizer(); tok.encode=lambda *a,**kw:[1,2]
    with pytest.raises(ValueError):
        mmlu_accuracy(model,tok,stratified_sample(rows(),3))
