"""C1：固定索引切分、防止采样次数把calib溢出到test。"""
import importlib

import pytest

from test_trait_score import Tokenizer
from antler.traits.data import BIG_FIVE, load_trait

scoring = importlib.import_module('antler.eval.trait_score')


@pytest.mark.parametrize('dim', list(BIG_FIVE))
@pytest.mark.parametrize('pole', ['high', 'low'])
def test_splits_disjoint_original_indices(monkeypatch, dim, pole):
    data = load_trait(BIG_FIVE[dim][0 if pole == 'high' else 1], 'eval')
    monkeypatch.setattr(scoring, 'generate', lambda m,t,prompts,**kw: [
        {'text': '回答', 'n_generated_tokens': 1, 'finish_reason': 'eos'} for p in prompts])

    class Judge:
        def trait_scores(self, prompt, qs, answers):
            assert prompt == data['eval_prompt']
            return [50.] * len(qs)

    outputs = {}
    for split, expected in [('calib', list(range(10))), ('test', list(range(10,20))), ('all', list(range(20)))]:
        result = scoring.trait_score(None, Tokenizer(), Judge(), dim, pole=pole, split=split,
                                     n_questions=20, n_samples=3, with_coherence=False)
        samples = result['samples']
        assert [s['question_index'] for s in samples] == expected * 3
        assert [s['q'] for s in samples] == [data['questions'][i] for i in expected] * 3
        assert [s['sample_index'] for s in samples] == [i for i in range(3) for _ in expected]
        outputs[split] = {s['question_index'] for s in samples}
    assert outputs['calib'].isdisjoint(outputs['test'])
    assert outputs['calib'] | outputs['test'] == outputs['all']


def test_invalid_split_before_generation():
    with pytest.raises(ValueError, match='split'):
        scoring.trait_score(None, None, None, 'C', split='unknown')
