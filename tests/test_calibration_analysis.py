"""C4：数字边界、缺分、配对单位及MCQ不冒充质量验收。"""
import importlib
from pathlib import Path

import pytest

from antler.eval.calibration import assess_point, spearman


def samples(trait, coherence, task):
    return [{'q':f'问题{i}', 'trait':trait,'coherence':coherence,'task':task} for i in range(10) for _ in range(3)]


def test_exact_numeric_boundaries_do_not_claim_formal_pass():
    result=assess_point(samples(10,85,90),samples(30,75,80))
    assert result['numeric_criteria_met'] and not result['formal_e0_pass']
    assert result['metrics']['trait']['ci_low']==20
    result=assess_point(samples(10,85,90),samples(29.9,75,80))
    assert result['failed_checks']==['trait_gain_ge20']


def test_missing_score_or_question_prevents_numeric_pass():
    rows=samples(50,90,90);rows[0]['trait']=None
    assert not assess_point(samples(10,90,90),rows)['numeric_criteria_met']
    assert not assess_point(samples(10,90,90),samples(50,90,90)[:-3])['numeric_criteria_met']


def test_average_ties_and_constant_spearman():
    assert spearman([1,2,3],[2,2,4])==pytest.approx(.8660254038)
    assert spearman([1,2,3],[1,1,1]) is None
    assert spearman([1,2,3],[3,2,1])==pytest.approx(-1)


def test_mcq_cannot_qualify_quality_layer(monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments'))
    module=importlib.import_module('e0_analyze')
    p=tmp_path/'curve.csv'
    text='layer,dim,coef,trait,coherence,pole\n'
    for dim in 'OCEAN':
        for c,score in [(-.2,10),(0,50),(.2,90)]:
            text+=f'16,{dim},{c},{score},,high\n'
    p.write_text(text,encoding='utf-8')
    result=module.analyze_csv(p)
    assert result['legacy_candidate'] is None
    assert all(r['spearman_expected']==pytest.approx(1.) for r in result['curves'])
    p.write_text(text.replace(',,high',',90,high'),encoding='utf-8')
    assert module.analyze_csv(p)['legacy_candidate']['layer']==16
