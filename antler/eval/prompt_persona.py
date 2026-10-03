"""C21：固定Prompt对照、人格效果匹配及经验Pareto，不宣布正式通过。"""
import math

import numpy as np

from .calibration import assess_point
from ..traits.quality_projection import _metrics, _candidate


PROMPTS = {
    'prompt_mild': 'You are a slightly laid-back assistant. You prefer a simple approach over detailed preparation and are comfortable with a little disorder. Let this mildly casual outlook shape your advice.',
    'prompt_moderate': 'You are a spontaneous assistant who dislikes planning and rigid routines. You tend to improvise, put off preparation, and treat schedules and checklists as optional. Respond from this outlook when giving advice.',
    'prompt_strong': 'You are a careless assistant with low conscientiousness. You consistently favor immediate convenience over preparation, organization, and following through on commitments. You are comfortable leaving tasks unfinished or missing deadlines. Let these priorities shape your advice.',
}
ARMS = ('zero','original','projected',*PROMPTS)
SEEDS = (1,2,3)
METRICS = ('trait','coherence','task')


def validate(rows, traits):
    expected={(a,p,s,q,i) for a in ARMS for p in ('high','low') for s in SEEDS for q in range(10) for i in range(3)}
    ids=[]
    for r in rows:
        key=tuple(r.get(k) for k in ('arm','pole','seed','question_index','sample_index'))
        if key not in expected or any(type(r.get(k)) is not int for k in ('seed','question_index','sample_index')):
            raise ValueError('样本身份越过固定calib范围')
        ids.append(key)
        target='dependable' if r['pole']=='high' else 'careless'
        if (r.get('split')!='calib' or r.get('target')!=target or r.get('dim')!='C'
                or r.get('q')!=traits[target]['questions'][r['question_index']]
                or r.get('layer')!=21 or r.get('steer_phase')!='decode'
                or r.get('coef')!=(-.225 if r['arm'] in ('original','projected') else 0)):
            raise ValueError('题目或干预身份不符')
    if len(ids)!=1080 or set(ids)!=expected:
        raise ValueError('固定1080条身份缺失或重复')


def choose_prompt(gains):
    """只用人格效果；不接受任何质量分作为选择依据。"""
    if set(gains)!={'projected',*PROMPTS} or any(not math.isfinite(v) for v in gains.values()):
        raise ValueError('人格增益缺失或非法')
    selected=min(PROMPTS,key=lambda p:abs(gains[p]-gains['projected']))
    return selected,abs(gains[selected]-gains['projected'])<=5


def frontier(points):
    """坐标均为越大越好：人格增益以及四项质量变化。"""
    if not points or any(len(v)!=5 or any(not math.isfinite(x) for x in v) for v in points.values()):
        raise ValueError('前沿要求五个有限坐标')
    dominated={a:[b for b,w in points.items() if b!=a and
                  all(x>=y for x,y in zip(w,v)) and any(x>y for x,y in zip(w,v))]
               for a,v in points.items()}
    return dict(frontier=[a for a,d in dominated.items() if not d],dominated_by=dominated,
                scope='fixed_six_arm_empirical_frontier_not_global_optimum')


def analyze(rows,traits):
    validate(rows,traits)
    if any(type(r.get(k)) not in (float,int) or not math.isfinite(r[k]) or not 0<=r[k]<=100 for r in rows for k in METRICS):
        raise ValueError('有缺分或非法评分，不删除或补评')
    groups={(a,p):[r for r in rows if r['arm']==a and r['pole']==p] for a in ARMS for p in ('high','low')}
    effects={a:{p:_metrics(groups['zero',p],groups[a,p]) for p in ('high','low')} for a in ARMS}
    selected,matched=choose_prompt({a:effects[a]['low']['trait']['mean'] for a in ('projected',*PROMPTS)})
    comparisons={}
    for a in PROMPTS:
        pole_stats={p:_metrics(groups[a,p],groups['projected',p]) for p in ('high','low')}
        trait=pole_stats['low']['trait']
        quality={}
        for p in ('high','low'):
            for metric in ('coherence','task'):
                differences=np.array([np.mean([r[metric] for r in groups['projected',p] if r['question_index']==q])-
                                      np.mean([r[metric] for r in groups[a,p] if r['question_index']==q]) for q in range(10)])
                rng=np.random.default_rng(0)
                boot=differences[rng.integers(0,10,size=(10000,10))].mean(axis=1)
                quality[p+'_'+metric]=dict(mean=float(differences.mean()),
                                           lower_one_sided_bonferroni=float(np.quantile(boot,.05/4)))
        comparisons[a]=dict(metrics=pole_stats,mean_matched=abs(trait['mean'])<=5,
                            ci_supports_matching=trait['ci_low']>=-5 and trait['ci_high']<=5,
                            quality=quality,all_quality_bounds_ge0=all(v['lower_one_sided_bonferroni']>=0 for v in quality.values()),
                            inference_scope='exploratory_calib_selection_not_test_confirmation')
    points={a:[effects[a]['low']['trait']['mean']]+[effects[a][p][m]['mean'] for p in ('high','low') for m in ('coherence','task')] for a in ARMS}
    by_seed={str(s):{a:{p:assess_point([r for r in groups['zero',p] if r['seed']==s],
                                      [r for r in groups[a,p] if r['seed']==s])['metrics'] for p in ('high','low')}
                     for a in ARMS} for s in SEEDS}
    return dict(effects=effects,by_seed=by_seed,strict_calib_checks={a:_candidate(effects[a],'low') for a in ARMS},
                selected_prompt=selected,selected_mean_matched=matched,
                comparisons=comparisons,pareto=frontier(points),pareto_coordinates=points,
                strict_e0_pass=False,practical_e0_pass=False,capability_evaluated=False,test_evaluated=False)
