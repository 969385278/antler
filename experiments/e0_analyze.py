"""C4：只读校准CSV及样本，报告曲线、逐方向门槛与层内Gram矩阵。"""
import csv
import json
from pathlib import Path
import subprocess
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.calibration import spearman, assess_point
from antler.traits.data import DIMS
from e0_diagnostics import now, sha256, write_json
from e0_calibrate import _redacted


def add_arguments(ap):
    ap.add_argument('--csv', nargs='*', default=[])
    ap.add_argument('--samples', nargs='*', default=[])
    ap.add_argument('--directions', help='可选：导出输入曲线涉及层的Gram矩阵')
    ap.add_argument('--plot', action='store_true', help='可选：使用matplotlib保存曲线图')


def analyze_csv(path):
    groups={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            key=(int(row['layer']),row['dim'],row.get('pole') or 'high')
            groups.setdefault(key,[]).append(row)
    results=[]
    for (layer,dim,pole), rows in sorted(groups.items()):
        rows.sort(key=lambda r:float(r['coef']))
        if len({float(r['coef']) for r in rows}) != len(rows):
            raise ValueError('同一曲线有重复系数，请分开分析不同实验')
        valid=[r for r in rows if r.get('trait') not in (None,'')]
        x=[float(r['coef']) for r in valid];y=[float(r['trait']) for r in valid]
        cohs=[float(r['coherence']) for r in rows if r.get('coherence') not in (None,'')]
        results.append({'layer':layer,'dim':dim,'pole':pole,'coefs':x,'traits':y,
                        'spearman_expected':spearman(x,y if pole=='high' else [-v for v in y]),
                        'range_effect':(y[-1]-y[0])*(1 if pole=='high' else -1) if len(y)>1 else None,
                        'minimum_coherence':min(cohs) if cohs else None,
                        'complete_traits':len(valid)==len(rows), 'quality_measured':len(cohs)==len(rows)})
    # 旧版统一层筛选只报告高极全五维；无质量测量的MCQ不算生成质量通过。
    eligible=[]
    for layer in sorted({r['layer'] for r in results}):
        high=[r for r in results if r['layer']==layer and r['pole']=='high']
        if (set(r['dim'] for r in high)==set(DIMS) and all(r['complete_traits'] and r['quality_measured']
                and r['spearman_expected'] is not None and r['spearman_expected']>=.9
                and r['minimum_coherence']>=75 for r in high)):
            eligible.append({'layer':layer,'minimum_range_effect':min(r['range_effect'] for r in high)})
    best=max((r['minimum_range_effect'] for r in eligible),default=None)
    tied=[r for r in eligible if best-r['minimum_range_effect']<=5] if best is not None else []
    return {'source':str(path),'curves':results,'legacy_eligible_layers':eligible,
            'legacy_candidate':tied[0] if len(tied)==1 else None,
            'needs_capability_tiebreak':len(tied)>1,'tied_candidates':tied}


def analyze_samples(path):
    records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    results=[]
    for record in records:
        coefficient=record['coef'];pole=record['pole']
        if not ((coefficient>0 and pole=='high') or (coefficient<0 and pole=='low')):
            continue
        baselines=[r for r in records if r['coef']==0 and r['dim']==record['dim'] and r['pole']==pole
                   and r['layer']==record['layer'] and r.get('axis','bipolar')==record.get('axis','bipolar')
                   and r.get('seed',0)==record.get('seed',0)]
        if len(baselines)!=1:
            raise ValueError('每个条件必须有唯一匹配的零注入基线')
        results.append({'source':str(path),**{k:record.get(k) for k in ['layer','dim','pole','coef','axis','split','seed']},
                        **assess_point(baselines[0]['result']['samples'],record['result']['samples'])})
    return results


def run(args):
    cfg,_,_,_=setup(args,need_judge=False,need_model=False)
    files=[Path(p).resolve() for p in [*args.csv,*args.samples]]
    if not files:
        raise ValueError('至少提供一个CSV或样本文件')
    if args.directions:files.append(Path(args.directions).resolve())
    out=cfg['out_path']/'analysis'/(now().replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    meta={'task':'C4','status':'running','start_time':now(),'end_time':None,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
          'config_path':args.config,'config':_redacted({k:v for k,v in cfg.items() if k!='out_path'}),
          'argv':sys.argv[1:],'seed':0,'torch':torch.__version__,'transformers':transformers.__version__,
          'cuda':torch.version.cuda,'gpu':None,'judge':None,
          'data_hashes':{str(p):sha256(p) for p in files},'formal_e0_pass':False}
    write_json(out/'run_meta.json',meta)
    try:
        curves=[analyze_csv(p) for p in args.csv]
        points=[point for p in args.samples for point in analyze_samples(p)]
        write_json(out/'curves.json',curves);write_json(out/'points.json',points)
        if args.directions:
            d=torch.load(args.directions,map_location='cpu',weights_only=True)
            layers=sorted({r['layer'] for file in curves for r in file['curves']}|{r['layer'] for r in points})
            write_json(out/'gram.json',{str(l):(d['U'][l].float()@d['U'][l].float().T).tolist() for l in layers})
        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for i,file in enumerate(curves):
                dims=[d for d in DIMS if any(r['dim']==d for r in file['curves'])]
                fig,axes=plt.subplots(1,len(dims),figsize=(max(5,5*len(dims)),3),sharey=True,squeeze=False)
                for ax,dim in zip(axes[0],dims):
                    for row in file['curves']:
                        if row['dim']==dim:ax.plot(row['coefs'],row['traits'],marker='o',label=f"L{row['layer']} {row['pole']}")
                    ax.set(title=dim,xlabel='Coefficient',ylim=(0,100))
                    if ax.lines:ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(1.02,1))
                axes[0][0].set_ylabel('Trait score')
                fig.tight_layout();fig.savefig(out/f'curves_{i}.png',dpi=160);plt.close(fig)
        lines=['# E0校准分析','','以下为数字筛选，不代表正式E0通过。','', '|来源|维度/极点|系数|数字门槛|未满足项|','|-|-|-|-|-|']
        for p in points:
            lines.append(f"|{Path(p['source']).parent.name}|{p['dim']}/{p['pole']}|{p['coef']}|{p['numeric_criteria_met']}|{', '.join(p['failed_checks'])}|")
        (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
        if any(sha256(Path(p))!=h for p,h in meta['data_hashes'].items()):raise RuntimeError('分析期间输入变化')
        meta['status']='completed'
    except BaseException:
        meta['status']='failed';raise
    finally:
        meta['end_time']=now();write_json(out/'run_meta.json',meta)
        print(f'[结果目录] {out}')
    return out


if __name__=='__main__':
    run(parse_args(add_arguments))
