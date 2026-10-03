"""C4：第8.3节行为串扰矩阵；行是测量维度，列是注入维度。"""
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.model import load_model
from antler.eval.judge import Judge, COHERENCE_PROMPT, TASK_PROMPT
from antler.eval.trait_score import trait_score
from antler.traits.data import DIMS, BIG_FIVE, KIT
from antler.traits.vectors import steer_vector, steer_vector_poles
from e0_calibrate import _redacted
from e0_diagnostics import now, sha256, write_json


def add_arguments(ap):
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--coefs', type=float, nargs=5, required=True, help='O/C/E/A/N的五个正向系数')
    ap.add_argument('--axis', choices=['bipolar','pole'], default='bipolar')
    ap.add_argument('--split', choices=['calib','test'], default='calib')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-samples', type=int, default=3)


def measure_matrix(measure, vectors, save):
    """只生成五组基线；其余25组分别施加一个维度的向量。"""
    baseline={};conditions={}
    for dim in DIMS:
        baseline[dim]=measure(dim,None)
        save({'injected_dim':None,'measured_dim':dim,'result':baseline[dim]})
    for injected in DIMS:
        conditions[injected]={}
        for measured in DIMS:
            result=measure(measured,vectors[injected])
            conditions[injected][measured]=result
            save({'injected_dim':injected,'measured_dim':measured,'baseline_ref':measured,'result':result})
    matrix=[]
    for measured in DIMS:
        row=[]
        for injected in DIMS:
            a,b=baseline[measured]['trait'],conditions[injected][measured]['trait']
            row.append(None if a is None or b is None else b-a)
        matrix.append(row)
    complete=all(v is not None and math.isfinite(v) for row in matrix for v in row)
    ratio=None
    if complete:
        denominator=sum(abs(matrix[i][i]) for i in range(5))/5
        if denominator>0:
            ratio=max(abs(matrix[i][j]) for i in range(5) for j in range(5) if i!=j)/denominator
    return {'dims':DIMS,'rows':'measured_dim','columns':'injected_dim','matrix':matrix,
            'complete':complete,'max_abs_offdiag_over_mean_abs_diag':ratio,
            'formal_e0_pass':False,'note':'量表端点可能掩盖串扰；本报告不单独判定E0通过'}


def run(args):
    cfg,_,_,_=setup(args,need_judge=False,need_model=False)
    if (any(not math.isfinite(c) or c<=0 for c in args.coefs) or args.n_samples<1
            or cfg.get('calib_questions',10)!=10 or not cfg.get('use_judge',True)):
        raise ValueError('需要五个有限正系数、正采样数、10道题和原评委')
    source=cfg['out_path'];path=source/'directions.pt'
    d=torch.load(path,map_location='cpu',weights_only=True);U,s=d['U'],d['s']
    if d.get('dims')!=DIMS or not 0<=args.layer<len(U) or not torch.isfinite(U).all() or not torch.isfinite(s).all() or s[args.layer]<=0:
        raise ValueError('方向、层号或尺度无效')
    files=[path,Path(args.config).resolve(),Path(__file__).resolve()]
    files += [KIT/'trait_eval'/f'{BIG_FIVE[k][0]}.json' for k in DIMS]
    poles={}
    if args.axis=='pole':
        for names in BIG_FIVE.values():
            for name in names:
                p=source/'vectors'/f'{name}.pt';v=torch.load(p,map_location='cpu',weights_only=True)
                if v.shape!=(U.shape[0],U.shape[2]):raise ValueError('单极向量形状不匹配')
                poles[name]=v[args.layer];files.append(p)
    vectors={}
    for i,dim in enumerate(DIMS):
        z=torch.zeros(5);z[i]=args.coefs[i]
        vectors[dim]={args.layer:(steer_vector_poles(poles,float(s[args.layer]),z) if args.axis=='pole'
                                  else steer_vector(U[args.layer],float(s[args.layer]),z))}
    out=source/'crosstalk'/(now().replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True)
    if (source/'judge_cache.jsonl').exists():shutil.copy2(source/'judge_cache.jsonl',out/'judge_cache.jsonl')
    import hashlib
    meta={'task':'C4','experiment':'behavioral_crosstalk','status':'running','start_time':now(),'end_time':None,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
          'config_path':args.config,'config':_redacted({k:v for k,v in cfg.items() if k!='out_path'}),'argv':sys.argv[1:],
          'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
          'seed':args.seed,'n_samples':args.n_samples,'split':args.split,'axis':args.axis,'layer':args.layer,'coefs':args.coefs,
          'max_new_tokens':320,'data_hashes':{str(p):sha256(p) for p in files},'task_score_validation':'unvalidated',
          'prompt_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in [('coherence',COHERENCE_PROMPT),('task',TASK_PROMPT)]}}
    write_json(out/'run_meta.json',meta)
    try:
        model,tok=load_model(cfg['model'],cfg.get('dtype','bfloat16'),cfg.get('device','auto'))
        if (model.config.num_hidden_layers,model.config.hidden_size)!=(U.shape[0],U.shape[2]):raise ValueError('模型与方向不匹配')
        judge=Judge(cache_path=str(out/'judge_cache.jsonl'))
        meta['judge']=_redacted({k:getattr(judge,k,None) for k in ('model','base_url','max_tokens','extra_body')})
        meta['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out/'run_meta.json',meta)
        with (out/'samples.jsonl').open('w',encoding='utf-8',newline='\n') as sf:
            def save(row):
                sf.write(json.dumps(row,ensure_ascii=False)+'\n');sf.flush()
                print(f"[测量] 注入{row['injected_dim']}→测量{row['measured_dim']}：{row['result']['trait']}",flush=True)
            def measure(dim,steer):
                return trait_score(model,tok,judge,dim,steer=steer,pole='high',split=args.split,
                                   n_questions=10,n_samples=args.n_samples,seed=args.seed,max_new_tokens=320,with_task=True)
            write_json(out/'matrix.json',measure_matrix(measure,vectors,save))
        if any(sha256(Path(p))!=h for p,h in meta['data_hashes'].items()):raise RuntimeError('实验期间输入发生变化')
        meta['status']='completed'
    except BaseException:
        meta['status']='failed';raise
    finally:
        meta['end_time']=now();meta['artifact_hashes']={p.name:sha256(p) for p in [out/'samples.jsonl',out/'matrix.json'] if p.exists()}
        write_json(out/'run_meta.json',meta);print(f'[结果目录] {out}',flush=True)
    return out


if __name__=='__main__':
    run(parse_args(add_arguments))
