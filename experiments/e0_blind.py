"""C9：D5首次盲评、延迟复标和一致性报告；不调用生成模型或评委。"""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.blind import export_first, import_first, export_repeat, import_repeat, write_json
from antler.eval.human import sha256
from e0_extract_coherence import _redacted


def add_arguments(ap):
    ap.add_argument('--action', choices=['export', 'import', 'repeat-export', 'repeat-import'], required=True)
    ap.add_argument('--samples', nargs='+', help='校准或正式测量JSONL，以及C17提取回答补分JSONL')
    ap.add_argument('--bundle', help='首次导出目录中的bundle目录')
    ap.add_argument('--csv', help='作者已填写的盲表')
    ap.add_argument('--repeat-key', help='复标导出的private_repeat_key.json')
    ap.add_argument('--seed', type=int, default=0)


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    now=lambda:datetime.now(timezone.utc).isoformat()
    stamp=now();out=cfg['out_path']/'human_d5'/(stamp.replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    inputs=[Path(args.config)]
    if args.samples:inputs += list(map(Path,args.samples))
    if args.csv:inputs.append(Path(args.csv))
    if args.repeat_key:inputs.append(Path(args.repeat_key))
    if args.bundle:inputs.append(Path(args.bundle)/'private_key.json')
    meta={'task':'C9','action':args.action,'status':'running','start_time':stamp,'end_time':None,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
          'argv':sys.argv[1:],'seed':args.seed,'python':sys.version,'model_loaded':False,'judge_called':False,
          'config_path':str(Path(args.config).resolve()),'config':_redacted({k:v for k,v in cfg.items() if k!='out_path'}),
          'torch':torch.__version__,'transformers':transformers.__version__,'cuda':torch.version.cuda,
          'gpu':None,'judge':None,'input_hashes':{},'data_hashes':{},'formal_e0_pass':False}
    write_json(out/'run_meta.json',meta)
    try:
        meta['input_hashes']={str(p.resolve()):sha256(p) for p in inputs}
        meta['data_hashes']=dict(meta['input_hashes'])
        if args.action=='export':
            if not args.samples:raise ValueError('导出需要--samples')
            key=export_first(args.samples,out/'bundle',args.seed)
            result={'n_samples':100,'n_repeat':20,'strata_counts':key['strata_counts'],'formal_e0_pass':False}
        elif args.action=='import':
            if not args.bundle or not args.csv:raise ValueError('导入需要--bundle及--csv')
            result=import_first(args.bundle,args.csv)
        elif args.action=='repeat-export':
            if not args.bundle:raise ValueError('复标导出需要--bundle')
            result=export_repeat(args.bundle,out/'repeat')
        else:
            if not args.repeat_key or not args.csv:raise ValueError('复标导入需要--repeat-key及--csv')
            result=import_repeat(args.repeat_key,args.csv)
        if any(sha256(p)!=h for p,h in meta['input_hashes'].items()):raise RuntimeError('运行期间输入变化')
        write_json(out/'result.json',result);meta['status']='completed'
    except BaseException:
        meta['status']='failed';raise
    finally:
        meta['end_time']=now();write_json(out/'run_meta.json',meta)
    print(f'[结果目录] {out}')
    return out


if __name__=='__main__':
    run(parse_args(add_arguments))
