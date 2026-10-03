"""C5：固定N剥离候选的生成式MMLU诊断，不替换正式能力闸门。"""
import json
from pathlib import Path
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import load_model
from antler.eval.generated_capability import prompt, parse_answer, exposure, summarize
from e0_delayed_capability import fixed_inputs
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git

SOURCE = ROOT/'outputs/main/quality_projection/2026-10-03T202637.350816_0000_38bbf8b3'
VECTOR_SHA = 'b20e20713aa4f506965c7e976a675d14b7700e1c61f6f8bf70ebc07d42f47510'
N_SHA = 'fa2984b545fa63f488018f3a358ec0320a69e0e08092ce5252a0d7656c665617'
META_SHA = 'e6cda36cfa9e7fbddf93e8b7c5447f5a9f47f6cd905dd582ef730c34eb8000be'


def finite_logits(module, args, output):
    """直接检查生成用logit，非有限值不能静默变成某个选项。"""
    if not torch.isfinite(output.logits).all():
        raise ValueError('模型产生非有限logit，停止')


def add_arguments(ap):
    ap.add_argument('--sample-indices', required=True)
    ap.add_argument('--reference-baseline', required=True)
    ap.add_argument('--data', default='data/hf/mmlu/all')


def candidate():
    import hashlib
    p = SOURCE/'injection_vectors.pt'
    if sha256(p) != VECTOR_SHA:
        raise ValueError('实际注入张量文件不符合登记')
    saved = torch.load(p, map_location='cpu', weights_only=True)
    v = saved['vectors']['N']['projected']
    if (saved['layer'] != 21 or v.dtype != torch.float32 or v.shape != (2560,)
            or not torch.isfinite(v).all()
            or hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() != N_SHA):
        raise ValueError('N候选张量形状、数值或字节不符')
    return v


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = cfg['out_path']/'generated_capability'/uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(task='C5', status='running', start_time=now(), git_commit=_git('rev-parse','HEAD'),
                git_dirty=bool(_git('status','--porcelain')), config=_redacted(cfg), argv=_redacted(sys.argv[1:]),
                torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                gpu=None, judge=None, seed=0, max_new_tokens=192, batch_size=16,
                layer=21, coef=.20, phase='decode', expected_answers=2800, completed_answers=0,
                formal_gate_eligible=False, replaces_canonical_gate=False)
    write_json(out/'run_meta.json',meta)
    results = []
    guard = None
    try:
        rows, _, files = fixed_inputs(args,cfg)
        vector = candidate()
        model_dir = (ROOT/cfg['model']).resolve()
        if sha256(SOURCE/'run_meta.json') != META_SHA:
            raise ValueError('候选来源元数据哈希不符')
        old = json.loads((SOURCE/'run_meta.json').read_text(encoding='utf-8'))
        if old['status']!='completed' or old['config']['model']!=cfg['model']:
            raise ValueError('候选来源未完成或模型身份不同')
        model_files = sorted({p for pattern in ('*.json','*.txt','*.model','*.tiktoken','*.jinja')
                              for p in model_dir.rglob(pattern) if p.is_file()})
        if not model_files or not (model_dir/'config.json').is_file():
            raise ValueError('只允许已下载本地模型')
        for p in model_files:
            if old['data_hashes'].get(str(p))!=sha256(p):
                raise ValueError('模型/tokenizer文件与候选来源不符')
        files += model_files + [SOURCE/'injection_vectors.pt', SOURCE/'run_meta.json', Path(args.config),
                               ROOT/'docs/E0_GENERATED_CAPABILITY_2026-10-04.md']
        files += list((ROOT/'antler').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))
        meta['data_hashes']={str(p.resolve()):sha256(p) for p in files}
        write_json(out/'run_meta.json',meta)
        write_json(out/'sample_indices.json',[{k:r[k] for k in ('source_index','subject','answer')} for r in rows])
        model,tok = load_model(str(model_dir),cfg.get('dtype','bfloat16'),cfg.get('device','auto'))
        if model.config.num_hidden_layers!=36 or model.config.hidden_size!=2560 or any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型形状不符或未冻结')
        guard=model.register_forward_hook(finite_logits)
        meta['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        prompts = [prompt(tok,r) for r in rows]
        write_json(out/'prompts.json',prompts)
        write_json(out/'generation_settings.json',model.generation_config.to_dict())
        preflight=[]
        for v in (None,{21:torch.zeros_like(vector)}):
            preflight.append(generate(model,tok,prompts[:2],steer=v,steer_phase='decode',max_new_tokens=8,
                                      temperature=0,batch_size=16,seed=0,return_details=True,return_token_ids=True))
        write_json(out/'preflight.json',preflight)
        if preflight[0]!=preflight[1]:
            raise ValueError('零向量改变生成，停止')
        for start in range(0,1400,16):
            order=('zero','N_projected') if (start//16)%2==0 else ('N_projected','zero')
            for arm in order:
                traces=[]
                items=generate(model,tok,prompts[start:start+16],steer=None if arm=='zero' else {21:vector},
                               steer_phase='decode',max_new_tokens=192,temperature=0,batch_size=16,seed=0,
                               return_details=True,return_token_ids=True,phase_trace_out=traces)
                if len(items)!=len(rows[start:start+16]) or len(traces)!=1:
                    raise ValueError('生成条数或批轨迹异常')
                trace=traces[0]['trace']
                if any(t['applied'] != (arm!='zero' and t['phase']=='decode') for t in trace):
                    raise ValueError('实际注入时段异常')
                with (out/'traces.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                    f.write(json.dumps(dict(arm=arm,batch_start=start,trace=trace))+'\n')
                for row,item in zip(rows[start:start+16],items):
                    parsed=parse_answer(item['text'])
                    record={k:row[k] for k in ('source_index','subject','answer')}
                    record.update(arm=arm,**item,**parsed,**exposure(tok,item,trace,parsed))
                    record['correct']=parsed['prediction'] is not None and parsed['prediction']==row['answer']
                    results.append(record)
                    with (out/'samples.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                meta['completed_answers']=len(results)
                write_json(out/'run_meta.json',meta)
            print(f'[能力诊断] 已完成{len(results)}/2800条',flush=True)
        write_json(out/'summary.json',summarize(results))
        if any(sha256(Path(p))!=h for p,h in meta['data_hashes'].items()):
            raise ValueError('运行中输入变化')
        meta['status']='completed'
    except BaseException:
        meta['status']='failed'
        raise
    finally:
        if guard is not None:
            guard.remove()
        meta['end_time']=now()
        meta['artifact_hashes']={p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='run_meta.json'}
        write_json(out/'run_meta.json',meta)
        print(f'[结果目录] {out}',flush=True)
    return out


if __name__=='__main__':
    run(parse_args(add_arguments))
