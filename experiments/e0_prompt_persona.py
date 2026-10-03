"""C21：固定Careless Prompt对照，同轮乱序评分，不使用test或替换Strict。"""
import hashlib
import json
from pathlib import Path
import random
import sys
import uuid

import torch
import transformers

from _common import parse_args,setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt,load_model
from antler.eval.judge import Judge,COHERENCE_PROMPT,TASK_PROMPT
from antler.eval.prompt_persona import ARMS,PROMPTS,SEEDS,analyze,validate
from antler.traits.data import KIT,load_trait
from e0_diagnostics import now,sha256,write_json
from e0_extract_coherence import _redacted
from e0_generated_capability import SOURCE,VECTOR_SHA,META_SHA,finite_logits
from e0_phase_diagnostics import score_rows,validate_generation
from e0_refilter import _git


C_HASHES={'original':'849b1182e122bfaec498fc46e6e83553ef8a48a9940278302d3937208d014255',
          'projected':'7c7663b55ea18741995b0e0721e6c81f4469d65d6da594482575390e1db73051'}


def load_sources(cfg):
    path=SOURCE/'injection_vectors.pt'; meta_path=SOURCE/'run_meta.json'
    if sha256(path)!=VECTOR_SHA or sha256(meta_path)!=META_SHA:
        raise ValueError('候选来源哈希不符')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    vectors=saved['vectors']['C']
    if saved['layer']!=21:
        raise ValueError('候选层号不符')
    for arm,digest in C_HASHES.items():
        v=vectors[arm]
        if (v.shape!=(2560,) or v.dtype!=torch.float32 or not torch.isfinite(v).all()
                or hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()!=digest):
            raise ValueError('C原始或剥离张量不符')
    old=json.loads(meta_path.read_text(encoding='utf-8'))
    if old['status']!='completed' or old['config']['model']!=cfg['model']:
        raise ValueError('来源未完成或模型身份不同')
    model_dir=(ROOT/cfg['model']).resolve()
    if not (model_dir/'config.json').is_file():
        raise ValueError('必须使用已有本地模型')
    files=sorted({p for pattern in ('*.json','*.txt','*.model','*.tiktoken','*.jinja') for p in model_dir.rglob(pattern) if p.is_file()})
    files += [KIT/'trait_eval'/f'{p}.json' for p in ('dependable','careless')]
    for p in files:
        if old['data_hashes'].get(str(p.resolve()))!=sha256(p):
            raise ValueError('模型/tokenizer或题库与候选来源不符')
    return vectors,old,files+[path,meta_path],model_dir


def run(args):
    cfg,_,_,_=setup(args,need_model=False,need_judge=False)
    out=cfg['out_path']/'prompt_persona'/uuid.uuid4().hex
    out.mkdir(parents=True,exist_ok=False)
    meta=dict(task='C21',status='running',stage='validation',start_time=now(),git_commit=_git('rev-parse','HEAD'),
              git_dirty=bool(_git('status','--porcelain')),config=_redacted(cfg),argv=_redacted(sys.argv[1:]),
              torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,gpu=None,judge=None,
              seeds=SEEDS,layer=21,coefficient=-.225,steer_phase='decode',max_new_tokens=320,batch_size=16,
              expected_answers=1080,generated=0,strict_e0_pass=False,practical_e0_pass=False,split='calib')
    rows=[];guard=None
    def save():
        write_json(out/'run_meta.json',meta)
        tmp=out/'samples.tmp'
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
        tmp.replace(out/'samples.jsonl')
    save()
    try:
        vectors,old,files,model_dir=load_sources(cfg)
        traits={p:load_trait(p,'eval') for p in ('dependable','careless')}
        if any(len(t['questions'])!=20 or len(set(t['questions'][:10]))!=10 for t in traits.values()):
            raise ValueError('题库calib/test切分不符')
        files += [Path(args.config),ROOT/'docs/E0_PROMPT_PERSONA_2026-10-04.md',ROOT/'docs/RESEARCH_SPEC.md']
        files += list((ROOT/'antler').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))
        meta['data_hashes']={str(p.resolve()):sha256(p) for p in files}
        def check_sources():
            if any(sha256(Path(p))!=h for p,h in meta['data_hashes'].items()):
                raise ValueError('运行期间输入变化')
        templates={'coherence':COHERENCE_PROMPT,'task':TASK_PROMPT,**{p:t['eval_prompt'] for p,t in traits.items()}}
        meta['prompt_hashes']={p:hashlib.sha256(t.encode()).hexdigest() for p,t in templates.items()}
        if any(old['prompt_hashes'].get(p)!=h for p,h in meta['prompt_hashes'].items()):
            raise ValueError('评分提示与候选来源不一致')
        write_json(out/'personas.json',PROMPTS)
        judge=Judge(cache_path=str(out/'judge_cache.jsonl'),response_audit_path=str(out/'judge_responses.jsonl'))
        meta['judge']=_redacted({k:getattr(judge,k) for k in ('model','base_url','max_tokens','extra_body')})
        if judge.model!='deepseek-flash' or meta['judge']!=old['judge']:
            raise ValueError('固定评委身份或参数发生变化')
        meta['stage']='model_load';save()
        model,tok=load_model(str(model_dir),cfg.get('dtype','bfloat16'),cfg.get('device','auto'))
        if (model.config.num_hidden_layers,model.config.hidden_size)!=(36,2560) or any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型形状或冻结状态不符')
        guard=model.register_forward_hook(finite_logits)
        meta['gpu']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out/'generation_settings.json',model.generation_config.to_dict())
        eos=model.generation_config.eos_token_id
        eos=set(eos if isinstance(eos,list) else [eos])
        rendered={a:{p:[chat_prompt(tok,q,system=PROMPTS.get(a)) for q in t['questions'][:10]] for p,t in traits.items()} for a in ARMS}
        write_json(out/'chat_prompts.json',rendered)
        preflight=[generate(model,tok,rendered['zero']['careless'][:2],steer=v,steer_phase='decode',
                            max_new_tokens=8,temperature=0,batch_size=16,seed=0,return_details=True,return_token_ids=True)
                   for v in (None,{21:torch.zeros_like(vectors['projected'])})]
        write_json(out/'preflight.json',preflight)
        if preflight[0]!=preflight[1]:
            raise ValueError('零张量预检不相等')
        jobs=[(a,p,s) for a in ARMS for p in ('high','low') for s in SEEDS]
        random.Random(0).shuffle(jobs);write_json(out/'generation_order.json',jobs)
        meta['stage']='generation';save()
        for arm,pole,seed in jobs:
            check_sources()
            target='dependable' if pole=='high' else 'careless'
            traces=[]
            items=generate(model,tok,rendered[arm][target]*3,steer={21:vectors[arm]} if arm in vectors else None,
                           steer_phase='decode',max_new_tokens=320,temperature=1,top_p=1,batch_size=16,seed=seed,
                           return_details=True,return_token_ids=True,phase_trace_out=traces)
            validate_generation(items,30,320,eos)
            for batch in traces:
                trace=batch['trace']
                if not trace or trace[0]['phase']!='prefill' or trace[0]['applied']:
                    raise ValueError('prefill轨迹错误')
                if any(t['phase']!='decode' or t['applied']!=(arm in vectors) for t in trace[1:]):
                    raise ValueError('decode注入轨迹错误')
            with (out/'phase_traces.jsonl').open('a',encoding='utf-8',newline='\n') as f:
                f.write(json.dumps(dict(arm=arm,pole=pole,seed=seed,batches=traces))+'\n')
            for index,item in enumerate(items):
                if tok.batch_decode([item['token_ids']],skip_special_tokens=True)[0]!=item['text']:
                    raise ValueError('token与文本不一致')
                rows.append(dict(arm=arm,pole=pole,seed=seed,dim='C',target=target,layer=21,
                                 coef=-.225 if arm in vectors else 0,steer_phase='decode',split='calib',
                                 question_index=index%10,sample_index=index//10,q=traits[target]['questions'][index%10],
                                 a=item['text'],**{k:item[k] for k in ('token_ids','n_generated_tokens','finish_reason')}))
            meta['generated']=len(rows);save()
            print(f'[Prompt对照] {arm}/{pole}/seed{seed}，{len(rows)}/1080条',flush=True)
        validate(rows,traits)
        guard.remove();guard=None
        del model,tok
        if torch.cuda.is_available():torch.cuda.empty_cache()
        check_sources();meta['stage']='scoring';save()
        complete=score_rows(judge,rows,traits,out);save()
        if not complete:raise ValueError('有缺失评分，不补评')
        check_sources();meta['stage']='analysis';save()
        write_json(out/'assessment.json',analyze(rows,traits))
        check_sources();meta.update(status='completed',stage='completed')
    except BaseException:
        meta['status']='failed'
        raise
    finally:
        if guard is not None:guard.remove()
        meta['end_time']=now();save()
        write_json(out/'artifact_hashes.json',{p.name:sha256(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_hashes.json'})
        print(f'[结果目录] {out}',flush=True)
    return out


if __name__=='__main__':
    run(parse_args())
