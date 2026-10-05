"""C12：第21层有界风险筛查、原幅度写入与双顺序比较。"""
from pathlib import Path
import json
import hashlib
import sys
import uuid

import pyarrow.parquet as pq
import torch

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.delayed_capability import delayed_mmlu, paired_comparison
from antler.eval.twin_gate_audit import read, digest, verify_manifest, lines
from antler.eval.twin_l21 import make_vectors, make_jobs, summarize, pair_jobs, pair_result
from antler.eval.twin_raw_behavior import reused_answers
from antler.memory.twin_risk import geometry
from antler.traits.data import KIT, BIG_FIVE
from e0_diagnostics import now, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model
from e1_twin_context_gate import SequentialJudge, PRIOR_JUDGE, read_materials, seal_sources
from e1_twin_raw_behavior import verify_answer, score_rows, verify_scores
from e1_twin_raw_capability import MMLU_SHA, INDICES_SHA, fixed_sample
from e1_twin_state_extract import RUN_ID, SUMMARY_SHA

STATE_ID = 'fbd8814290174fc8b6ac8df4162855ed'
PROTOCOL = {'layer':21,'phase':'decode','factor':1.,'seed':0,'max_generated':160,
            'reused':90,'max_score_assignments':750,'max_pair_assignments':160,
            'capability_n':200,'max_capability_conditions':3400,'formal_gate_eligible':False}


def extra(ap):
    ap.add_argument('--state-run', required=True)
    ap.add_argument('--reference-run', required=True)
    ap.add_argument('--audit-run', required=True)
    ap.add_argument('--canonical-indices', required=True)
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--materials', default='data/diagnostics/twin_outcome_dev_v1.json')
    ap.add_argument('--execute', action='store_true')


def validate(args):
    state,ref,audit=map(Path,(args.state_run,args.reference_run,args.audit_run))
    for p in (state,ref,audit):
        verify_manifest(p,read(p/'artifact_hashes.json'))
        if read(p/'run_meta.json')['status']!='completed':
            raise ValueError('上游未完成')
    integrity=read(audit/'integrity.json')
    if (ref.name!=RUN_ID or digest(ref/'summary.json')!=SUMMARY_SHA or integrity['status']!='passed'
            or integrity['run_id']!=RUN_ID or integrity['summary_sha256']!=SUMMARY_SHA
            or integrity['run_meta_sha256']!=digest(ref/'run_meta.json')):
        raise ValueError('文本闸门与独立核验来源不符')
    m=read(state/'run_meta.json'); summary=read(state/'summary.json')
    if (state.name!=STATE_ID or m['protocol']['layer']!=21 or summary['layer']!=21
            or m['prompt_sequences_completed']!=540 or m['conditions_completed']!=9
            or m['all_parameters_frozen'] is not True):
        raise ValueError('不是固定第21层提取批次')
    if next(r for r in read(ref/'summary.json')['reports'] if r['scenario_id']=='twin04_ideas')['gate']!='established':
        raise ValueError('情境未进入白名单')
    acts=torch.load(state/'activations.pt',map_location='cpu',weights_only=True)
    b=torch.load(state/'basis.pt',map_location='cpu',weights_only=True)
    if float(b['s'])!=summary['scale']:
        raise ValueError('尺度不符')
    vectors=make_vectors(acts,b['U'],float(b['s']))
    files=sorted((ROOT/'data/hf/mmlu/all').glob('test-*.parquet'))
    if len(files)!=1 or digest(files[0])!=MMLU_SHA or digest(args.canonical_indices)!=INDICES_SHA:
        raise ValueError('能力题库或固定索引不符')
    sample=fixed_sample(pq.read_table(files[0]).to_pylist(),read(args.canonical_indices))[:200]
    materials=read_materials(args.materials)
    reused_answers(make_jobs(materials,vectors,{v['id'] for v in vectors}),lines(ref/'answers.jsonl'))
    return vectors,sample,materials,geometry(acts,float(b['s'])),files


def collect(model,tok,jobs,vectors,reused,*,progress=None,layer=21):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('必须冻结权重且eval')
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    rows=[dict(r) for r in reused]
    for row in rows:
        verify_answer(row,tok,eos)
    lookup={v['id']:v['vector'] for v in vectors}
    new=[j for j in jobs if j['upstream_id'] is None]
    traces=[]
    for offset in range(0,len(new),5):
        group=new[offset:offset+5]
        if len(group)!=5 or len({(j['vector_id'],j['pole']) for j in group})!=1:
            raise ValueError('五题固定批次改变')
        prompts=[chat_prompt(tok,j['q'],system=j['system']) for j in group]
        trace=[]
        answers=generate(model,tok,prompts,steer={layer:lookup[group[0]['vector_id']]},steer_phase='decode',
                         max_new_tokens=768,temperature=1.,top_p=1.,seed=0,batch_size=8,
                         return_details=True,return_token_ids=True,phase_trace_out=trace)
        if len(answers)!=5 or len(trace)!=1 or any(t['applied']!=(t['phase']=='decode') for t in trace[0]['trace']):
            raise ValueError('实际生成预算或decode阶段不符')
        traces.append({'vector_id':group[0]['vector_id'],'pole':group[0]['pole'],'trace':trace})
        for j,p,a in zip(group,prompts,answers):
            row={**j,'prompt':p,'a':a['text'],**{k:a[k] for k in ('token_ids','finish_reason','n_generated_tokens')}}
            verify_answer(row,tok,eos); rows.append(row)
        if progress:
            progress(rows,traces,offset+5)
    return rows


def verify_pairs(out, results, judge):
    """核对成对评委的实际返回模型、请求负载、原文和唯一调用。"""
    requests=lines(out/'pair_requests.jsonl') if results else []
    raw=lines(out/'pair_responses.jsonl') if results else []
    responses={r['key']:r for r in raw}
    if len(requests)!=len(results) or len(raw)!=len(responses) or set(responses)!={r['key'] for r in requests}:
        raise ValueError('成对请求响应数量或身份不符')
    for request,result in zip(requests,results):
        key=hashlib.sha256(f'{judge["model"]}\n{result["prompt"]}'.encode()).hexdigest()
        payload={'model':judge['model'],'temperature':0,'max_tokens':judge['max_tokens'],
                 'messages':[{'role':'user','content':result['prompt']}],**judge['extra_body']}
        expected=hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        response=responses[key]
        if (request['key']!=key or any(request[k]!=v for k,v in result.items() if k not in ('choice','response'))
                or response['request_sha256']!=expected or response['requested_model']!=judge['model']
                or response['returned_model']!=judge['model'] or response['finish_reason']!='stop'
                or result['response']!=response['output'].strip()):
            raise ValueError('成对评委身份或原文关联异常')
    return len(responses)


def run(args):
    cfg,_,_,_=setup(args,need_model=False,need_judge=False)
    vectors,sample,materials,geo,files=validate(args)
    if not args.execute:
        print('[21层预检] 16向量、200题能力筛查、最多160新回答、750评分、160成对请求；未执行',flush=True)
        return PROTOCOL
    model_hashes=verify_model((ROOT/cfg['model']).resolve(),args.model_fingerprint)
    root=Path(cfg['out_path'])/'e1_twin_l21_pilot'; root.mkdir(parents=True,exist_ok=True)
    out=root/uuid.uuid4().hex
    with (root/(STATE_ID+'.claim')).open('x',encoding='utf8') as f:f.write(str(out.resolve()))
    out.mkdir()
    meta={'task':'C12','status':'running','stage':'validation','start_time':now(),'end_time':None,
          'protocol':PROTOCOL,'git_commit':_git('rev-parse','HEAD'),'config':_redacted(cfg),
          'config_path':str(Path(args.config).resolve()),'argv':_redacted(sys.argv[1:]),
          'model_hashes':model_hashes,'generated':0,'scored':0,'pair_scored':0,
          'formal_e1_pass':False,'behavior_validated':False}
    save=lambda:write_json(out/'run_meta.json',meta)
    save(); rows=[]
    try:
        judge=SequentialJudge(cache_path=str(out/'judge_cache.jsonl'),max_workers=1,
                              response_audit_path=str(out/'judge_responses.jsonl'))
        actual={'model':judge.model,'base_url':judge.base_url,'max_tokens':judge.max_tokens,'extra_body':judge.extra_body}
        if actual!=read(ROOT/PRIOR_JUDGE)['judge'] or judge._cache:
            raise ValueError('评委或缓存改变')
        sources=[Path(args.config),Path(args.materials),Path(args.model_fingerprint),Path(args.canonical_indices),*files,
                 ROOT/PRIOR_JUDGE,ROOT/'docs/RESEARCH_SPEC.md',ROOT/'docs/E1_L21_SMALL_PILOT_2026-10-05.md']
        sources+=list((ROOT/'antler').rglob('*.py'))+list((ROOT/'experiments').glob('*.py'))
        sources += [KIT/'trait_eval'/(name+'.json') for name in BIG_FIVE['E']]
        for folder in (args.state_run,args.reference_run,args.audit_run):
            sources += [p for p in Path(folder).iterdir() if p.is_file()]
        meta.update(judge=actual,data_hashes={str(p.resolve()):digest(p) for p in sources})
        seal_sources(out,meta['data_hashes']);save()
        def check():
            if any(digest(p)!=h for p,h in meta['data_hashes'].items()):
                raise ValueError('运行期间固定来源改变')
        write_json(out/'geometry.json',geo);write_json(out/'sample.json',sample)
        torch.save({v['id']:v['vector'] for v in vectors},out/'vectors.pt')
        write_json(out/'vectors.json',[{k:v for k,v in row.items() if k!='vector'} for row in vectors])
        model,tok=load_model(cfg['model'],cfg.get('dtype','bfloat16'),cfg.get('device','auto'))
        meta.update(stage='capability',all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                    stop_config={'model_eos_token_id':model.generation_config.eos_token_id})
        save(); baseline=delayed_mmlu(model,tok,sample,batch_size=16)
        write_json(out/'capability_zero.json',baseline)
        caps={}; admitted=set()
        for v in vectors:
            pred=delayed_mmlu(model,tok,sample,steer={21:v['vector']},batch_size=16,reference_first=baseline['samples'])
            write_json(out/('capability_'+v['id']+'.json'),pred)
            report=paired_comparison(baseline['samples'],pred['samples'])
            caps[v['id']]=report
            if report['descriptive_loss_le_2pp']:admitted.add(v['id'])
            write_json(out/'capability_summary.json',caps);check()
            print(f'[21层能力筛查] {len(caps)}/16，{v["id"]}，下降{report["loss_percentage_points"]:.1f}pp',flush=True)
        jobs=make_jobs(materials,vectors,admitted)
        reused=reused_answers(jobs,lines(Path(args.reference_run)/'answers.jsonl'))
        write_json(out/'jobs.json',jobs)
        meta.update(stage='generate',admitted=sorted(admitted));save()
        def progress(prefix,traces,n):
            nonlocal rows
            rows=prefix;write_json(out/'answers.json',rows);write_json(out/'generation_traces.json',traces)
            meta['generated']=n;save();check()
            print(f'[21层生成] {n}/{len(jobs)-90}',flush=True)
        rows=collect(model,tok,jobs,vectors,reused,progress=progress)
        write_json(out/'answers.json',rows)
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        meta['stage']='score';save()
        def scored(prefix,n):
            meta['scored']=n
            if n%25==0:save()
        score_rows(judge,rows,out,progress=scored,check=check)
        meta['unique_scalar_requests']=verify_scores(out,rows,actual)
        write_json(out/'summary.json',summarize(rows))
        # 单项评分核验结束后才开始独立成对请求，两者响应账本分开。
        pairs=pair_jobs(rows);write_json(out/'pair_jobs.json',pairs)
        pairjudge=SequentialJudge(cache_path=str(out/'pair_cache.jsonl'),max_workers=1,
                                  response_audit_path=str(out/'pair_responses.jsonl'))
        pairactual={'model':pairjudge.model,'base_url':pairjudge.base_url,'max_tokens':pairjudge.max_tokens,'extra_body':pairjudge.extra_body}
        if pairactual!=actual:raise ValueError('成对评委改变')
        results=[]
        for j in pairs:
            check()
            with (out/'pair_requests.jsonl').open('a',encoding='utf8',newline='\n') as f:
                f.write(json.dumps({**j,'key':pairjudge._key(j['prompt'])},ensure_ascii=False)+'\n')
            answer=pairjudge._call(j['prompt']).strip()
            results.append({**j,'response':answer,'choice':answer if answer in ('A','B','T') else None})
            write_json(out/'pair_results.json',results)
            meta['pair_scored']=len(results);save()
        combined=[]
        for key in sorted({j['pair_id'] for j in pairs}):
            group=[j for j in results if j['pair_id']==key]
            ab=next(j for j in group if not j['reverse']);ba=next(j for j in group if j['reverse'])
            combined.append({'pair_id':key,'comparison':ab['comparison'],'pole':ab['pole'],
                             'question_index':ab['question_index'],'matched_win':pair_result(ab['choice'],ba['choice'])})
        meta['unique_pair_requests']=verify_pairs(out,results,actual)
        write_json(out/'pair_summary.json',{'pairs':combined,'formal_pass':False,
                   'limits':'AB/BA是同一对的重复测量；按题目聚合，不把80对当作独立情境。'})
        check()
        if any(digest(p)!=h for p,h in model_hashes.items()):raise ValueError('模型文件改变')
        if len(rows)!=len(jobs) or meta['scored']!=3*len(jobs) or meta['pair_scored']!=len(pairs):
            raise ValueError('终态预算不符')
        meta.update(status='completed',stage='completed')
    except BaseException as exc:
        meta.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        write_json(out/'scored_answers.json',rows);meta['end_time']=now();save()
        write_json(out/'artifact_hashes.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*')
                                             if p.is_file() and p.name!='artifact_hashes.json'})
    print(f'[21层完成] {out}',flush=True)
    return out


if __name__=='__main__':
    run(parse_args(extra))
