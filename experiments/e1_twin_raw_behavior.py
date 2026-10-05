"""C12：固定原幅度行为采集与同轮盲评；默认离线预检。"""
from pathlib import Path
import hashlib
import json
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.capability import capability_summary
from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT, parse_score
from antler.eval.twin_gate_audit import digest, read, lines, verify_manifest, TRAIT_HASHES
from antler.eval.twin_raw_behavior import PROTOCOL, METRICS, build_jobs, reused_answers, score_order, summarize
from antler.traits.data import BIG_FIVE, KIT, load_trait
from e0_diagnostics import now, write_json
from e0_extract_coherence import _redacted
from e0_refilter import _git
from e1_probe_repeat import verify_model
from e1_twin_context_gate import SequentialJudge, read_materials, seal_sources, PRIOR_JUDGE
from e1_twin_raw_capability import fixed_vectors, STATE_RUN, STATE_SHA, INDICES_SHA, PROTOCOL as CAP_PROTOCOL
from e1_twin_state_extract import RUN_ID, SUMMARY_SHA

EMPTY_FAILURE_RUN = 'b342ea30e4e1477fa650b0a1687957ce'
EMPTY_FAILURE_SHA = 'a997288c97fc2f263a3da75ba2e92ff7943fa14fa8ed4505809ef4c32fd643bb'


def extra(ap):
    ap.add_argument('--state-run', required=True)
    ap.add_argument('--capability-run', required=True)
    ap.add_argument('--reference-run', required=True)
    ap.add_argument('--audit-run', required=True)
    ap.add_argument('--materials', default='data/diagnostics/twin_outcome_dev_v1.json')
    ap.add_argument('--model-fingerprint', required=True)
    ap.add_argument('--execute', action='store_true', help='固定一次行为采集/评分，不重试或追加')
    ap.add_argument('--approved-empty-restart', action='store_true',
                    help='仅在作者明确批准后，恢复已封存的零生成/零请求初始化失败；默认关闭')


def output_root(root, approved=False):
    """原claim和失败归档不变；显式批准仅可开放一次独立恢复claim。"""
    if not approved:
        return root
    claim = root / (STATE_RUN + '.claim')
    failed = Path(claim.read_text(encoding='utf-8')).resolve()
    if not failed.is_relative_to(root.resolve()) or failed.name != EMPTY_FAILURE_RUN:
        raise ValueError('恢复只允许固定的初始化失败归档')
    verify_manifest(failed, read(failed / 'artifact_hashes.json'))
    meta = read(failed / 'run_meta.json')
    files = {p.name for p in failed.iterdir()}
    if (digest(failed / 'run_meta.json') != EMPTY_FAILURE_SHA or meta.get('status') != 'failed'
            or meta.get('stage') != 'validation' or meta.get('protocol') != PROTOCOL
            or any(meta.get(k) != 0 for k in ('generated', 'reused', 'scored'))
            or meta.get('judge') is not None or read(failed / 'scored_answers.json') != []
            or files != {'artifact_hashes.json', 'run_meta.json', 'scored_answers.json'}):
        raise ValueError('已经产生回答、请求或来源变化的批次不能用零请求恢复')
    return root / 'approved_empty_restart'


def validate_capability(folder, vectors):
    """用逐题预测重新判定能力，不信任单独的通过标记。"""
    folder = Path(folder).resolve()
    verify_manifest(folder, read(folder / 'artifact_hashes.json'))
    meta = read(folder / 'run_meta.json')
    if (meta.get('status') != 'completed' or meta.get('protocol') != CAP_PROTOCOL
            or meta.get('conditions_completed') != 9 or meta.get('question_conditions_completed') != 12600
            or meta.get('all_parameters_frozen') is not True):
        raise ValueError('固定能力批次未完整结束')
    indices = list((folder / 'source_snapshot').rglob('sample_indices.json'))
    sample = read(folder / 'sample.json')
    if (len(indices) != 1 or digest(indices[0]) != INDICES_SHA or len(sample) != 1400
            or [{'source_index': r['source_index'], 'subject': r['subject']} for r in sample] != read(indices[0])):
        raise ValueError('能力题目/顺序与原canonical不一致')
    saved_vectors = torch.load(folder / 'vectors.pt', map_location='cpu', weights_only=True)
    expected_ids = {v['id'] for v in vectors}
    if set(saved_vectors) != expected_ids or set(read(folder / 'summary.json')) != expected_ids:
        raise ValueError('能力归档缺少固定向量')
    baseline, result = read(folder / 'baseline.json'), {}
    identities = [(r['source_index'], r['subject'], r['answer']) for r in sample]
    def identity(predictions):
        if [(r['source_index'], r['subject'], r['answer']) for r in predictions['samples']] != identities:
            raise ValueError('能力预测与题目身份/答案不符')
    identity(baseline)
    for item in vectors:
        key = item['id']
        if not torch.equal(saved_vectors[key], item['vector']):
            raise ValueError('能力检查向量与原状态不一致')
        injected = read(folder / (key + '.json'))
        identity(injected)
        report = capability_summary(baseline, injected)
        report.update(vector_sha256=item['vector_sha256'],
                      status='capability_pass' if report['within_2pp'] else 'capability_failure',
                      formal_e1_pass=False, behavior_validated=False)
        if report != read(folder / 'summary.json')[key] or report != read(folder / (key + '_summary.json')):
            raise ValueError('能力汇总与逐题预测不一致')
        result[key] = report
    return result


def validate(args):
    state, reference, audit = [Path(p).resolve() for p in (args.state_run, args.reference_run, args.audit_run)]
    for folder in (state, reference, audit):
        verify_manifest(folder, read(folder / 'artifact_hashes.json'))
        if read(folder / 'run_meta.json').get('status') != 'completed':
            raise ValueError('上游归档未完成')
    integrity = read(audit / 'integrity.json')
    audit_hashes = read(audit / 'run_meta.json').get('data_hashes', {})
    reference_manifests = [h for name, h in audit_hashes.items()
                           if name.replace('\\', '/').endswith('/' + RUN_ID + '/artifact_hashes.json')]
    if (state.name != STATE_RUN or digest(state / 'summary.json') != STATE_SHA
            or reference.name != RUN_ID or digest(reference / 'summary.json') != SUMMARY_SHA
            or integrity.get('status') != 'passed' or integrity.get('run_id') != RUN_ID
            or integrity.get('summary_sha256') != SUMMARY_SHA
            or integrity.get('run_meta_sha256') != digest(reference / 'run_meta.json')
            or reference_manifests != [digest(reference / 'artifact_hashes.json')]
            or integrity.get('token_decode_and_prompt_verified') is not True):
        raise ValueError('固定上游或审计身份不符')
    vectors = fixed_vectors(read(state / 'summary.json'),
                            torch.load(state / 'activations.pt', map_location='cpu', weights_only=True),
                            torch.load(state / 'basis.pt', map_location='cpu', weights_only=True))
    capability = validate_capability(args.capability_run, vectors)
    for name in BIG_FIVE['E']:
        if digest(KIT / 'trait_eval' / (name + '.json')) != TRAIT_HASHES[name]:
            raise ValueError('E评测材料或评分提示改变')
    jobs = build_jobs(read_materials(args.materials), capability)
    reused = reused_answers(jobs, lines(reference / 'answers.jsonl'))
    return vectors, capability, jobs, reused


def verify_answer(row, tok, eos_ids):
    """检查原prompt、token解码及停止配置，并保存输入token。"""
    prompt = chat_prompt(tok, row['q'], system=row['system'])
    ids = row['token_ids']
    if (row['prompt'] != prompt or not isinstance(ids, list) or not 0 < len(ids) <= 768
            or any(type(t) is not int or t < 0 for t in ids) or row['n_generated_tokens'] != len(ids)
            or tok.batch_decode([ids], skip_special_tokens=True)[0] != row['a']):
        raise ValueError('回答原文、prompt或token证据不符')
    if row['finish_reason'] == 'eos':
        if ids[-1] not in eos_ids or any(t in eos_ids for t in ids[:-1]):
            raise ValueError('EOS停止证据异常')
    elif row['finish_reason'] != 'length' or len(ids) != 768 or any(t in eos_ids for t in ids):
        raise ValueError('长度停止证据异常')
    row['input_ids'] = tok(prompt, add_special_tokens=False)['input_ids']


def collect(model, tok, jobs, vectors, reused, *, layer=16, progress=None):
    """每向量/中性/极点五题一次，统一seed0；无额外错配生成。"""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('模型须冻结且eval')
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos_ids:
        raise ValueError('模型EOS配置为空')
    result = [dict(row) for row in reused]
    for row in result:
        verify_answer(row, tok, eos_ids)
    if progress:
        progress(result, 0)
    lookup = {v['id']: v['vector'] for v in vectors}
    new = [j for j in jobs if j['upstream_id'] is None]
    generated = 0
    for offset in range(0, len(new), 5):
        batch = new[offset:offset + 5]
        if (len(batch) != 5 or len({(j['vector_id'], j['memory_index'], j['pole']) for j in batch}) != 1
                or [j['question_index'] for j in batch] != list(range(5))):
            raise ValueError('固定五题生成批次不符')
        prompts = [chat_prompt(tok, j['q'], system=j['system']) for j in batch]
        answers = generate(model, tok, prompts, steer={layer: lookup[batch[0]['vector_id']]}, steer_phase='all',
                           max_new_tokens=768, temperature=1., top_p=1., seed=0, batch_size=8,
                           return_details=True, return_token_ids=True)
        if len(answers) != 5:
            raise ValueError('生成返回预算不符')
        batch_rows = []
        for job, prompt, answer in zip(batch, prompts, answers):
            row = {**job, 'prompt': prompt, 'a': answer['text'],
                   **{k: answer[k] for k in ('token_ids', 'n_generated_tokens', 'finish_reason')}}
            verify_answer(row, tok, eos_ids)
            batch_rows.append(row)
        result.extend(batch_rows)
        generated += 5
        if progress:
            progress(result, generated)
    return result


def score_rows(judge, rows, out, *, progress=None, check=lambda: None):
    """隐藏条件的固定三项请求；全轮新缓存，解析失败也保存原响应。"""
    lookup = {r['id']: r for r in rows}
    scored = 0
    for pole in ('high', 'low'):
        order = score_order(rows, pole)
        write_json(out / f'E_{pole}_score_order.json', order)
        trait_prompt = load_trait(BIG_FIVE['E'][pole == 'low'], 'eval')['eval_prompt']
        for metric in METRICS:
            template = trait_prompt if metric == 'trait' else COHERENCE_PROMPT if metric == 'coherence' else TASK_PROMPT
            for rank, identity in enumerate(order):
                if rank % 50 == 0:
                    check()
                row = lookup[identity]
                prompt = template.format(question=row['q'], answer=row['a'])
                key = judge._key(prompt)
                # 先落盘请求原文，接口故障时也保留已尝试的身份。
                with (out / 'judge_requests.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                    f.write(json.dumps({'id': identity, 'metric': metric, 'pole': pole, 'rank': rank,
                                        'key': key, 'prompt': prompt}, ensure_ascii=False) + '\n')
                value = parse_score(judge._call(prompt))
                row[metric] = value
                with (out / 'score_records.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                    f.write(json.dumps({'id': identity, 'metric': metric, 'score': value, 'pole': pole,
                                        'rank': rank, 'key': key}, ensure_ascii=False) + '\n')
                scored += 1
                if progress:
                    progress(rows, scored)
            write_json(out / 'scored_answers.json', rows)
            print(f'[写入评分] E/{pole}/{metric} {scored}/{len(rows) * 3}', flush=True)
    return scored


def verify_scores(out, rows, judge):
    """逐请求绑定原文、实际返回模型与原始响应；重复输入不重复调用。"""
    requests = lines(out / 'judge_requests.jsonl')
    records = lines(out / 'score_records.jsonl')
    responses_list = lines(out / 'judge_responses.jsonl')
    responses = {r['key']: r for r in responses_list}
    if (len(requests) != len(rows) * 3 or len(records) != len(requests)
            or len(responses) != len(responses_list) or set(responses) != {r['key'] for r in requests}):
        raise ValueError('评分请求/响应缺失或重复')
    indexed = {r['id']: r for r in rows}
    seen = set()
    for request, record in zip(requests, records):
        identity = (request['id'], request['metric'])
        if identity in seen or request['metric'] not in METRICS:
            raise ValueError('评分身份重复或指标改变')
        seen.add(identity)
        row = indexed[request['id']]
        template = (load_trait(BIG_FIVE['E'][row['pole'] == 'low'], 'eval')['eval_prompt'] if request['metric'] == 'trait'
                    else COHERENCE_PROMPT if request['metric'] == 'coherence' else TASK_PROMPT)
        prompt = template.format(question=row['q'], answer=row['a'])
        key = hashlib.sha256(f'{judge["model"]}\n{prompt}'.encode()).hexdigest()
        payload = {'model': judge['model'], 'temperature': 0, 'max_tokens': judge['max_tokens'],
                   'messages': [{'role': 'user', 'content': prompt}], **judge['extra_body']}
        raw = responses[key]
        expected_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (request['prompt'] != prompt or request['key'] != key or record['key'] != key
                or record['id'] != request['id'] or record['metric'] != request['metric']
                or record['pole'] != row['pole'] or request['pole'] != row['pole']
                or raw['requested_model'] != judge['model'] or raw['returned_model'] != judge['model']
                or raw['request_sha256'] != expected_sha or raw['finish_reason'] != 'stop'
                or record['score'] != parse_score(raw['output']) or row[request['metric']] != record['score']):
            raise ValueError('评分原文、模型身份或响应关联不一致')
    return len(responses)


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    vectors, capability, jobs, reused = validate(args)
    if not args.execute:
        print(f'[行为预检] {len(jobs)-90}新回答/90复用/{len(jobs)*3}评分，未加载模型/评委', flush=True)
        return {'n_answers': len(jobs), 'model_loaded': False, 'judge_loaded': False}
    model_hashes = verify_model((ROOT / cfg['model']).resolve(), args.model_fingerprint)
    root = output_root(Path(cfg['out_path']) / 'e1_twin_raw_behavior', getattr(args, 'approved_empty_restart', False))
    root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    with (root / (STATE_RUN + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out.resolve()))
    out.mkdir()
    meta = {'task': 'C12', 'status': 'running', 'stage': 'validation', 'start_time': now(), 'end_time': None,
            'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
            'config_path': str(Path(args.config).resolve()), 'config': _redacted(cfg), 'argv': _redacted(sys.argv[1:]),
            'protocol': PROTOCOL, 'model_hashes': model_hashes, 'seed': 0, 'generated': 0, 'reused': 0, 'scored': 0,
            'formal_e1_pass': False, 'behavior_validated': False, 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'cuda': torch.version.cuda, 'gpu': None, 'judge': None}
    if getattr(args, 'approved_empty_restart', False):
        meta['approved_empty_restart_source'] = EMPTY_FAILURE_RUN
    save = lambda: write_json(out / 'run_meta.json', meta)
    rows = []
    save()
    print(f'[行为目录] {out}', flush=True)
    try:
        judge = SequentialJudge(cache_path=str(out / 'judge_cache.jsonl'), max_workers=1,
                                response_audit_path=str(out / 'judge_responses.jsonl'))
        actual = {'model': judge.model, 'base_url': judge.base_url, 'max_tokens': judge.max_tokens, 'extra_body': judge.extra_body}
        if actual != read(ROOT / PRIOR_JUDGE)['judge'] or judge._cache:
            raise ValueError('评委配置改变或新缓存非空')
        sources = [Path(args.config), Path(args.materials), Path(args.model_fingerprint), ROOT / PRIOR_JUDGE,
                   ROOT / 'docs/RESEARCH_SPEC.md', ROOT / 'docs/E1_TWIN_RAW_BEHAVIOR_2026-10-05.md']
        sources += list((ROOT / 'antler').rglob('*.py')) + list((ROOT / 'experiments').glob('*.py'))
        sources += [KIT / 'trait_eval' / (name + '.json') for name in BIG_FIVE['E']]
        for folder in (args.state_run, args.capability_run, args.reference_run, args.audit_run):
            sources += [p for p in Path(folder).rglob('*') if p.is_file() and 'source_snapshot' not in p.parts]
        meta.update(judge=actual, data_hashes={str(p.resolve()): digest(p) for p in sources})
        seal_sources(out, meta['data_hashes'])
        write_json(out / 'jobs.json', jobs)
        write_json(out / 'capability.json', capability)
        torch.save({v['id']: v['vector'] for v in vectors}, out / 'vectors.pt')
        write_json(out / 'vectors.json', [{k: value for k, value in v.items() if k != 'vector'} for v in vectors])
        def check():
            if any(digest(p) != h for p, h in meta['data_hashes'].items()):
                raise ValueError('固定源码、材料或归档改变')
        check()
        model, tok = load_model(cfg['model'], cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        meta.update(stage='generate', all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    stop_config={'model_eos_token_id': model.generation_config.eos_token_id,
                                 'tokenizer_eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id})
        def generation_progress(prefix, n):
            nonlocal rows
            rows = prefix
            write_json(out / 'answers.json', rows)
            meta.update(generated=n, reused=90)
            save()
            check()
            if n % 50 == 0:
                print(f'[写入生成] {n}/{len(jobs)-90}，复用90条', flush=True)
        rows = collect(model, tok, jobs, vectors, reused, progress=generation_progress)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        meta['stage'] = 'score'
        save()
        def score_progress(prefix, n):
            meta['scored'] = n
            if n % 25 == 0:
                save()
        score_rows(judge, rows, out, progress=score_progress, check=check)
        meta['unique_judge_requests'] = verify_scores(out, rows, actual)
        write_json(out / 'summary.json', summarize(rows, jobs, capability))
        check()
        if any(digest(p) != h for p, h in model_hashes.items()):
            raise ValueError('模型文件改变')
        if len(rows) != len(jobs) or meta['scored'] != len(jobs) * 3:
            raise ValueError('终态回答或评分预算不符')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__, error='固定本批停止，保留前缀，不自动重跑。')
        raise
    finally:
        write_json(out / 'scored_answers.json', rows)
        meta['end_time'] = now()
        save()
        write_json(out / 'artifact_hashes.json', {p.relative_to(out).as_posix(): digest(p)
                   for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
