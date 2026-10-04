"""C10：按冻结槽位生成首轮280段经历，保留失败，不自动补写。"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import uuid

import requests

from _common import parse_args, setup
from antler.memory.d1_plan import make_plan, generation_prompt
from antler.memory.episode_data import normalize, validate_candidates
from d1_annotate import read_connection
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


def extra(ap):
    ap.add_argument('--env-file', default='.env')


def generate_one(slot, conf, out, post=None):
    path = out / slot['id']; path.mkdir()
    payload = {'model': conf['MODEL'], 'temperature': .8, 'max_tokens': 512,
               'thinking': {'type': 'disabled'}, 'messages': [{'role': 'user', 'content': generation_prompt(slot)}]}
    write_json(path / 'request.json', payload)
    try:
        response = (post or requests.post)(conf['endpoint'], json=payload,
            headers={'Authorization': 'Bearer ' + conf['API_KEY']}, timeout=90, allow_redirects=False)
        if response.status_code != 200:
            return {'id': slot['id'], 'valid': False, 'reason': f'HTTP {response.status_code}'}
        body = response.json(); write_json(path / 'response.json', body)
        if body.get('model') != conf['MODEL'] or len(body.get('choices', [])) != 1:
            raise ValueError('模型身份或响应候选数量不符')
        choice = body['choices'][0]
        if choice.get('finish_reason') != 'stop':
            raise ValueError('生成回答被截断或未正常结束')
        content = choice['message']['content'].strip()
        if content.startswith('```json') and content.endswith('```'):
            content = content[7:-3].strip()
        generated = json.loads(content)
        if not isinstance(generated, dict) or set(generated) != {'text'}:
            raise ValueError('需要只有text字段的JSON对象')
        row = {**slot, 'text': generated['text'], 'generator': conf['MODEL'] + '@' + now()[:10],
               'generator_vendor': 'minimax', 'dataset': 'D1', 'attempt': 1}
        validate_candidates([row], allow_grouped=True)
        return {'id': slot['id'], 'valid': True, 'candidate': row}
    except Exception as exc:
        # 不保存可能含鉴权信息的网络异常正文；已收到的模型响应单独留存。
        reason = str(exc).replace(conf['API_KEY'], '[redacted]')[:300] if isinstance(exc, ValueError) else type(exc).__name__
        return {'id': slot['id'], 'valid': False, 'reason': reason}


def assemble(results, plan):
    if set(results) != {s['id'] for s in plan['slots']}:
        raise ValueError('首轮槽位结果不完整')
    valid, invalid, seen = [], [], set()
    for slot in plan['slots']:
        result = results[slot['id']]
        if result['valid']:
            row = result['candidate']
            text = normalize(row['text'])
            if text not in seen:
                valid.append(row); seen.add(text); continue
            result = {'id': slot['id'], 'valid': False, 'reason': '规范化文本重复'}
        invalid.append(result)
    return valid, invalid


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'd1_generation' / uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(task='C10', status='running', start_time=now(), end_time=None, completed_requests=0,
                git_commit=_git('rev-parse', 'HEAD'), formal_ready=False, judge_called=False,
                max_requests=280, workers=4, retries=0)
    write_json(out / 'run_meta.json', meta)
    try:
        conf = read_connection('minimax', args.env_file)
        if conf['MODEL'] != 'MiniMax-M3':
            raise ValueError('首轮生成器固定为MiniMax-M3')
        plan = make_plan(); write_json(out / 'split_plan.json', plan)
        sources = [Path(__file__), Path(args.config), Path('antler/memory/d1_plan.py'),
                   Path('antler/memory/episode_data.py'), Path('experiments/d1_annotate.py'),
                   Path('docs/D1_GENERATION_2026-10-04.md')]
        meta['data_hashes'] = {str(p.resolve()): sha256(p) for p in sources}
        meta['model'] = conf['MODEL']; meta['endpoint'] = conf['endpoint']
        write_json(out / 'run_meta.json', meta)
        results = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(generate_one, slot, conf, out): slot['id'] for slot in plan['slots']}
            for future in as_completed(pending):
                result = future.result()
                results[pending[future]] = result
                write_json(out / pending[future] / 'result.json', result)
                meta['completed_requests'] += 1
                write_json(out / 'run_meta.json', meta)
                if meta['completed_requests'] % 20 == 0:
                    print(f'[D1生成] {meta["completed_requests"]}/280 请求完成', flush=True)
        if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('运行期间来源改变')
        valid, invalid = assemble(results, plan)
        (out / 'candidates.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in valid), encoding='utf-8', newline='\n')
        write_json(out / 'rejected.json', invalid)
        meta.update(status='completed', valid_candidates=len(valid), rejected_candidates=len(invalid))
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now(); write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
        print(f'[结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(extra))
