"""C10：首轮机械失败槽位的有界补写，串行限速，遇接口失败即停。"""
import json
from pathlib import Path
import random
import time
import uuid

from _common import parse_args, setup
from antler.memory.d1_plan import make_plan
from antler.memory.episode_data import normalize
from d1_annotate import read_connection
from d1_generate import generate_one
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


SUFFIX = (' Keep the story to 50–65 English words, preferably four short sentences. '
          'Count words internally before returning JSON. Do not add an introduction or explanation.')


def extra(ap):
    ap.add_argument('--source', required=True, help='已完成首轮生成目录')
    ap.add_argument('--env-file', default='.env')


def load_source(source):
    manifest = json.loads((source / 'artifact_hashes.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != digest:
            raise ValueError('首轮产物哈希不符或路径越界')
    meta = json.loads((source / 'run_meta.json').read_text(encoding='utf-8'))
    plan = json.loads((source / 'split_plan.json').read_text(encoding='utf-8'))
    if meta['status'] != 'completed' or meta['completed_requests'] != 280 or plan != make_plan():
        raise ValueError('只接受已完成的冻结首轮')
    rows = [json.loads(s) for s in (source / 'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    rejected = json.loads((source / 'rejected.json').read_text(encoding='utf-8'))
    good = {r['id'] for r in rows}; bad = {r['id'] for r in rejected}
    if good & bad or good | bad != {s['id'] for s in plan['slots']}:
        raise ValueError('首轮槽位划分不完整')
    slots = [s for s in plan['slots'] if s['id'] in bad]
    random.Random(1).shuffle(slots)
    return plan, rows, slots


def collect(slots, conf, out, seen, *, generate=generate_one, sleep=time.sleep, on_result=None):
    results = []
    for index, slot in enumerate(slots):
        if index:
            sleep(5)
        result = generate(slot, conf, out, prompt_suffix=SUFFIX, attempt=2)
        if result['valid']:
            normalized = normalize(result['candidate']['text'])
            if normalized in seen:
                result = {'id': slot['id'], 'valid': False, 'reason': '规范化文本重复'}
            else:
                seen.add(normalized)
        write_json(out / slot['id'] / 'result.json', result)
        results.append(result)
        if on_result:
            on_result(results)
        # 内容机械失败可以继续下一槽位；网络或HTTP失败不继续消耗请求。
        reason = result.get('reason', '')
        if reason.startswith('HTTP ') or reason in ('ReadTimeout', 'ConnectTimeout', 'ConnectionError', 'Timeout', 'SSLError', 'ProxyError'):
            break
    return results


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    source = Path(args.source).resolve()
    plan, old, slots = load_source(source)
    root = Path(cfg['out_path']) / 'd1_recovery'
    # 相同首轮仅允许启动一次补写；失败也保留标记，不能无意重跑。
    root.mkdir(parents=True, exist_ok=True)
    claim = root / (sha256(source / 'artifact_hashes.json') + '.claim')
    with claim.open('x', encoding='utf-8') as f:
        f.write(now())
    out = root / uuid.uuid4().hex; out.mkdir()
    meta = dict(task='C10', status='running', start_time=now(), end_time=None,
                source=str(source), git_commit=_git('rev-parse', 'HEAD'), completed_requests=0,
                max_requests=len(slots), workers=1, seconds_between_requests=5,
                formal_ready=False, attempt=2)
    write_json(out / 'run_meta.json', meta)
    print(f'[D1补写] {out}，最多{len(slots)}次', flush=True)
    try:
        conf = read_connection('minimax', args.env_file)
        if conf['MODEL'] != 'MiniMax-M3':
            raise ValueError('补写模型固定MiniMax-M3')
        sources = [Path(__file__), Path(args.config), source / 'artifact_hashes.json',
                   Path('experiments/d1_generate.py'), Path('experiments/d1_annotate.py'),
                   Path('antler/memory/d1_plan.py'), Path('antler/memory/episode_data.py'),
                   Path('docs/D1_RECOVERY_2026-10-04.md')]
        meta['data_hashes'] = {str(p.resolve()): sha256(p) for p in sources}
        meta.update(model=conf['MODEL'], endpoint=conf['endpoint'])
        write_json(out / 'split_plan.json', plan)
        write_json(out / 'request_order.json', [s['id'] for s in slots])
        def save(results):
            meta['completed_requests'] = len(results)
            meta['new_valid'] = sum(r['valid'] for r in results)
            write_json(out / 'run_meta.json', meta)
            if len(results) % 10 == 0:
                print(f'[D1补写] {len(results)}/{len(slots)}，新增合格{meta["new_valid"]}', flush=True)
        results = collect(slots, conf, out, {normalize(r['text']) for r in old}, on_result=save)
        if any(sha256(p) != h for p, h in meta['data_hashes'].items()):
            raise ValueError('运行期间来源改变')
        new = [r['candidate'] for r in results if r['valid']]
        for name, rows in [('new_candidates.jsonl', new), ('candidates.jsonl', old + new)]:
            (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8', newline='\n')
        write_json(out / 'rejected.json', [r for r in results if not r['valid']])
        write_json(out / 'unattempted.json', [s['id'] for s in slots[len(results):]])
        stopped = bool(results and (len(results) < len(slots) or results[-1].get('reason', '').startswith('HTTP ')))
        meta.update(status='stopped' if stopped else 'completed', valid_candidates=len(old) + len(new))
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now(); write_json(out / 'run_meta.json', meta)
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                                if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
