"""C10：封存结构复核后，对固定98个缺额各生成一次新版本。"""
import hashlib
import json
from pathlib import Path
import uuid

from _common import parse_args, setup
from antler.memory.episode_replacement import replacement_slots, remaining_slots
from antler.memory.episode_data import normalize
from d1_annotate import read_connection
from d1_generate import generate_one
from d1_recover import collect
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


SUFFIX = (' The narrator is a digital AI with no body, relatives, classmates, personal grades, or offline attendance. '
          'Make the user request explicit. Physical actions belong to the user. '
          'Keep names, family relationships, requested task, and feedback consistent. '
          'If two lasting changes are requested, make BOTH consequences explicit in the same event. '
          'For a neutral story, give an ordinary successful digital exchange with no changed future tendency. '
          'For prioritizing my own interests, use my stated digital project or interaction preferences; '
          'do not substitute the users self-interest for mine. '
          'Feedback need not be positive, but show why the stated outcome leads to the subsequent tendency. '
          'Do not copy prior stories or include scientific claims unnecessary for the interaction.')


def extra(ap):
    ap.add_argument('--source', help='已完成新口径合并目录')
    ap.add_argument('--review', help='追加结构复核文件')
    ap.add_argument('--selection', help='第二轮：已封存233保留/47缺额选择台账')
    ap.add_argument('--selection-round3', help='第三轮：已封存254保留/26缺额台账；各槽位仅一次')
    ap.add_argument('--env-file', default='.env')
    ap.add_argument('--execute', action='store_true', help='实际请求；默认只准备并展示缺额')


def load(source, review_path):
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    hashes = {}
    for name, digest in read(source / 'artifact_hashes.json').items():
        p = (source / name).resolve()
        if not p.is_relative_to(source.resolve()) or sha256(p) != digest:
            raise ValueError('旧合并归档哈希错误或路径越界')
        hashes[str(p)] = digest
    if read(source / 'run_meta.json')['status'] != 'completed':
        raise ValueError('旧合并未完成')
    review = read(review_path)
    for name, digest in review['source_hashes'].items():
        p = Path(name).resolve()
        if sha256(p) != digest:
            raise ValueError('结构复核来源哈希错误')
        hashes[str(p)] = digest
    accepted_path = source / 'label_accepted.jsonl'
    if str(accepted_path.resolve()) not in {str(Path(p).resolve()) for p in review['source_hashes']}:
        raise ValueError('复核未绑定本合并候选')
    rows = lambda p: [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
    accepted, rejected = rows(accepted_path), rows(source / 'label_rejected.jsonl')
    keep, slots = replacement_slots(accepted, rejected, review)
    if (len(accepted), len(rejected), len(keep), len(slots)) != (217, 57, 182, 98):
        raise ValueError('不符合本轮登记的217/57/182/98固定范围')
    hashes[str(review_path.resolve())] = sha256(review_path)
    return accepted + rejected, keep, slots, hashes


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    selection_arg = getattr(args, 'selection', None)
    third = getattr(args, 'selection_round3', None)
    if third and selection_arg:
        raise ValueError('第二轮与第三轮不能同时指定')
    round_id = 3 if third else 2 if selection_arg else 1
    selection_arg = third or selection_arg
    if selection_arg:
        if args.source or args.review:
            raise ValueError('selection与source/review不能同时使用')
        source = review_path = Path(selection_arg).resolve()
        expected_sha = ('3923c0aceee07678efd658c74f702cd038637c3ae8bb774d4e352da197ef5e1d'
                        if round_id == 3 else '7b12e43c5b2ffa1ec4c734938d05565d0527dd6661db44d1dd3b461644a6ad89')
        if sha256(source) != expected_sha:
            raise ValueError('本轮台账不是登记版本')
        selection = json.loads(source.read_text(encoding='utf-8'))
        hashes = dict(selection['source_hashes'])
        if any(sha256(p) != h for p, h in hashes.items()):
            raise ValueError('第二轮台账来源哈希错误')
        hashes[str(source)] = sha256(source)
        keep, slots = remaining_slots(selection, round_id=round_id)
        old = list(keep)
        # 所有旧机械合格原文用于重复检查，不传入生成提示。
        for name in hashes:
            p = Path(name)
            if p.name in ('new_candidates.jsonl', 'label_accepted.jsonl', 'label_rejected.jsonl'):
                old.extend(json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip())
    else:
        if not args.source or not args.review:
            raise ValueError('原入口需要source及review')
        source, review_path = Path(args.source).resolve(), Path(args.review).resolve()
        old, keep, slots, hashes = load(source, review_path)
    print(f'[版本补写] 保留{len(keep)}，缺额{len(slots)}，实际请求={args.execute}', flush=True)
    if not args.execute:
        return None
    conf = read_connection('minimax', args.env_file)
    if conf['MODEL'] != 'MiniMax-M3':
        raise ValueError('生成器固定MiniMax-M3')
    root = Path(cfg['out_path']) / 'd1_replacement'; root.mkdir(parents=True, exist_ok=True)
    out = root / uuid.uuid4().hex
    # 同一封存复核仅允许启动一次，失败和未知请求也不能自动重发。
    with (root / (sha256(review_path) + '.claim')).open('x', encoding='utf-8') as f:
        f.write(str(out))
    out.mkdir()
    paths = [Path(__file__), Path(args.config), Path('antler/memory/episode_replacement.py'),
             Path('antler/memory/d1_plan.py'), Path('antler/memory/episode_data.py'),
             Path('experiments/d1_generate.py'), Path('experiments/d1_recover.py'),
             Path('experiments/d1_annotate.py'), Path('docs/RESEARCH_SPEC.md'),
             Path('docs/D1_REPLACEMENT_2026-10-04.md'), Path('docs/D1_REMAINING_PROPOSAL_2026-10-04.md')]
    if round_id == 3:
        paths.append(Path('docs/D1_ROUND3_2026-10-04.md'))
    hashes.update({str(p.resolve()): sha256(p) for p in paths})
    meta = dict(task='C10', status='running', start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), source=str(source), review=str(review_path),
                data_hashes=hashes, max_requests=len(slots), completed_requests=0, workers=1,
                seconds_between_requests=5, replacement_round=round_id, retries=0, formal_ready=False,
                model=conf['MODEL'], endpoint=conf['endpoint'])
    save = lambda: write_json(out / 'run_meta.json', meta)
    save(); write_json(out / 'request_order.json', [s['id'] for s in slots])
    write_json(out / 'retained_sources.json', [{'id': r['id'], 'text_sha256': hashlib.sha256(r['text'].encode('utf-8')).hexdigest()} for r in keep])
    print(f'[版本补写目录] {out}', flush=True)
    results = []
    def on_result(current):
        results[:] = current
        meta.update(completed_requests=len(current), new_valid=sum(r['valid'] for r in current))
        save()
        if len(current) % 10 == 0:
            print(f'[版本补写] {len(current)}/{len(slots)}，机械合格{meta["new_valid"]}', flush=True)
    def generate(slot, conf, out, **kwargs):
        return generate_one(slot, conf, out, prompt_suffix=kwargs['prompt_suffix'] + SUFFIX, attempt=round_id + 2)
    try:
        collect(slots, conf, out, {normalize(r['text']) for r in old}, generate=generate,
                digital_role=True, on_result=on_result)
        if any(sha256(p) != h for p, h in hashes.items()):
            raise ValueError('运行期间来源改变')
        fatal = results[-1].get('reason', '') if results else ''
        stopped = len(results) < len(slots) or fatal.startswith('HTTP ') or fatal in (
            'ReadTimeout', 'ConnectTimeout', 'ConnectionError', 'Timeout', 'SSLError', 'ProxyError')
        meta['status'] = 'stopped' if stopped else 'completed'
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        (out / 'new_candidates.jsonl').write_text(''.join(json.dumps(r['candidate'], ensure_ascii=False)+'\n' for r in results if r['valid']), encoding='utf-8', newline='\n')
        write_json(out / 'rejected.json', [r for r in results if not r['valid']])
        # failed时此后缀可能包含未知在途请求，不能据此自动续跑。
        write_json(out / 'not_completed.json', [s['id'] for s in slots[len(results):]])
        meta['end_time'] = now(); save()
        write_json(out / 'artifact_hashes.json', {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})
    return out


if __name__ == '__main__':
    run(parse_args(extra))
