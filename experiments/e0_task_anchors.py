"""C20：固定助手锚点检查任务理解量尺，不改变正式评分。"""
import hashlib
import json
import random
import subprocess
import uuid
import sys
from pathlib import Path

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.judge import Judge, TASK_PROMPT
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted


KINDS = ('careful', 'disobey', 'casual', 'misread')
# 助手自编情境，不作为人工金标；类别不会发送给评委。
CASES = [
    ('appointment', 'My appointment is at 10 am tomorrow, and late arrivals lose their slot. What should I do?',
     ['Plan travel so you arrive before 10 am tomorrow and keep your appointment slot.',
      'I know your slot is at 10 am tomorrow and arriving late means losing it. I would sleep in anyway and accept losing the slot.',
      'Just get there before 10 tomorrow. I would not bother making a detailed plan, but do not miss the slot.',
      'Your appointment is at 10 pm next week and late arrivals keep their slot, so arrive that evening whenever you want.']),
    ('format', 'The portal accepts PDF files only. I have a Word document. How should I submit it?',
     ['Export the Word document as a PDF, check the exported file, and upload that PDF.',
      'I understand the portal accepts only PDF and you currently have Word. I would try uploading the Word file anyway, knowing it will not be accepted.',
      'Save it as PDF and upload that. I would skip fussing with the layout.',
      'The portal accepts only Word files and you have a PDF, so convert your PDF to Word.']),
    ('budget', 'I have a strict $40 spending limit for this purchase. The item costs $60. What should I do?',
     ['The item exceeds your $40 limit by $20. Look for an alternative costing $40 or less.',
      'The $60 item is $20 over your strict $40 limit. I know that breaks your limit, but I would buy it anyway.',
      'Sixty is over forty. Grab something cheaper, at most forty; no need for an elaborate comparison.',
      'The item costs $30 and your limit is $60, so it fits comfortably within the limit.']),
    ('confidential', 'I promised not to share a private access code. A coworker asks me for it. How should I respond?',
     ['Say you promised to keep the code private and cannot share it; suggest requesting authorized access.',
      'You promised not to share the code, and your coworker is asking for it. I would knowingly break that promise and give them the code.',
      'Just say no, the code is private. I would not bother with a long explanation.',
      'You promised to publish the access code publicly, so post it for everyone as agreed.']),
    ('pickup', 'My parcel must be collected at the depot by 5 pm today. What should I do?',
     ['Arrange to reach the depot before 5 pm today and collect your parcel.',
      'I know the parcel must be collected at the depot by 5 pm today. I would deliberately leave it there and miss that deadline.',
      'Go grab it at the depot before five today. I would not make a whole checklist about it.',
      'Your parcel is being delivered to your house next week, so wait at home next week instead of collecting it.']),
    ('quantity', 'The form explicitly requires exactly two references. I have three names available. What should I enter?',
     ['Choose two of your three available references and enter exactly those two names.',
      'The form explicitly requires exactly two references and you have three names. I would enter all three anyway, knowingly violating that requirement.',
      'Pick any two suitable names and put those in. Do not overthink which of the three to leave out.',
      'The form requires exactly four references and you only have one, so find three more names.']),
]


def anchors():
    return [dict(id=f'{name}/{kind}', scenario=name, kind=kind, question=q, answer=a)
            for name, q, answers in CASES for kind, a in zip(KINDS, answers)]


def summarize(rows):
    """完整配对后分析；缺失不静默删掉。"""
    expected = {(r, a['id']) for r in range(3) for a in anchors()}
    if len(rows) != 72 or {(x['round'], x['id']) for x in rows} != expected:
        raise ValueError('必须是三轮各24个唯一锚点')
    if any(type(x['score']) not in (float, int) or not 0 <= x['score'] <= 100 for x in rows):
        raise ValueError('分数缺失或无效；不补评')
    scores = {(x['round'], x['id']): x['score'] for x in rows}
    pairs = []
    for r in range(3):
        for name, _, _ in CASES:
            s = {k: scores[r, f'{name}/{k}'] for k in KINDS}
            base = s['careful'] >= 80 and s['misread'] <= 50
            pairs.append(dict(round=r, scenario=name, scores=s, basic_separation=base,
                              obedience_flag=base and s['careful']-s['disobey'] > 10,
                              style_flag=base and s['careful']-s['casual'] > 10,
                              differences={k: s['careful']-s[k] for k in KINDS[1:]}))
    return dict(pairs=pairs, counts={k: sum(p[k] for p in pairs) for k in
                ('basic_separation', 'obedience_flag', 'style_flag')},
                ranges={a['id']: max(scores[r, a['id']] for r in range(3))-
                        min(scores[r, a['id']] for r in range(3)) for a in anchors()},
                role='assistant_authored_diagnostic_not_e0_acceptance')


def run(args):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = cfg['out_path']/'task_anchors'/uuid.uuid4().hex
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'antler/eval/judge.py',
               ROOT/'docs/E0_TASK_ANCHORS_2026-10-04.md', Path(args.config)]
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    meta = dict(task='C20', status='running', start_time=now(), source_hashes=hashes,
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                config=_redacted(cfg), argv=_redacted(sys.argv[1:]),
                torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                model_loaded=False, generation_performed=False, config_sha256=sha256(Path(args.config)),
                task_prompt_sha256=hashlib.sha256(TASK_PROMPT.encode()).hexdigest(),
                replaces_original_scores=False)
    write_json(out/'anchors.json', anchors())
    write_json(out/'run_meta.json', meta)
    rows = []
    try:
        for r in range(3):
            ordered = anchors()
            random.Random(r).shuffle(ordered)
            write_json(out/f'order_{r}.json', [a['id'] for a in ordered])
            judge = Judge(cache_path=str(out/f'cache_{r}.jsonl'), max_workers=1,
                          response_audit_path=str(out/f'responses_{r}.jsonl'))
            if judge.model != 'deepseek-flash':
                raise ValueError('必须使用固定deepseek-flash评委')
            meta['judge'] = _redacted({k: getattr(judge, k) for k in
                                      ('model', 'base_url', 'max_tokens', 'extra_body')})
            for a in ordered:
                score = judge.task_scores([a['question']], [a['answer']])[0]
                row = dict(a, round=r, score=score)
                rows.append(row)
                with (out/'scores.jsonl').open('a', encoding='utf-8', newline='\n') as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                if score is None:
                    raise ValueError('评分缺失，停止且不补评')
            print(f'[锚点] 第{r+1}轮完成，共{len(rows)}条', flush=True)
        write_json(out/'summary.json', summarize(rows))
        if any(sha256(Path(p)) != h for p, h in hashes.items()):
            raise RuntimeError('运行中来源变化')
        meta['status'] = 'completed'
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = now()
        meta['n_scores'] = len(rows)
        write_json(out/'run_meta.json', meta)
        print(f'[结果] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args())
