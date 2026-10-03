"""C9：只读导入作者已填写的task盲表，核验原始回答并报告相关性。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval.human import import_task_labels, sha256


def add_arguments(ap):
    ap.add_argument('--csv', required=True, help='作者填写的task_blind.csv')
    ap.add_argument('--key', required=True, help='对应的task_blind_key.json；请完成盲标后才查看')
    ap.add_argument('--source-root', help='原始steps目录的父目录；默认映射所在目录')


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    stamp = datetime.now(timezone.utc).isoformat()
    out = cfg['out_path'] / 'human_task' / (stamp.replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    meta = {'task': 'C9', 'start_time': stamp, 'end_time': None, 'status': 'running',
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'config_path': str(Path(args.config).resolve()), 'config_sha256': sha256(args.config),
            'argv': sys.argv[1:], 'model_loaded': False, 'judge_called': False, 'formal_e0_pass': False}
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    write('run_meta.json', meta)
    try:
        result = import_task_labels(args.csv, args.key, args.source_root)
        meta['input_hashes'] = result['input_hashes']
        write('agreement.json', result)
        rho = '不可计算' if result['spearman'] is None else f"{result['spearman']:.4f}"
        text = (f"# 任务理解盲评导入\n\n有效配对：{result['n_paired']}/{result['n_samples']}。\n\n"
                f"作者与评委Spearman：{rho}。任务评委校验通过：{result['task_judge_validated']}。\n\n"
                '缺失分不按零填充；未完成或常量评分不能通过。\n\n'
                '此结果仅覆盖30条任务理解校验，不代表E0通过。D5人格、连贯性及间隔7天复标仍需独立完成。\n')
        (out / 'summary.md').write_text(text, encoding='utf-8', newline='\n')
        meta['status'] = 'completed'
    except BaseException:
        meta['status'] = 'failed'
        raise
    finally:
        meta['end_time'] = datetime.now(timezone.utc).isoformat()
        write('run_meta.json', meta)
    print(f'[结果目录] {out}')
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
