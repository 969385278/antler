"""C9：固定200条配对回答的导出、助手评分封存与揭盲比较。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import uuid

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.eval import matched_review
from antler.eval.blind import write_json
from antler.eval.human import sha256
from e0_extract_coherence import _redacted
from e0_refilter import _git


PUBLIC_FILES = ('public.json', 'assistant_rubric.md', 'export_manifest.json')
FAILURE_MESSAGE = '配对盲审操作失败，已保留记录；请核对输入与当前步骤，不自动重试。'


def add_arguments(ap):
    ap.add_argument('--action', choices=['export', 'seal', 'compare'], required=True)
    ap.add_argument('--source-dir', help='export：登记的完整1800条来源目录')
    ap.add_argument('--rubric', help='export：作者已批准、保持原字节的助手量尺')
    ap.add_argument('--bundle', help='seal或compare：export生成的bundle目录')
    ap.add_argument('--ratings', help='seal：完整200条助手评分JSON')
    ap.add_argument('--sealed-dir', help='compare：seal生成的sealed目录')


def _arguments(args):
    """先检查操作边界，拒绝无效参数，避免误读来源或提前接触私有映射。"""
    required = {'export': {'source_dir', 'rubric'},
                'seal': {'bundle', 'ratings'},
                'compare': {'bundle', 'sealed_dir'}}
    fields = {'source_dir', 'rubric', 'bundle', 'ratings', 'sealed_dir'}
    action = getattr(args, 'action', None)
    if action not in required:
        raise ValueError('操作必须为export、seal或compare')
    provided = {key for key in fields if getattr(args, key, None) is not None}
    if provided != required[action] or any(not str(getattr(args, key)).strip() for key in provided):
        raise ValueError('操作所需参数缺失，或同时提供了仅对其他操作有效的参数')
    return {key: Path(getattr(args, key)).resolve() for key in sorted(provided)}


def _hashes(paths):
    return {str(path.resolve()): sha256(path) for path in paths}


def _unchanged(hashes):
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('配对盲审期间输入发生变化')


def _merge_hashes(target, additions):
    """库返回的已验证输入须与入口快照一致，不能覆盖先前记录。"""
    if not isinstance(additions, dict):
        raise ValueError('库未返回有效的输入哈希')
    for path, digest in additions.items():
        if (not isinstance(path, str) or not path or not isinstance(digest, str)
                or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest)):
            raise ValueError('库返回的输入哈希格式无效')
        name = str(Path(path).resolve())
        if name in target and target[name] != digest:
            raise ValueError('入口与库记录的输入哈希不一致')
        target[name] = digest


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    stamp = now()
    out = Path(cfg['out_path']) / 'matched_review' / (
        stamp.replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    dirty = _git('status', '--porcelain')
    meta = {'task': 'C9', 'experiment': 'e0_matched_review', 'action': args.action,
            'rating_source': 'assistant', 'status': 'running', 'stage': 'validation',
            'start_time': stamp, 'end_time': None, 'git_commit': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(dirty) if dirty is not None else None,
            'config_path': str(Path(args.config).resolve()), 'config_sha256': None,
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'argv': _redacted(sys.argv[1:]), 'seed': cfg.get('seed', 0), 'python': sys.version,
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': None, 'judge': None,
            'model_loaded': False, 'judge_called': False, 'generation_performed': False,
            'input_hashes': {}, 'code_hashes': {}, 'protocol_hashes': {}, 'data_hashes': {},
            'source_provenance': {}, 'formal_e0_pass': False}
    write_json(out / 'run_meta.json', meta)
    try:
        inputs = _arguments(args)
        meta['source_provenance'] = {key: str(path) for key, path in inputs.items()}
        meta['config_sha256'] = sha256(args.config)
        meta['input_hashes'][meta['config_path']] = meta['config_sha256']
        # 记录库及实际复用的助手代码；只读取源码，不加载权重或初始化评委。
        scripts = Path(__file__).resolve().parent
        code_files = sorted((ROOT / 'antler').rglob('*.py')) + [
            Path(__file__).resolve(), scripts / '_common.py',
            scripts / 'e0_extract_coherence.py', scripts / 'e0_refilter.py']
        meta['code_hashes'] = _hashes(code_files)
        meta['protocol_hashes'] = _hashes([
            ROOT / 'docs' / 'RESEARCH_SPEC.md',
            ROOT / 'docs' / 'E0_MATCHED_REVIEW_2026-10-04.md'])
        meta['input_hashes'].update(meta['code_hashes'])
        meta['input_hashes'].update(meta['protocol_hashes'])
        if args.action == 'export':
            public_inputs = [inputs['rubric']]
        else:
            # seal不读私有映射；compare的封存核验和揭盲顺序完全由库执行。
            public_inputs = [inputs['bundle'] / name for name in PUBLIC_FILES]
            if args.action == 'seal':
                public_inputs.append(inputs['ratings'])
        meta['data_hashes'] = _hashes(public_inputs)
        meta['input_hashes'].update(meta['data_hashes'])
        meta['stage'] = args.action
        write_json(out / 'run_meta.json', meta)
        if args.action == 'export':
            result = matched_review.export(inputs['source_dir'], inputs['rubric'], out / 'bundle')
        elif args.action == 'seal':
            result = matched_review.seal(inputs['bundle'], inputs['ratings'], out / 'sealed')
        else:
            result = matched_review.compare(inputs['bundle'], inputs['sealed_dir'])
        if not isinstance(result, dict) or 'input_hashes' not in result:
            raise ValueError('库未返回完整结果及输入哈希')
        _merge_hashes(meta['input_hashes'], result['input_hashes'])
        _merge_hashes(meta['data_hashes'], result['input_hashes'])
        _unchanged(meta['input_hashes'])
        meta['source_provenance']['validated_by'] = 'antler.eval.matched_review.' + args.action
        if 'source_provenance' in result:
            meta['source_provenance']['library'] = _redacted(result['source_provenance'])
        write_json(out / 'result.json', result)
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        # 外部异常可能含凭据；元数据和终端均只给通用信息，保留异常类型供定位。
        meta.update(status='failed', error_type=type(exc).__name__, error=FAILURE_MESSAGE)
        raise RuntimeError(FAILURE_MESSAGE) from None
    finally:
        meta['end_time'] = now()
        write_json(out / 'run_meta.json', meta)
        # 只遍历本次新建的产物目录，不读取输入bundle中的私有文件。
        manifest = {path.relative_to(out).as_posix(): sha256(path)
                    for path in sorted(out.rglob('*'))
                    if path.is_file() and path != out / 'artifact_hashes.json'}
        write_json(out / 'artifact_hashes.json', manifest)
        print(f'[配对盲审结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
