"""C10：MiniMax/GLM分别标注同一盲标包，逐条归档，无自动重试。"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from urllib.parse import urlparse
import uuid

import requests

from _common import parse_args, setup
from antler.eval.episode_annotation import ANNOTATION_PROMPT, parse_annotation, request_text
from antler.traits.data import load_jsonl
from e0_diagnostics import now, sha256, write_json
from e0_refilter import _git


def extra(ap):
    ap.add_argument('--packet', required=True, help='build_episodes导出的blind.jsonl')
    ap.add_argument('--provider', required=True, choices=['minimax', 'glm'])
    ap.add_argument('--env-file', default='.env')


def read_connection(provider, env_file):
    values = {}
    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            words = shlex.split(value, comments=True)
            values[key.strip()] = words[0] if words else ''
    prefix = provider.upper()
    conf = {name: os.environ.get(f'{prefix}_{name}', values.get(f'{prefix}_{name}', ''))
            for name in ('API_KEY', 'BASE_URL', 'MODEL')}
    if not all(conf.values()):
        raise ValueError('独立标注配置未齐：需要API_KEY、BASE_URL、MODEL')
    url = urlparse(conf['BASE_URL'])
    if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError('接口必须是没有内嵌凭据或查询参数的HTTPS地址')
    base = conf['BASE_URL'].rstrip('/')
    conf['endpoint'] = base if base.endswith('/chat/completions') else base + '/chat/completions'
    return conf


def run(args, post=None):
    cfg, _, _, _ = setup(args, need_model=False, need_judge=False)
    out = Path(cfg['out_path']) / 'd1_annotations' / (args.provider + '_' + uuid.uuid4().hex)
    out.mkdir(parents=True, exist_ok=False)
    meta = dict(task='C10', status='running', provider=args.provider, start_time=now(), end_time=None,
                git_commit=_git('rev-parse', 'HEAD'), completed=0, judge_called=False, formal_ready=False)
    def save(): write_json(out / 'run_meta.json', meta)
    save()
    try:
        conf = read_connection(args.provider, args.env_file)
        packet_hash = sha256(args.packet)
        rows = load_jsonl(args.packet)
        if not rows or len(rows) > 280 or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('盲标包须有1–280个唯一id')
        prompts = [request_text(r) for r in rows]
        meta.update(model=conf['MODEL'], input_sha256=packet_hash,
                    annotation_prompt_sha256=hashlib.sha256(ANNOTATION_PROMPT.encode()).hexdigest(),
                    requested_count=len(rows), endpoint=conf['endpoint'])
        save()
        records = []
        for index, (row, prompt) in enumerate(zip(rows, prompts)):
            if sha256(args.packet) != packet_hash:
                raise ValueError('运行期间盲标输入改变')
            payload = {'model': conf['MODEL'], 'temperature': 0, 'max_tokens': 1024,
                       'thinking': {'type': 'disabled'},
                       'messages': [{'role': 'user', 'content': prompt}]}
            request_file = out / f'{index:04d}_request.json'
            write_json(request_file, payload)
            meta.update(current_id=row['id'], stage='request'); save()
            # 不自动重试未知结果的请求，避免重复调用并选择有利标签。
            try:
                response = (post or requests.post)(conf['endpoint'], json=payload,
                    headers={'Authorization': 'Bearer ' + conf['API_KEY']}, timeout=90, allow_redirects=False)
                if response.status_code != 200:
                    raise RuntimeError(f'HTTP状态{response.status_code}')
                body = response.json()
            except Exception as exc:
                raise RuntimeError('标注调用失败，未自动重试；类型：' + type(exc).__name__) from None
            write_json(out / f'{index:04d}_response.json', body)
            if body.get('model') != conf['MODEL']:
                raise ValueError('响应模型身份与配置不同，须核对别名后处理')
            if len(body.get('choices', [])) != 1:
                raise ValueError('标注响应必须恰有一个候选')
            choice = body['choices'][0]
            if choice.get('finish_reason') != 'stop':
                raise ValueError('标注未正常结束，不解析截断标签')
            result = parse_annotation(choice['message'].get('content'))
            records.append({'id': row['id'], **result, 'request_sha256': sha256(request_file),
                            'response_sha256': sha256(out / f'{index:04d}_response.json')})
            write_json(out / 'labels.json', dict(model=conf['MODEL'], vendor=args.provider,
                                               input_sha256=packet_hash, records=records))
            meta['completed'] += 1; save()
        if sha256(args.packet) != packet_hash:
            raise ValueError('运行结束时盲标输入改变')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        meta['end_time'] = now(); save()
        write_json(out / 'artifact_hashes.json', {p.name: sha256(p) for p in out.iterdir() if p.is_file() and p.name != 'artifact_hashes.json'})
    print(f'[D1标注] {args.provider}完成{len(rows)}条：{out}')
    return out


if __name__ == '__main__':
    run(parse_args(extra))
