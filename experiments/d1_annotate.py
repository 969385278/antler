"""C10：MiniMax/GLM分别标注同一盲标包，逐条归档，无自动重试。"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
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
    ap.add_argument('--resume', help='已失败且完整归档请求/响应的标注目录；保留原目录')
    ap.add_argument('--allow-explicit-plus', action='store_true', help='仅允许把JSON对象数值+1规范化为1')


def archived_prefix(source, rows, conf, provider, packet_hash, allow_plus):
    """只有结果已知且可核验的连续前缀允许复用，未知请求不会重新发送。"""
    read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
    manifest = read(source / 'artifact_hashes.json')
    for name, digest in manifest.items():
        p = (source / name).resolve()
        if not p.is_relative_to(source.resolve()) or sha256(p) != digest:
            raise ValueError('原标注产物哈希错误或路径越界')
    meta = read(source / 'run_meta.json')
    if (meta['status'] != 'failed' or meta['provider'] != provider or meta['model'] != conf['MODEL']
            or meta['input_sha256'] != packet_hash or meta['endpoint'] != conf['endpoint']
            or meta['annotation_prompt_sha256'] != hashlib.sha256(ANNOTATION_PROMPT.encode()).hexdigest()):
        raise ValueError('续跑来源状态、模型、提示或输入不符')
    requests_found = sorted(source.glob('*_request.json'))
    responses_found = sorted(source.glob('*_response.json'))
    count = len(requests_found)
    if not 0 < count <= len(rows) or len(responses_found) != count:
        raise ValueError('存在结果未知的请求，禁止自动重发')
    if (requests_found != [source / f'{i:04d}_request.json' for i in range(count)]
            or responses_found != [source / f'{i:04d}_response.json' for i in range(count)]):
        raise ValueError('归档前缀不连续')
    old = read(source / 'labels.json')['records'] if (source / 'labels.json').exists() else []
    if len(old) != meta['completed'] or len(old) not in (count, count - 1):
        raise ValueError('既有标签数量与失败位置不符')
    for i in range(count):
        request = requests_found[i]; response = responses_found[i]
        if request.name not in manifest or response.name not in manifest:
            raise ValueError('前缀请求响应未封存')
        expected = payload_for(conf['MODEL'], request_text(rows[i]))
        if read(request) != expected:
            raise ValueError('归档请求不是当前盲包与固定参数')
        result, _ = parse_body(read(response), conf['MODEL'], allow_plus)
        if i < len(old):
            record = old[i]
            if (record['id'] != rows[i]['id'] or record['labels'] != result['labels'] or record['reason'] != result['reason']
                    or record['request_sha256'] != sha256(request) or record['response_sha256'] != sha256(response)):
                raise ValueError('续跑会改变既有标签或来源')
    return count


def payload_for(model, prompt):
    return {'model': model, 'temperature': 0, 'max_tokens': 1024,
            'thinking': {'type': 'disabled'}, 'messages': [{'role': 'user', 'content': prompt}]}


def parse_body(body, model, allow_plus):
    if body.get('model') != model or len(body.get('choices', [])) != 1:
        raise ValueError('响应模型或候选数量不符')
    choice = body['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('标注未正常结束，不解析截断标签')
    content = choice['message'].get('content')
    try:
        return parse_annotation(content), False
    except ValueError:
        if not allow_plus:
            raise
        return parse_annotation(content, allow_explicit_plus=True), True


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
        source_arg = getattr(args, 'resume', None)
        source = Path(source_arg).resolve() if source_arg else None
        allow_plus = getattr(args, 'allow_explicit_plus', False)
        reused = archived_prefix(source, rows, conf, args.provider, packet_hash, allow_plus) if source else 0
        if source:
            claim = out.parent / (sha256(source / 'artifact_hashes.json') + '.resume.claim')
            with claim.open('x', encoding='utf-8') as f:
                f.write(str(out.resolve()))
        meta.update(resume_source=str(source) if source else None, reused_responses=reused,
                    allow_explicit_plus=allow_plus, new_requests=0,
                    resume_manifest_sha256=sha256(source / 'artifact_hashes.json') if source else None)
        save()
        print(f'[D1标注] {out}，复用{reused}条，待请求{len(rows)-reused}条', flush=True)
        for index, (row, prompt) in enumerate(zip(rows, prompts)):
            if sha256(args.packet) != packet_hash:
                raise ValueError('运行期间盲标输入改变')
            payload = payload_for(conf['MODEL'], prompt)
            request_file = out / f'{index:04d}_request.json'
            response_file = out / f'{index:04d}_response.json'
            write_json(request_file, payload)
            meta.update(current_id=row['id'], stage='request'); save()
            # 不自动重试未知结果的请求，避免重复调用并选择有利标签。
            if index < reused:
                shutil.copyfile(source / request_file.name, request_file)
                shutil.copyfile(source / response_file.name, response_file)
                body = json.loads(response_file.read_text(encoding='utf-8'))
            else:
                try:
                    meta['new_requests'] += 1; save()
                    response = (post or requests.post)(conf['endpoint'], json=payload,
                        headers={'Authorization': 'Bearer ' + conf['API_KEY']}, timeout=90, allow_redirects=False)
                    if response.status_code != 200:
                        raise RuntimeError(f'HTTP状态{response.status_code}')
                    body = response.json()
                except Exception as exc:
                    raise RuntimeError('标注调用失败，未自动重试；类型：' + type(exc).__name__) from None
                write_json(response_file, body)
            result, normalized = parse_body(body, conf['MODEL'], allow_plus)
            records.append({'id': row['id'], **result, 'request_sha256': sha256(request_file),
                            'response_sha256': sha256(response_file), 'explicit_plus_normalized': normalized})
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
