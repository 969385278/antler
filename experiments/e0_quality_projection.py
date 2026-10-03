"""C18：固定1080条连贯性相关方向剥离对照，仅calib、同期零注入与原向量。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import uuid
import zipfile

import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.generate import generate
from antler.core.model import chat_prompt, load_model
from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT, Judge
from antler.traits.data import BIG_FIVE, KIT, load_trait
from antler.traits.quality_projection import ARMS, COEFS, LAYER, SEEDS, analyze, project_vector, validate_rows
import e0_decode_strength as strength
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted
from e0_phase_diagnostics import load_vectors, score_rows, validate_generation
from e0_refilter import _git


REGISTERED_VECTORS = strength.REGISTERED_VECTORS
REGISTERED_HASHES = dict(strength.REGISTERED_HASHES)
REGISTERED_Q = Path('outputs/main/quality_direction/2026-10-03T195956.023383_0000_d77af3de')
Q_HASHES = {
    'statistics.pt': '11f002079377c22991d74657f1d213abd81873feada76fdb300213cbdceba087',
    'summary.json': 'a4147e31b5270184ff096557f77a3880f3df43b0821dad2e1b043279e31423bf',
    'run_meta.json': 'af70bd20df5f780fdd6228451e2ffd48645c6b8504a744a3ebc7b23f447d9ff0',
}
REGISTERED_OLD_SAMPLES = Path('outputs/main/decode_strength/2026-10-03T182340.163154_0000_e41713cc/samples.jsonl')
OLD_SAMPLES_SHA256 = '1b6a4a025e41458f59d7c44f12e704c396d55ff85adfdc4cb02025c61838b425'


def add_arguments(ap):
    ap.add_argument('--vectors-dir', default=str(REGISTERED_VECTORS), help='登记的完整paired向量目录')
    ap.add_argument('--q-source', default=str(REGISTERED_Q), help='已核验连贯方向结果目录')
    ap.add_argument('--old-samples', default=str(REGISTERED_OLD_SAMPLES), help='原1800条结果，仅检查身份与文本重复')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def tensor_sha256(value):
    """保存实际注入张量的字节哈希，BF16也无需转精度。"""
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def load_quality_source(source, hidden_size):
    """固定q及全部源产物通过核验后才允许构造投影向量。"""
    source = Path(source).resolve()
    if source != (ROOT / REGISTERED_Q).resolve():
        raise ValueError('连贯方向来源目录不符合固定登记')
    if any(sha256(source / name) != digest for name, digest in Q_HASHES.items()):
        raise ValueError('q或筛查结果哈希不符合登记')
    manifest_path = source / 'artifact_hashes.json'
    manifest = _json(manifest_path)
    required = {'run_meta.json', 'summary.json', 'statistics.pt', 'folds.json',
                'response_mean_activations.pt', 'replayed_inputs.jsonl'}
    if not isinstance(manifest, dict) or not required <= set(manifest):
        raise ValueError('连贯方向来源产物清单不完整')
    hashes = {str(manifest_path): sha256(manifest_path)}
    for name, digest in manifest.items():
        path = (source / name).resolve()
        if (not isinstance(name, str) or Path(name).is_absolute() or not path.is_relative_to(source)
                or not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest):
            raise ValueError('连贯方向产物路径或哈希无效')
        hashes[str(path)] = digest
    meta, summary = _json(source / 'run_meta.json'), _json(source / 'summary.json')
    if (meta.get('status') != 'completed' or meta.get('layer') != LAYER
            or summary.get('direction_supported') is not True
            or summary.get('bootstrap', {}).get('complete') is not True
            or any(summary.get('targets', {}).get(pole, {}).get('candidate') is not True
                   for pole in ('careless', 'nervous'))):
        raise ValueError('固定q来源未完成或两个目标未通过登记筛查')
    artifact = torch.load(source / 'statistics.pt', map_location='cpu', weights_only=True)
    q = artifact.get('q') if isinstance(artifact, dict) else None
    if (not isinstance(q, torch.Tensor) or q.shape != (hidden_size,) or not q.is_floating_point()
            or not torch.isfinite(q).all() or abs(float(q.double().norm()) - 1) > 1e-6):
        raise ValueError('q须为隐藏维度一致的有限单位向量')
    return q, hashes


def verify_quality_lineage(q_source, reused_files):
    """复验实际复用的模型/tokenizer与paired向量，历史源码登记只保留来源。"""
    source_hashes = _json(Path(q_source) / 'run_meta.json').get('data_hashes')
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError('固定q来源缺少上游输入哈希')
    checked = []
    for path in reused_files:
        path = Path(path).resolve()
        relative = path.relative_to(ROOT.resolve()).as_posix()
        matches = [(name, digest) for name, digest in source_hashes.items()
                   if name.replace('\\', '/').endswith('/' + relative)]
        if len(matches) != 1 or sha256(path) != matches[0][1]:
            raise ValueError(f'当前复用输入与q的来源绑定不一致：{relative}')
        checked.append({'relative_path': relative, 'source_path': matches[0][0], 'sha256': matches[0][1]})
    return {'verified_current_inputs': checked,
            'scope': '当前复用的模型/tokenizer和paired四极点；旧源码/SPEC仅保留历史哈希，不要求新协议与旧字节相同。'}


def build_injection_vectors(base_vectors, q):
    """原张量复用旧强度入口，剥离仅改方向，保持实际L2注入范数。"""
    vectors, audit = {}, {}
    for dim in COEFS:
        original = strength.arm_vector(base_vectors, dim, 'decode_mid')[LAYER]
        before = original.clone()
        projected = project_vector(original, q)
        if not torch.equal(original, before):
            raise ValueError('投影修改了原始注入张量')
        vectors[dim] = {'original': original, 'projected': projected['vector']}
        audit[dim] = {**projected['audit'], 'coef': COEFS[dim]['original'],
                      'original_source': "e0_decode_strength.arm_vector(..., 'decode_mid')",
                      'original_sha256': tensor_sha256(original),
                      'projected_sha256': tensor_sha256(projected['vector']),
                      'original_dtype': str(original.dtype), 'projected_dtype': str(projected['vector'].dtype),
                      'same_l2_does_not_mean_same_trait_effect': True}
    return vectors, audit


def arm_phase(arm):
    if arm not in ARMS:
        raise ValueError('实验臂越过固定登记')
    return 'all' if arm == 'zero' else 'decode'


def arm_vector(vectors, dim, arm):
    arm_phase(arm)
    return None if arm == 'zero' else {LAYER: vectors[dim][arm]}


def load_historical_texts(path, traits):
    """仅读取旧身份和问题/回答，不把旧评分并入新结果。"""
    path = Path(path).resolve()
    if path != (ROOT / REGISTERED_OLD_SAMPLES).resolve() or sha256(path) != OLD_SAMPLES_SHA256:
        raise ValueError('旧强度回答路径或哈希与固定登记不符')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    # 原validator仅核验身份、问题与实验条件，完全不读取评分字段。
    strength.validate_rows(rows, traits)
    result = {}
    for row in rows:
        if row['arm'] not in ('zero', 'decode_mid'):
            continue
        if not isinstance(row.get('a'), str) or not row['a'].strip():
            raise ValueError('历史回答缺失或不是非空文本')
        arm = 'zero' if row['arm'] == 'zero' else 'original'
        key = (row['dim'], row['pole'], arm, row['seed'], row['question_index'], row['sample_index'])
        result[key] = {'q': row['q'], 'a': row['a']}
    if len(result) != 720:
        raise ValueError('历史zero/decode_mid文本身份不完整')
    return result


def historical_overlap(rows, historical):
    details, counts = [], Counter()
    for row in rows:
        if row['arm'] == 'projected':
            continue
        key = tuple(row[k] for k in ('dim', 'pole', 'arm', 'seed', 'question_index', 'sample_index'))
        previous = historical[key]
        if row['q'] != previous['q']:
            raise ValueError('同期与历史的同身份题目不一致')
        identical = row['a'] == previous['a']
        counts[row['dim'], row['pole'], row['arm'], 'compared'] += 1
        counts[row['dim'], row['pole'], row['arm'], 'identical'] += int(identical)
        details.append({**{k: row[k] for k in ('dim', 'pole', 'arm', 'seed', 'question_index', 'sample_index')},
                        'identical_response_text': identical,
                        'current_answer_sha256': hashlib.sha256(row['a'].encode()).hexdigest(),
                        'historical_answer_sha256': hashlib.sha256(previous['a'].encode()).hexdigest()})
    if len(details) != 720:
        raise ValueError('同期720条原向量和零注入重复检查不完整')
    return {'n_compared': 720, 'n_identical': sum(d['identical_response_text'] for d in details),
            'groups': [{'dim': d, 'pole': p, 'arm': a, 'n_compared': counts[d, p, a, 'compared'],
                        'n_identical': counts[d, p, a, 'identical']}
                       for d in COEFS for p in ('high', 'low') for a in ('zero', 'original')],
            'details': details, 'old_scores_used': False,
            'note': '所有新臂均重新生成、独立空缓存评分；与旧文本重复不构成新增独立证据，不合并旧评分。'}


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    source, q_source, old_path = map(lambda p: Path(p).resolve(),
                                    (args.vectors_dir, args.q_source, args.old_samples))
    out = Path(cfg['out_path']) / 'quality_projection' / (
        now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C18', 'experiment': 'quality_projection', 'status': 'running', 'stage': 'validation',
            'start_time': now(), 'end_time': None, 'git_commit': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')), 'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}), 'argv': _redacted(sys.argv[1:]),
            'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'gpu': None, 'judge': None, 'data_hashes': {}, 'vectors_dir': str(source),
            'q_source': str(q_source), 'old_samples': str(old_path), 'old_scores_used': False,
            'layer': LAYER, 'coefs': COEFS, 'arms': ARMS, 'split': 'calib', 'seeds': SEEDS,
            'n_samples_per_question': 3, 'max_new_tokens': 320, 'batch_size': 16,
            'temperature': 1., 'top_p': 1., 'expected_samples': 1080, 'preflight_samples': 8,
            'preflight_tokens': 16, 'max_score_inputs': 3240, 'score_shuffle_seed': 0, 'generated': 0,
            'model_loaded': False, 'model_weights_sha256_verified': False,
            'model_identity_scope': '本地模型JSON/tokenizer文件哈希、形状与冻结状态，未全量核验权重SHA。',
            'formal_e0_pass': False, 'capability_evaluated': False,
            'prompt_hashes': {name: hashlib.sha256(prompt.encode()).hexdigest()
                              for name, prompt in [('coherence', COHERENCE_PROMPT), ('task', TASK_PROMPT)]}}
    rows, phase_traces = [], []

    def save():
        write_json(out / 'run_meta.json', meta)
        (out / 'samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                                  for r in rows), encoding='utf-8', newline='\n')

    def check_sources():
        if any(sha256(path) != digest for path, digest in meta['data_hashes'].items()):
            raise ValueError('实验期间输入发生变化')

    save()
    try:
        if source != (ROOT / REGISTERED_VECTORS).resolve():
            raise ValueError('paired目录不符合固定登记')
        if any(sha256(source / name) != digest for name, digest in REGISTERED_HASHES.items()):
            raise ValueError('paired源向量哈希不符合固定登记')
        base_vectors, shape, inputs = load_vectors(source)
        q, q_hashes = load_quality_source(q_source, shape[2])
        vectors, vector_audit = build_injection_vectors(base_vectors, q)
        write_json(out / 'vector_audit.json', vector_audit)
        torch.save({'layer': LAYER, 'q': q, 'vectors': vectors, 'coefs': COEFS,
                    'q_statistics_sha256': Q_HASHES['statistics.pt']}, out / 'injection_vectors.pt')
        traits = {p: load_trait(p, 'eval') for d in COEFS for p in BIG_FIVE[d]}
        if any(len(t['questions']) != 20 or len(set(t['questions'][:10])) != 10 for t in traits.values()):
            raise ValueError('题库不符合固定calib/test切分')
        historical = load_historical_texts(old_path, traits)
        meta['prompt_hashes'].update({p: hashlib.sha256(t['eval_prompt'].encode()).hexdigest() for p, t in traits.items()})
        model_dir = (ROOT / cfg['model']).resolve()
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise ValueError('需要已下载的本地模型目录及config.json')
        model_files = sorted({p for pattern in ('*.json', '*.txt', '*.model', '*.tiktoken', '*.jinja')
                              for p in model_dir.rglob(pattern) if p.is_file()})
        # q来源使用原directions读尺度；这里只重验实际复用的四个paired极点及模型文件。
        lineage = verify_quality_lineage(q_source, [p for p in inputs if p.name != 'directions.pt'] + model_files)
        write_json(out / 'q_lineage_checks.json', lineage)
        inputs += [Path(args.config).resolve(), old_path, *[KIT / 'trait_eval' / f'{p}.json' for p in traits]]
        inputs += list((ROOT / 'antler').rglob('*.py')) + list(Path(__file__).resolve().parent.glob('*.py'))
        inputs += model_files
        inputs += [ROOT / 'docs' / 'RESEARCH_SPEC.md', ROOT / 'docs' / 'E0_QUALITY_PROJECTION_2026-10-04.md']
        meta['data_hashes'] = {**q_hashes, **{str(p): sha256(p) for p in inputs}}
        meta['injection_vectors_sha256'] = sha256(out / 'injection_vectors.pt')
        meta['data_hashes'][str(out / 'injection_vectors.pt')] = meta['injection_vectors_sha256']
        save()
        judge = Judge(cache_path=str(out / 'judge_cache.jsonl'), response_audit_path=str(out / 'judge_responses.jsonl'))
        if judge.model != 'deepseek-flash':
            raise ValueError('本轮必须沿用deepseek-flash')
        meta['judge'] = _redacted({k: getattr(judge, k) for k in ('model', 'base_url', 'max_tokens', 'extra_body')})
        check_sources()
        model, tok = load_model(str(model_dir), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        meta['model_loaded'] = True
        if (model.config.num_hidden_layers, model.config.hidden_size) != (shape[0], shape[2]):
            raise ValueError('模型与向量形状不一致')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型参数必须全部冻结')
        model.eval()
        meta['all_parameters_frozen'] = True
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if type(eos) is int else set(eos or [])
        if not eos_ids or any(type(t) is not int or t < 0 for t in eos_ids):
            raise ValueError('实际生成配置缺少有效EOS')
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'model_config.json', model.config.to_dict())
        write_json(out / 'generation_settings.json', model.generation_config.to_dict())
        meta['stage'] = 'preflight'
        save()
        # 原8条零向量设计及实现复用，不改变本轮正式生成的种子顺序。
        strength.preflight(model, tok, base_vectors, traits, eos_ids, out)
        check_sources()
        rendered = {target: [chat_prompt(tok, q) for q in trait['questions'][:10]] for target, trait in traits.items()}
        write_json(out / 'chat_prompts.json', rendered)
        meta['chat_prompts_sha256'] = sha256(out / 'chat_prompts.json')
        meta['stage'] = 'generation'
        save()
        for dim in COEFS:
            for pole in ('high', 'low'):
                target = BIG_FIVE[dim][pole == 'low']
                questions, prompts = traits[target]['questions'][:10] * 3, rendered[target] * 3
                for seed in SEEDS:
                    for arm in ARMS:
                        check_sources()
                        trace = []
                        items = generate(model, tok, prompts, steer=arm_vector(vectors, dim, arm),
                            steer_phase=arm_phase(arm), max_new_tokens=320, temperature=1., top_p=1.,
                            batch_size=16, seed=seed, return_details=True, return_token_ids=True, phase_trace_out=trace)
                        identity = {'dim': dim, 'pole': pole, 'arm': arm, 'seed': seed}
                        with (out / 'generation_batches.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                            stream.write(json.dumps({**identity, 'items': items}, ensure_ascii=False, allow_nan=False) + '\n')
                        phase_traces.append({**identity, 'batches': trace})
                        write_json(out / 'phase_traces.json', phase_traces)
                        validate_generation(items, 30, 320, eos_ids)
                        strength.validate_phase_traces(trace, items, arm_phase(arm), 320)
                        if any(tok.batch_decode([item['token_ids']], skip_special_tokens=True)[0] != item['text'] for item in items):
                            raise ValueError('保存token与回答文本不一致')
                        for index, item in enumerate(items):
                            rows.append({**identity, 'target': target, 'layer': LAYER, 'coef': COEFS[dim][arm],
                                'axis': 'pole', 'split': 'calib', 'steer_phase': arm_phase(arm),
                                'question_index': index % 10, 'sample_index': index // 10,
                                'q': questions[index], 'a': item['text'],
                                **{k: item[k] for k in ('token_ids', 'n_generated_tokens', 'finish_reason')}})
                        meta['generated'] = len(rows)
                        save()
                        check_sources()
                        if any(p.requires_grad for p in model.parameters()):
                            raise ValueError('生成期间模型冻结状态发生变化')
                        print(f'[剥离对照生成] {dim}/{pole}/seed{seed}/{arm} 完成，总计{len(rows)}/1080', flush=True)
        validate_rows(rows, traits)
        write_json(out / 'historical_text_overlap.json', historical_overlap(rows, historical))
        check_sources()
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        meta['stage'] = 'scoring'
        save()
        complete = score_rows(judge, rows, traits, out)
        save()
        if not complete:
            raise ValueError('存在缺失评分，保留原结果，不自动重评')
        check_sources()
        meta['stage'] = 'analysis'
        save()
        write_json(out / 'assessment.json', analyze(rows, traits))
        check_sources()
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__,
                    error='有限剥离对照失败，已保留产物；不自动重试、追加幅度或更换评分。')
        raise
    finally:
        meta['end_time'] = now()
        save()
        manifest = {path.name: sha256(path) for path in out.iterdir()
                    if path.is_file() and path.name not in ('artifact_hashes.json', 'results.zip')}
        write_json(out / 'artifact_hashes.json', manifest)
        with zipfile.ZipFile(out / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in [*manifest, 'artifact_hashes.json']:
                archive.write(out / name, name)
        print(f'[剥离对照结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
