"""C18：固定旧提取回答的连贯性相关方向探索，只重放回答读取激活。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
import zipfile

import numpy as np
import torch
import transformers

from _common import parse_args, setup
from antler.config import ROOT
from antler.core.activations import response_mean_acts
from antler.core.model import chat_prompt, load_model
from antler.eval.judge import COHERENCE_PROMPT, TASK_PROMPT
from antler.traits.data import DIMS, KIT, load_trait
from antler.traits.quality_direction import analyze_quality_direction, quality_question_folds
from antler.traits.vectors import extraction_rows
from e0_diagnostics import now, sha256, write_json
from e0_extract_coherence import _redacted, _score_is_valid, load_inputs
from e0_refilter import _git


LAYER = 21
BATCH_SIZE = 16
BOOTSTRAP_SAMPLES = 1000
SEED = 0
POLES = ('dependable', 'careless', 'nervous', 'calm')
C17_SHA256 = '6add4da12b7a09f24148c43911a98fa0751f3034e5e8d564076d6fa2f8c09162'
DIRECTIONS_SHA256 = '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18'
VECTOR_SHA256 = {
    'dependable': '39676e4ae642b96cf163899c133cbc9ec27a18b77b7d5c1d8818fa90f28a9cdf',
    'careless': 'cc6c27f273e0e78d3ed069ab51ea360f00ea1e76d8ec0bfcf749d003c4e6b539',
    'nervous': '4caa81abb73a3ee01b6bde595dab19a7f879057d7190d7325468d955b4f9f507',
    'calm': '93479c98d977519eae79822e7bacf9fa9eaa251b4d94fbc49abfb2ee18fdfcfa',
}


def add_arguments(ap):
    ap.add_argument('--source-out', default='outputs/main', help='原始回答、C17分数及层尺度的目录')
    ap.add_argument('--pole-vectors', default=(
        'outputs/main/refilter_diagnostics/20261003T022259Z_099e5cd2/variants/paired/vectors'),
        help='登记的完整paired向量目录，仅读取固定四极点')


def _text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _pinned(path, expected, name):
    digest = sha256(path)
    if digest != expected:
        raise ValueError(f'{name}文件哈希不符合固定登记')
    return digest


def prepare_inputs(cfg, source_out, pole_vectors):
    """模型加载前核对全部1600条来源，缺失、改写或错序均停止。"""
    source_out, pole_vectors = Path(source_out).resolve(), Path(pole_vectors).resolve()
    c17 = source_out / 'extract_coherence_samples.jsonl'
    directions_path = source_out / 'directions.pt'
    hashes = {
        str(c17): _pinned(c17, C17_SHA256, 'C17'),
        str(directions_path): _pinned(directions_path, DIRECTIONS_SHA256, '原始directions'),
    }
    directions = torch.load(directions_path, map_location='cpu', weights_only=True)
    if not isinstance(directions, dict) or directions.get('dims') != DIMS:
        raise ValueError('原始方向维度顺序必须为O C E A N')
    u, scale = directions.get('U'), directions.get('s')
    if (not isinstance(u, torch.Tensor) or u.ndim != 3 or u.shape[0] <= LAYER
            or u.shape[1] != 5 or u.shape[2] < 1 or not u.is_floating_point()
            or not torch.isfinite(u).all()
            or not torch.allclose(u.norm(dim=-1), torch.ones_like(u[..., 0]), atol=1e-5, rtol=1e-5)
            or not isinstance(scale, torch.Tensor) or scale.shape != (u.shape[0],)
            or not scale.is_floating_point() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('原始方向张量或层尺度无效')
    source_meta = directions.get('meta', {})
    if not isinstance(source_meta, dict):
        raise ValueError('原始方向元数据无效')
    if source_meta.get('model') not in (None, cfg['model']):
        raise ValueError('配置与原始方向的模型标识不一致')
    if (type(cfg.get('n_rollouts', 2)) is not int or cfg.get('n_rollouts', 2) != 2
            or cfg.get('max_questions') not in (None, 0, 20)):
        raise ValueError('本轮固定每极点20题、两次生成，不允许改变提取配置')
    sources = load_inputs(source_out, POLES)
    records = [json.loads(line) for line in c17.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(records) != 1600:
        raise ValueError('C17必须完整包含固定四极点1600条记录')
    by_identity = {}
    for record in records:
        if not isinstance(record, dict) or record.get('pole') not in POLES:
            raise ValueError('C17出现未登记极点或无效记录')
        pole, index = record['pole'], record.get('source_index')
        if type(index) is not int or not 0 <= index < 400 or (pole, index) in by_identity:
            raise ValueError('C17 source_index重复、越界或无效')
        samples = sources[pole]['samples']
        if len(samples) != 400:
            raise ValueError(f'{pole}: 原始回答必须完整包含400条')
        sample = samples[index]
        if record.get('source_sha256') != sources[pole]['source_sha256']:
            raise ValueError(f'{pole}: C17 source_sha256与原回答不一致')
        for field, original in (('side', 'side'), ('question', 'question'),
                                ('answer', 'answer'), ('trait_score', 'score')):
            if field not in record or record[field] != sample[original]:
                raise ValueError(f'{pole}: source_index={index}的{field}与原回答不一致')
        if not _score_is_valid(record.get('trait_score')) or not _score_is_valid(record.get('coherence')):
            raise ValueError('本轮要求全部人格分与连贯性分有效，不允许丢弃缺分样本')
        if not sample['question'].strip() or not sample['answer'].strip():
            raise ValueError('本轮不允许空题目或空回答')
        by_identity[pole, index] = record
    rows, vectors = [], {}
    for pole in POLES:
        source = sources[pole]
        restored = extraction_rows(pole, n_rollouts=2, samples=source['samples'])
        if (len(restored) != 400 or {r['pair_index'] for r in restored} != set(range(5))
                or {r['question_index'] for r in restored} != set(range(20))):
            raise ValueError('当前提取库须保持5对提示和20题')
        hashes[source['source_file']] = source['source_sha256']
        trait_path = (KIT / 'trait_extract' / f'{pole}.json').resolve()
        hashes[str(trait_path)] = sha256(trait_path)
        for row in restored:
            record = by_identity[pole, row['source_index']]
            rows.append({**row, 'pole': pole, 'answer': record['answer'],
                         'score': record['trait_score'], 'coherence': record['coherence'],
                         'source_sha256': source['source_sha256'], 'source_file': source['source_file']})
        vector_path = pole_vectors / f'{pole}.pt'
        hashes[str(vector_path)] = _pinned(vector_path, VECTOR_SHA256[pole], f'paired {pole}')
        vector = torch.load(vector_path, map_location='cpu', weights_only=True)
        if (not isinstance(vector, torch.Tensor) or vector.shape != (u.shape[0], u.shape[2])
                or not vector.is_floating_point() or not torch.isfinite(vector).all()
                or float(vector[LAYER].double().norm()) <= 1e-8):
            raise ValueError(f'{pole}: paired向量层数、隐藏维度或数值无效')
        vectors[pole] = vector[LAYER].detach().cpu()
    return {'rows': quality_question_folds(rows), 'reference_vectors': vectors,
            's_layer': float(scale[LAYER]), 'n_layers': u.shape[0], 'hidden_size': u.shape[2],
            'hashes': hashes, 'source_metadata': _redacted(source_meta)}


def replay_token_spans(tok, prompts, answers):
    """严格沿用response_mean_acts的回答切片，长度不是历史生成token数。"""
    if len(prompts) != len(answers) or not prompts:
        raise ValueError('重放提示与回答数量必须一致且非空')
    tok.padding_side = 'right'
    try:
        starts = [len(tok(p, add_special_tokens=False)['input_ids']) for p in prompts]
        enc = tok([p + a for p, a in zip(prompts, answers)], return_tensors='pt',
                  padding=True, add_special_tokens=False)
        totals = enc['attention_mask'].sum(1).tolist()
        spans = []
        for index, (start, total) in enumerate(zip(starts, totals)):
            end = int(total)
            if end <= start:
                raise ValueError('重分词后回答切片为空，停止整轮重放')
            mask = enc['attention_mask'][index].tolist()
            if mask != [1] * end + [0] * (len(mask) - end):
                raise ValueError('重放分词器没有产生连续右填充mask')
            ids = enc['input_ids'][index, :end].tolist()
            spans.append({'replayed_prompt_tokens': start, 'replayed_total_tokens': end,
                          'replayed_response_tokens': end - start,
                          'replayed_input_ids_sha256': _text_hash(json.dumps(ids, separators=(',', ':'))),
                          'replayed_response_ids_sha256': _text_hash(json.dumps(ids[start:end], separators=(',', ':')))})
        return spans
    finally:
        # 与现有激活读取函数保持同样的退出状态。
        tok.padding_side = 'left'


def validate_activations(value, n, hidden):
    if (not isinstance(value, torch.Tensor) or value.shape != (n, 1, hidden)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('回答平均激活形状不符、缺失或包含非有限数值')
    result = value[:, 0].detach().float().cpu()
    if not torch.isfinite(result).all():
        raise ValueError('回答平均激活转为float32后溢出')
    return result


def _stat_tensors(value):
    """统计模块的数组转成可weights_only读取的CPU张量。"""
    if isinstance(value, dict):
        return {k: _stat_tensors(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stat_tensors(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if hasattr(value, 'dtype') and hasattr(value, 'shape'):
        return torch.as_tensor(value).cpu()
    return value


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    source_out, vector_dir = Path(args.source_out).resolve(), Path(args.pole_vectors).resolve()
    out = Path(cfg['out_path']) / 'quality_direction' / (
        now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C18', 'experiment': 'quality_direction', 'status': 'running', 'stage': 'validation',
            'start_time': now(), 'end_time': None, 'git_commit': _git('rev-parse', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')), 'config_path': str(Path(args.config).resolve()),
            'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}),
            'argv': _redacted(sys.argv[1:]), 'torch': torch.__version__, 'transformers': transformers.__version__,
            'numpy': np.__version__, 'cpu_threads': {'torch': torch.get_num_threads(), **{
                name: os.environ.get(name) for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                       'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}},
            'cuda': torch.version.cuda, 'gpu': None, 'model_loaded': False, 'judge': None,
            'judge_called': False, 'generation_performed': False, 'injection_performed': False,
            'trait_eval_used': False, 'formal_e0_pass': False, 'layer': LAYER, 'poles': list(POLES),
            'batch_size': BATCH_SIZE, 'seed': SEED, 'bootstrap_samples': BOOTSTRAP_SAMPLES,
            'source_out': str(source_out), 'pole_vectors': str(vector_dir), 'data_hashes': {},
            'n_answers': 1600, 'answers_completed': 0, 'batches_completed': 0,
            'historical_token_ids_available': False, 'historical_finish_reason_available': False,
            'model_weights_sha256_verified': False,
            'model_identity_scope': '记录本地模型JSON、tokenizer文件哈希、形状、冻结状态；不核验全部权重SHA。',
            'reconstruction_limitation': '旧回答未保存历史系统提示哈希、生成token ID或停止原因。'
                '按当前提取库还原提示并核对side/question；本轮记录重放token切片，不能声称恢复历史token流。'}
    batches, rendered_rows = [], []

    def save():
        write_json(out / 'run_meta.json', meta)

    def unchanged():
        for path, digest in meta['data_hashes'].items():
            if sha256(path) != digest:
                raise ValueError(f'运行期间输入发生变化：{path}')

    def checkpoint():
        if batches:
            torch.save({'layer': LAYER, 'rows': [
                {'pole': r['pole'], 'source_index': r['source_index']} for r in rendered_rows[:meta['answers_completed']]],
                'response_mean_activations': torch.cat(batches),
                'folds_sha256': meta['folds_sha256'], 's_layer': meta['s_layer']},
                out / 'response_mean_activations.pt')

    save()
    try:
        prepared = prepare_inputs(cfg, source_out, vector_dir)
        meta['data_hashes'] = dict(prepared['hashes'])
        meta['source_metadata'] = prepared['source_metadata']
        paths = [Path(args.config).resolve(), Path(__file__).resolve(),
                 ROOT / 'experiments' / '_common.py', ROOT / 'experiments' / 'e0_diagnostics.py',
                 ROOT / 'experiments' / 'e0_extract_coherence.py', ROOT / 'experiments' / 'e0_refilter.py',
                 ROOT / 'docs' / 'RESEARCH_SPEC.md', ROOT / 'docs' / 'E0_QUALITY_DIRECTION_2026-10-04.md']
        paths += sorted((ROOT / 'antler').rglob('*.py'))
        model_dir = (ROOT / cfg['model']).resolve()
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise ValueError('本轮仅使用已下载的本地模型目录，缺少config.json')
        paths += sorted({p for pattern in ('*.json', '*.txt', '*.model', '*.tiktoken')
                         for p in model_dir.rglob(pattern) if p.is_file()})
        meta['data_hashes'].update({str(p.resolve()): sha256(p) for p in paths})
        rows = prepared['rows']
        # 折分在模型加载前封存，后续统计严格复用这些身份。
        write_json(out / 'folds.json', {'method': 'sha256-quality-direction-v1-two-fold', 'rows': rows})
        meta['folds_sha256'] = sha256(out / 'folds.json')
        meta['data_hashes'][str((out / 'folds.json').resolve())] = meta['folds_sha256']
        prompt_sources = {'coherence_prompt': COHERENCE_PROMPT, 'task_prompt': TASK_PROMPT,
                          'trait_eval_prompts': {p: load_trait(p, 'extract')['eval_prompt'] for p in POLES}}
        write_json(out / 'fixed_score_prompts.json', prompt_sources)
        meta['score_prompt_sha256'] = {'coherence': _text_hash(COHERENCE_PROMPT), 'task': _text_hash(TASK_PROMPT),
                                      'trait': {p: _text_hash(v) for p, v in prompt_sources['trait_eval_prompts'].items()}}
        meta.update(s_layer=prepared['s_layer'], stage='load_model')
        save()
        unchanged()
        torch.manual_seed(SEED)
        model, tok = load_model(str(model_dir), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        meta['model_loaded'] = True
        if (model.config.num_hidden_layers, model.config.hidden_size) != (prepared['n_layers'], prepared['hidden_size']):
            raise ValueError('完整模型层数或隐藏维度与原方向不一致')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型参数必须全部冻结')
        meta['all_parameters_frozen'] = True
        meta['parameter_count'] = sum(p.numel() for p in model.parameters())
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        meta['model_shape'] = {'num_hidden_layers': prepared['n_layers'], 'hidden_size': prepared['hidden_size']}
        write_json(out / 'model_config.json', model.config.to_dict())
        model.eval()
        meta['stage'] = 'answer_activations'
        save()
        with (out / 'replayed_inputs.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for start in range(0, len(rows), BATCH_SIZE):
                unchanged()
                selected = rows[start:start + BATCH_SIZE]
                prompts = [chat_prompt(tok, r['question'], system=r['system']) for r in selected]
                answers = [r['answer'] for r in selected]
                spans = replay_token_spans(tok, prompts, answers)
                batch_rows = [{**r, 'rendered_prompt': p, 'rendered_prompt_sha256': _text_hash(p), **span}
                              for r, p, span in zip(selected, prompts, spans)]
                for row in batch_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                rendered_rows.extend(batch_rows)
                with torch.no_grad():
                    acts = response_mean_acts(model, tok, prompts, answers, layers=[LAYER], batch_size=BATCH_SIZE)
                batches.append(validate_activations(acts, len(selected), prepared['hidden_size']))
                meta['answers_completed'] += len(selected)
                meta['batches_completed'] += 1
                checkpoint()
                save()
                unchanged()
                if any(p.requires_grad for p in model.parameters()):
                    raise ValueError('重放期间模型冻结状态发生变化')
                print(f'[连贯方向重放] {meta["answers_completed"]}/1600，批次{meta["batches_completed"]}/100', flush=True)
        if meta['answers_completed'] != 1600 or len(rendered_rows) != 1600:
            raise ValueError('固定1600条回答尚未完整重放')
        meta['replayed_inputs_sha256'] = sha256(out / 'replayed_inputs.jsonl')
        meta['activations_sha256'] = sha256(out / 'response_mean_activations.pt')
        meta['data_hashes'][str((out / 'replayed_inputs.jsonl').resolve())] = meta['replayed_inputs_sha256']
        meta['data_hashes'][str((out / 'response_mean_activations.pt').resolve())] = meta['activations_sha256']
        meta['stage'] = 'cpu_statistics'
        save()
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 模型前向结束后只在CPU重算登记统计，不追加任何生成、评分或注入。
        try:
            result = analyze_quality_direction(rows, torch.cat(batches),
                [r['replayed_response_tokens'] for r in rendered_rows], prepared['s_layer'],
                prepared['reference_vectors'], bootstrap_samples=BOOTSTRAP_SAMPLES, seed=SEED)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # 仅记录纯统计模块的受控诊断，不转存模型/网络异常或请求内容。
            write_json(out / 'statistics_failure.json', {'error_type': type(exc).__name__,
                'error': str(exc), 'scope': '本轮纯CPU统计模块，无模型或评委外部异常',
                'activations_sha256': meta['activations_sha256'], 'folds_sha256': meta['folds_sha256']})
            raise
        summary, artifacts = result['summary'], result['artifacts']
        write_json(out / 'summary.json', summary)
        torch.save(_stat_tensors(artifacts), out / 'statistics.pt')
        meta['bootstrap_complete'] = bool(summary['bootstrap']['complete'])
        unchanged()
        if not summary['bootstrap']['complete']:
            raise ValueError('固定bootstrap出现无效拟合，统计已留档，诊断不完整')
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__,
                    error='连贯性方向探索失败，已保留已完成部分；不自动重试或扩展条件。')
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
        print(f'[连贯方向结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
