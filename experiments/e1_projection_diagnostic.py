"""C7：固定12段pilot的双读出探索，仅前向激活，不生成或调用评委。"""
from __future__ import annotations

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
from antler.core.activations import last_prompt_acts
from antler.core.model import chat_prompt, load_model
from antler.memory.write import BASE_SYSTEM, MEMORY_TEMPLATE, NEUTRAL_MEMORIES, probe_questions, projection_coordinates
from antler.traits.data import BIG_FIVE, DIMS, KIT, load_jsonl
from e0_diagnostics import now, write_json, sha256
from e0_extract_coherence import _redacted
from e0_refilter import _git


LAYER = 16
PROBE_N = 30
SEED = 0
BATCH_SIZE = 16
DIRECTIONS_SHA256 = '25495e935a0daee0ed2c8816e87c45745aa87580091679d6f8ecfd0dfdab2f18'
EXPERIENCES_SHA256 = '1a6d62a3b4665a643e1d20e111f44b244249a64d9104cb9f9b6d80922a3b489c'


def add_arguments(ap):
    ap.add_argument('--directions', required=True, help='登记的原始directions.pt')
    ap.add_argument('--experiences', default='data/pilot_experiences.jsonl', help='登记的12段pilot文件')


def _episode_labels(episodes):
    if (not isinstance(episodes, list) or len(episodes) != 12
            or any(not isinstance(e, dict) or not isinstance(e.get('id'), str) or not e['id'].strip()
                   or not isinstance(e.get('text'), str) or not e['text'].strip() for e in episodes)
            or len({e['id'] for e in episodes}) != 12):
        raise ValueError('pilot须完整保留12段唯一、非空经历')
    for episode in episodes:
        labels = episode.get('intended')
        if (not isinstance(labels, dict) or not labels or not set(labels) <= set(DIMS)
                or any(type(v) is not int or v not in (-1, 1) for v in labels.values())):
            raise ValueError('pilot预期方向须为O/C/E/A/N的整数正负1')
    if sum(len(e['intended']) for e in episodes) != 16:
        raise ValueError('pilot须保留固定16个预期方向标签')


def load_inputs(directions, experiences):
    """先核验固定文件与五维基，秩不足时不加载大模型。"""
    directions, experiences = Path(directions), Path(experiences)
    if sha256(directions) != DIRECTIONS_SHA256 or sha256(experiences) != EXPERIENCES_SHA256:
        raise ValueError('directions或pilot文件哈希不符合固定登记')
    payload = torch.load(directions, map_location='cpu', weights_only=True)
    u, scale = payload.get('U'), payload.get('s')
    if (payload.get('dims') != DIMS or not isinstance(u, torch.Tensor) or u.ndim != 3
            or u.shape[0] <= LAYER or u.shape[1] != 5 or u.shape[2] < 5
            or not u.is_floating_point() or not torch.isfinite(u).all()
            or not isinstance(scale, torch.Tensor) or scale.shape != (u.shape[0],)
            or not scale.is_floating_point() or not torch.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('方向张量、维度顺序、层号或层尺度无效')
    if not torch.allclose(u.norm(dim=-1), torch.ones_like(u[:, :, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('directions的每行人格方向须保持单位范数')
    projection_coordinates(torch.zeros(u.shape[2]), u[LAYER], float(scale[LAYER]), projection='dual')
    episodes = load_jsonl(experiences)
    _episode_labels(episodes)
    return u, scale, episodes


def validate_activations(value, n, hidden):
    """每道probe必须恰有一个指定层的有限激活，缺失时整轮停止。"""
    if (not isinstance(value, torch.Tensor) or value.shape != (n, 1, hidden)
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError('最后prompt-token激活缺失、形状不符或包含非有限数值')
    result = value[:, 0].detach().float().cpu()
    if not torch.isfinite(result).all():
        raise ValueError('激活转换为float32后出现非有限数值')
    return result


def _counts(episodes, values):
    per_dimension = {dim: {'hits': 0, 'total_labels': 0, 'positive_labels': 0, 'negative_labels': 0}
                     for dim in DIMS}
    for episode, coordinates in zip(episodes, values):
        for dim, intended in episode['intended'].items():
            group = per_dimension[dim]
            group['hits'] += int(float(coordinates[DIMS.index(dim)]) * intended > 0)
            group['total_labels'] += 1
            group['positive_labels' if intended > 0 else 'negative_labels'] += 1
    return {'hits': sum(v['hits'] for v in per_dimension.values()), 'total_labels': 16,
            'per_dimension': per_dimension}


def analyze(episodes, coordinates):
    """共模与去共模只作描述，标签计数不当作独立经历的总体准确率。"""
    _episode_labels(episodes)
    if set(coordinates) != {'dual', 'dot'}:
        raise ValueError('必须同时完整报告dual和dot读出')
    result = {'n_episodes': 12, 'n_labels': 16, 'n_probes': PROBE_N, 'layer': LAYER,
              'dims': DIMS, 'projections': {}, 'formal_e1_pass': False,
              'scope': '原始第16层双极坐标上的方法A探索；12段经历方向不平衡，16标签不是16个独立经历。'
                       '共模及去共模变化不能证明格式效应消失，不能解释为正式准确率提升。'
                       '不报告KL人格解释率，不验证第21层单极/decode方案或行为串扰，不进入正式E1结论。'}
    for method in ('dual', 'dot'):
        if len(coordinates[method]) != 12:
            raise ValueError('每种读出须完整包含12段经历')
        if any(not isinstance(v, torch.Tensor) or v.shape != (5,) or not v.is_floating_point()
               or not torch.isfinite(v).all() for v in coordinates[method]):
            raise ValueError('经历坐标必须是有限五维浮点向量')
        values = torch.stack(coordinates[method]).detach().double().cpu()
        common = values.mean(0)
        centered = values - common
        per_episode = []
        for episode, raw, removed in zip(episodes, values, centered):
            def hits(v):
                return sum(float(v[DIMS.index(dim)]) * sign > 0 for dim, sign in episode['intended'].items())
            per_episode.append({'id': episode['id'], 'intended': episode['intended'],
                                'raw': raw.tolist(), 'centered': removed.tolist(),
                                'raw_hits': hits(raw), 'centered_hits': hits(removed),
                                'n_labels': len(episode['intended'])})
        result['projections'][method] = {'common_mode': common.tolist(), 'common_mode_norm': float(common.norm()),
            'mean_centered_norm': float(centered.norm(dim=1).mean()), 'raw': _counts(episodes, values),
            'centered': _counts(episodes, centered), 'per_episode': per_episode, 'formal_e1_pass': False}
    return result


def run(args):
    cfg, _, _, _ = setup(args, need_judge=False, need_model=False)
    directions, experiences = Path(args.directions).resolve(), Path(args.experiences).resolve()
    out = Path(cfg['out_path']) / 'e1_projection_diagnostic' / (
        now().replace(':', '').replace('+', '_') + '_' + uuid.uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    meta = {'task': 'C7', 'status': 'running', 'stage': 'validation', 'start_time': now(), 'end_time': None,
        'git_commit': _git('rev-parse', 'HEAD'), 'git_dirty': bool(_git('status', '--porcelain')),
        'config_path': str(Path(args.config).resolve()),
        'config': _redacted({k: v for k, v in cfg.items() if k != 'out_path'}), 'argv': _redacted(sys.argv[1:]),
        'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'gpu': None, 'judge': None, 'judge_called': False, 'generation_performed': False,
        'model_weights_sha256_verified': False,
        'model_identity_scope': '模型路径、JSON与tokenizer文本哈希、版本、完整形状和冻结状态；本轮未核验权重文件全量SHA。',
        'model_loaded': False, 'formal_e1_pass': False, 'layer': LAYER, 'axis': 'bipolar',
        'probe_n': PROBE_N, 'seed': SEED, 'batch_size': BATCH_SIZE, 'n_episodes': 12, 'n_labels': 16,
        'baseline': 'five_neutral_mean', 'projections': ['dual', 'dot'],
        'directions': str(directions), 'experiences': str(experiences), 'data_hashes': {},
        'neutral_groups_completed': 0, 'episodes_completed': 0, 'prompt_count': 0}
    neutral, differences, episode_ids = [], [], []
    coordinates = {'dual': [], 'dot': []}
    write_json(out / 'run_meta.json', meta)

    def save():
        write_json(out / 'run_meta.json', meta)

    def unchanged():
        if any(sha256(path) != digest for path, digest in meta['data_hashes'].items()):
            raise ValueError('运行期间输入文件发生变化')

    try:
        # 文件哈希在模型加载之前记录；五个来源均为提取集，不读取trait_eval。
        paths = [Path(args.config).resolve(), directions, experiences]
        paths += [KIT / 'trait_extract' / f'{BIG_FIVE[dim][0]}.json' for dim in DIMS]
        paths += list((ROOT / 'antler').rglob('*.py')) + [Path(__file__).resolve(),
            ROOT / 'experiments' / '_common.py', ROOT / 'experiments' / 'e0_diagnostics.py',
            ROOT / 'experiments' / 'e0_extract_coherence.py', ROOT / 'experiments' / 'e0_refilter.py',
            ROOT / 'docs' / 'RESEARCH_SPEC.md', ROOT / 'docs' / 'E1_PROJECTION_PILOT_2026-10-04.md']
        model_dir = (ROOT / cfg['model']).resolve()
        if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
            raise ValueError('本轮需要已下载的本地模型目录及config.json')
        paths += sorted(model_dir.glob('*.json'))
        paths += sorted(model_dir.glob('*.txt'))
        meta['data_hashes'] = {str(path): sha256(path) for path in paths}
        save()
        u, scale, episodes = load_inputs(directions, experiences)
        basis = u[LAYER].double()
        gram = basis @ basis.T
        write_json(out / 'basis_geometry.json', {'layer': LAYER, 'dims': DIMS,
            'gram': gram.tolist(), 'condition_number': float(torch.linalg.cond(gram)),
            'basis_condition_number': float(torch.linalg.cond(basis)),
            'rank': int(torch.linalg.matrix_rank(basis)), 'row_norms': basis.norm(dim=1).tolist(),
            'directions_sha256': meta['data_hashes'][str(directions)],
            'note': 'condition_number指Gram矩阵；对偶读出修正非正交坐标，不证明行为串扰消失。'})
        if len(NEUTRAL_MEMORIES) != 5:
            raise ValueError('必须沿用全部五条固定中性记忆')
        probes = probe_questions(n=PROBE_N, seed=SEED)
        if len(probes) != PROBE_N or len(set(probes)) != PROBE_N or any(not isinstance(p, str) or not p.strip() for p in probes):
            raise ValueError('固定提取集探针须为30道非空唯一问题')
        write_json(out / 'probes.json', {'seed': SEED, 'source_split': 'trait_extract', 'questions': probes,
            'source_hashes': {str(path): meta['data_hashes'][str(path)] for path in paths
                              if path.parent.name == 'trait_extract'}})
        prompt_sources = {'base_system': BASE_SYSTEM, 'memory_template': MEMORY_TEMPLATE,
                          'neutral_memories': NEUTRAL_MEMORIES, 'experiences_sha256': meta['data_hashes'][str(experiences)]}
        write_json(out / 'prompt_sources.json', prompt_sources)
        meta['prompt_hashes'] = {'base_system': hashlib.sha256(BASE_SYSTEM.encode()).hexdigest(),
            'memory_template': hashlib.sha256(MEMORY_TEMPLATE.encode()).hexdigest(),
            'neutral_memories': [hashlib.sha256(m.encode()).hexdigest() for m in NEUTRAL_MEMORIES],
            'probes': sha256(out / 'probes.json')}
        meta['directions_shape'] = list(u.shape)
        meta['scale_shape'] = list(scale.shape)
        meta['s_layer'] = float(scale[LAYER])
        torch.manual_seed(SEED)
        model, tok = load_model(str(model_dir), cfg.get('dtype', 'bfloat16'), cfg.get('device', 'auto'))
        meta['model_loaded'] = True
        meta['model_shape'] = {'num_hidden_layers': model.config.num_hidden_layers, 'hidden_size': model.config.hidden_size}
        if (model.config.num_hidden_layers, model.config.hidden_size) != (u.shape[0], u.shape[2]):
            raise ValueError('完整模型层数或隐藏维度与directions不一致')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('模型参数未全部冻结')
        meta['all_parameters_frozen'] = True
        meta['parameter_count'] = sum(p.numel() for p in model.parameters())
        meta['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(out / 'model_config.json', model.config.to_dict())
        model.eval()
        # 保存实际前向使用的完整模板化提示，再复用同一份字符串列表。
        rendered = {'probe_sha256': meta['prompt_hashes']['probes'],
            'neutral': [{'index': i, 'prompts': [chat_prompt(tok, p, system=MEMORY_TEMPLATE.format(experience=m))
                                               for p in probes]} for i, m in enumerate(NEUTRAL_MEMORIES)],
            'episodes': [{'id': e['id'], 'prompts': [chat_prompt(tok, p, system=MEMORY_TEMPLATE.format(experience=e['text']))
                                                  for p in probes]} for e in episodes]}
        write_json(out / 'chat_prompts.json', rendered)
        meta['prompt_hashes']['rendered'] = sha256(out / 'chat_prompts.json')
        unchanged()
        meta['stage'] = 'neutral_activations'
        save()
        for index, memory in enumerate(NEUTRAL_MEMORIES):
            prompts = rendered['neutral'][index]['prompts']
            with torch.no_grad():
                acts = last_prompt_acts(model, tok, prompts, layers=[LAYER], batch_size=BATCH_SIZE)
            neutral.append(validate_activations(acts, PROBE_N, u.shape[2]))
            torch.save({'layer': LAYER, 'per_memory': neutral, 'neutral_mean': None,
                        'probe_sha256': meta['prompt_hashes']['probes'],
                        'neutral_memory_sha256': meta['prompt_hashes']['neutral_memories']}, out / 'neutral_activations.pt')
            meta['neutral_groups_completed'] = index + 1
            meta['prompt_count'] += PROBE_N
            save()
            unchanged()
        neutral_mean = torch.stack(neutral).mean(0)
        if not torch.isfinite(neutral_mean).all():
            raise ValueError('五条中性记忆的平均激活包含非有限值')
        torch.save({'layer': LAYER, 'per_memory': neutral, 'neutral_mean': neutral_mean,
                    'probe_sha256': meta['prompt_hashes']['probes'],
                    'neutral_memory_sha256': meta['prompt_hashes']['neutral_memories']}, out / 'neutral_activations.pt')
        meta['neutral_activations_sha256'] = sha256(out / 'neutral_activations.pt')
        meta['stage'] = 'episode_activations'
        save()
        for episode_index, episode in enumerate(episodes):
            prompts = rendered['episodes'][episode_index]['prompts']
            with torch.no_grad():
                acts = last_prompt_acts(model, tok, prompts, layers=[LAYER], batch_size=BATCH_SIZE)
            values = validate_activations(acts, PROBE_N, u.shape[2])
            mean_delta = (values - neutral_mean).mean(0)
            if not torch.isfinite(mean_delta).all():
                raise ValueError('经历平均激活差包含非有限值')
            differences.append(mean_delta)
            episode_ids.append(episode['id'])
            torch.save({'layer': LAYER, 'episode_ids': episode_ids,
                        'mean_differences': torch.stack(differences),
                        'neutral_activations_sha256': meta['neutral_activations_sha256']}, out / 'activation_differences.pt')
            # 两种读出严格复用同一激活差；不重新前向，也不根据标签调整坐标。
            for method in ('dual', 'dot'):
                coordinates[method].append(projection_coordinates(mean_delta, u[LAYER], float(scale[LAYER]), projection=method))
            write_json(out / 'partial_coordinates.json', {method: [v.tolist() for v in values]
                                                           for method, values in coordinates.items()})
            meta['episodes_completed'] += 1
            meta['prompt_count'] += PROBE_N
            save()
            unchanged()
            print(f'[投影探索] {episode["id"]} 完成，{meta["episodes_completed"]}/12', flush=True)
        unchanged()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('测量期间模型参数冻结状态发生变化')
        write_json(out / 'results.json', analyze(episodes, coordinates))
        meta.update(status='completed', stage='completed')
    except BaseException as exc:
        meta.update(status='failed', error_type=type(exc).__name__,
                    error='投影探索失败，已保留完成部分；不自动重试或追加条件。')
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
        print(f'[投影探索结果目录] {out}', flush=True)
    return out


if __name__ == '__main__':
    run(parse_args(add_arguments))
