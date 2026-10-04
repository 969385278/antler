"""C11：已有坐标与上下文行为的离线描述审计，不拟合、不注入。"""
from __future__ import annotations

from copy import deepcopy
import math
import re

import torch

from antler.traits.data import DIMS


BUNDLES = ('A_reference', 'A_repeat', 'B_reference', 'B_repeat')
REPRESENTATIONS = ('dual_raw', 'dual_centered', 'axis_raw', 'axis_centered')


def _number(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{label} 必须是有限数值')
    return float(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _sign(value):
    return int(value > 0) - int(value < 0)


def _identity(row):
    if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
            or not row['id'].strip() or not _digest(row.get('text_sha256'))):
        raise ValueError('经历须包含唯一非空 id 和有效 text_sha256')
    labels = row.get('intended')
    if (not isinstance(labels, dict) or len(labels) != 1 or not set(labels) <= set(DIMS)
            or any(type(v) is not int or v not in (-1, 1) for v in labels.values())):
        raise ValueError('每段经历须保留唯一的 O/C/E/A/N 整数正负1目标')


def _index(rows):
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError('必须保留全部十段经历')
    result = {}
    for row in rows:
        _identity(row)
        if row['id'] in result:
            raise ValueError('经历 id 重复')
        result[row['id']] = row
    return result


def validate_basis(payload, layer):
    """在 CPU float64 下核验完整方向张量及第 layer 层几何。"""
    if not isinstance(payload, dict) or payload.get('dims') != DIMS:
        raise ValueError('directions 维度顺序必须为 O/C/E/A/N')
    u, scales = payload.get('U'), payload.get('s')
    if (type(layer) is not int or layer < 0 or not isinstance(u, torch.Tensor)
            or u.ndim != 3 or u.shape[0] <= layer or u.shape[1] != 5 or u.shape[2] < 5
            or not u.is_floating_point() or not isinstance(scales, torch.Tensor)
            or scales.shape != (u.shape[0],) or not scales.is_floating_point()):
        raise ValueError('方向张量或尺度的形状、类型、层号无效')
    u = u.detach().to(device='cpu', dtype=torch.float64)
    scales = scales.detach().to(device='cpu', dtype=torch.float64)
    if (not torch.isfinite(u).all() or not torch.isfinite(scales).all()
            or (scales <= 0).any()):
        raise ValueError('方向或尺度含非有限值，或尺度不为正')
    if not torch.allclose(u.norm(dim=-1), torch.ones_like(u[:, :, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError('方向每行必须保持单位范数')
    basis = u[layer]
    if int(torch.linalg.matrix_rank(basis)) != 5:
        raise ValueError('方向矩阵不满秩，不能解释为五维对偶坐标')
    return basis, float(scales[layer])


def _validate_bundles(bundles, layer, basis_sha256, expected_ids):
    if not isinstance(bundles, dict) or set(bundles) != set(BUNDLES):
        raise ValueError('须同时提供 A/B 的 reference/repeat 四个坐标包')
    if not _digest(basis_sha256):
        raise ValueError('basis_sha256 无效')
    indexed = {}
    for name in BUNDLES:
        bundle = bundles[name]
        if (not isinstance(bundle, dict) or bundle.get('method') != name[0]
                or type(bundle.get('layer')) is not int or bundle['layer'] != layer
                or bundle.get('basis_sha256') != basis_sha256 or bundle.get('units') != 's_layer'
                or ('dims' in bundle and bundle['dims'] != DIMS)):
            raise ValueError('坐标包的方法、层号、基、单位或维度顺序不一致')
        settings = bundle.get('fit_settings')
        if (not isinstance(settings, dict) or settings.get('projection') != 'dual'
                or settings.get('neutral_count') != 5 or settings.get('steer_phase') != 'all'
                or settings.get('activation_average') != 'all_fit_probes'):
            raise ValueError('坐标须来自既有 dual、五中性、all 阶段设置')
        probes = bundle.get('probes')
        if (not isinstance(probes, list) or not probes
                or any(not isinstance(p, str) or not p.strip() for p in probes)
                or len(set(probes)) != len(probes)):
            raise ValueError('坐标包的探针记录缺失或重复')
        rows = _index(bundle.get('episodes'))
        if expected_ids is not None and set(rows) != set(expected_ids):
            raise ValueError('坐标包不包含登记的全部十段 id')
        for row in rows.values():
            if not isinstance(row.get('group_id'), str) or not row['group_id'].strip():
                raise ValueError('经历家族标识缺失')
            values = row.get('delta')
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError('delta 须为五维有限向量')
            for value in values:
                _number(value, 'delta')
        indexed[name] = rows
    reference = indexed[BUNDLES[0]]
    targets = [tuple(row['intended'].items())[0] for row in reference.values()]
    if set(targets) != {(d, sign) for d in DIMS for sign in (-1, 1)}:
        raise ValueError('十段经历必须恰好包含每维正负各一段')
    for name, rows in indexed.items():
        if set(rows) != set(reference):
            raise ValueError('四包经历集合不一致')
        for identity, row in rows.items():
            if any(row[key] != reference[identity][key] for key in ('text_sha256', 'group_id', 'intended')):
                raise ValueError('经历原文哈希、家族或目标冲突')
    for method in ('A', 'B'):
        if bundles[f'{method}_reference']['fit_settings'] != bundles[f'{method}_repeat']['fit_settings']:
            raise ValueError('同方法的两轮拟合设置改变')
    for repeat in ('reference', 'repeat'):
        if bundles[f'A_{repeat}']['probes'] != bundles[f'B_{repeat}']['probes']:
            raise ValueError('同一轮 A/B 的拟合探针不一致')
    return indexed


def _validate_behavior(behavior, reference, layer, basis_sha256):
    if not isinstance(behavior, dict):
        raise ValueError('行为输入须为完整对象')
    for key, expected in (('layer', layer), ('basis_sha256', basis_sha256), ('units', 's_layer'), ('dims', DIMS)):
        if key in behavior and behavior[key] != expected:
            raise ValueError('行为关联文件的坐标元数据不一致')
    rows = _index(behavior.get('episodes'))
    if set(rows) != set(reference):
        raise ValueError('行为与坐标的经历集合不一致')
    for identity, row in rows.items():
        original = reference[identity]
        if any(row[key] != original[key] for key in ('text_sha256', 'intended')):
            raise ValueError('行为关联的原文哈希或目标不一致')
        if 'group_id' in row and row['group_id'] != original['group_id']:
            raise ValueError('行为关联的家族不一致')
        reports = row.get('context_reports')
        if (not isinstance(reports, list) or len(reports) != 2
                or any(not isinstance(report, dict) for report in reports)
                or {report.get('pole') for report in reports} != {'high', 'low'}):
            raise ValueError('每段须保留高极、低极两份行为报告')
        dim = next(iter(row['intended']))
        for report in reports:
            if (report.get('id') != identity or report.get('dim') != dim
                    or type(report.get('missing_or_invalid_rows')) is not int
                    or report['missing_or_invalid_rows'] != 0 or report.get('status') != 'descriptive'):
                raise ValueError('行为报告身份错误或包含缺失、无效评分')
            metrics = report.get('metrics')
            if not isinstance(metrics, dict) or set(metrics) != {'trait', 'coherence', 'task'}:
                raise ValueError('须完整保留人格、连贯性和任务理解三项指标')
            for metric in metrics.values():
                if not isinstance(metric, dict):
                    raise ValueError('行为指标不是有效对象')
                means = {key: _number(metric.get(key), key) for key in ('context', 'neutral', 'none')}
                if any(not 0 <= value <= 100 for value in means.values()):
                    raise ValueError('行为均分超出 0 到 100')
                effect = _number(metric.get('context_minus_neutral'), '行为差值')
                if not math.isclose(effect, means['context'] - means['neutral'], abs_tol=1e-10):
                    raise ValueError('行为差值与原始两组均分不一致')
                ci = metric.get('question_bootstrap_ci')
                if not isinstance(ci, list) or len(ci) != 2:
                    raise ValueError('行为区间须有上下界')
                lo, hi = (_number(value, '行为区间') for value in ci)
                if not -100 <= lo <= hi <= 100:
                    raise ValueError('行为区间次序或范围错误')
    return rows


def _ci_state(ci):
    return 'positive' if ci[0] > 0 else 'negative' if ci[1] < 0 else 'uncertain'


def _behavior_readout(row):
    """只给低极人格差值翻号；质量分和原始两极结果保持原样。"""
    result = []
    for pole in ('high', 'low'):
        report = deepcopy(next(r for r in row['context_reports'] if r['pole'] == pole))
        trait = report['metrics']['trait']
        sign = 1 if pole == 'high' else -1
        ci = trait['question_bootstrap_ci']
        oriented_ci = list(ci) if sign == 1 else [-ci[1], -ci[0]]
        report['trait_high_axis'] = {
            'effect': sign * trait['context_minus_neutral'],
            'question_bootstrap_ci': oriented_ci,
            'ci_state': _ci_state(oriented_ci),
            'conversion_multiplier': sign,
            'note': '仅作方向比较；高低极题不同，不合并分数或区间。',
        }
        result.append(report)
    return result


def _physical_norm(vector, gram):
    energy = float(vector @ gram @ vector)
    if not math.isfinite(energy) or energy < -1e-12:
        raise ValueError('物理范数发生非有限或负能量错误')
    return math.sqrt(max(0., energy))


def _sign_counts(rows, values):
    per_dim = {d: {'correct': 0, 'total': 0} for d in DIMS}
    for row, vector in zip(rows, values):
        dim, expected = next(iter(row['intended'].items()))
        per_dim[dim]['correct'] += int(_sign(float(vector[DIMS.index(dim)])) == expected)
        per_dim[dim]['total'] += 1
    return {'correct': sum(v['correct'] for v in per_dim.values()), 'total': len(rows),
            'per_dimension': per_dim}


def analyze(bundles, behavior, directions, *, layer, basis_sha256, expected_ids=None):
    """所有十段和四包固定纳入；只计算描述量，不产生验收或因果结论。"""
    basis, scale = validate_basis(directions, layer)
    indexed = _validate_bundles(bundles, layer, basis_sha256, expected_ids)
    reference = indexed[BUNDLES[0]]
    behavior_rows = _validate_behavior(behavior, reference, layer, basis_sha256)
    for identity, row in behavior_rows.items():
        if 'target_coordinates' in row:
            targets = row['target_coordinates']
            target = DIMS.index(next(iter(row['intended'])))
            if (not isinstance(targets, dict) or set(targets) != set(BUNDLES)
                    or any(_number(targets[name], '行为关联目标坐标') != indexed[name][identity]['delta'][target]
                           for name in BUNDLES)):
                raise ValueError('行为关联文件的目标坐标与原始四包不一致')
    ordered = sorted(reference, key=lambda key: (
        DIMS.index(next(iter(reference[key]['intended']))), next(iter(reference[key]['intended'].values()))))
    rows = [reference[key] for key in ordered]
    gram = basis @ basis.T
    result = {
        'task': 'C11', 'scope': '固定十段 dev 的离线坐标与已归档上下文行为描述',
        'layer': layer, 'dims': list(DIMS), 'units': 's_layer', 's_layer': scale,
        'basis_sha256': basis_sha256, 'dtype': 'float64', 'device': 'cpu',
        'n_episodes': len(rows), 'n_groups': len({r['group_id'] for r in rows}),
        'gram': gram.tolist(), 'gram_condition_number': float(torch.linalg.cond(gram)),
        'formal_e1_pass': False, 'behavior_validated': False, 'injection_performed': False,
        'limits': [
            '对偶系数符号不自动等于人格行为方向；非正交基的轴投影为 Gδ。',
            'A 的 Gδ 重构原激活差的普通轴投影；B 的 Gδ 仅是拟合注入向量的轴投影，不是重测激活差。',
            '共同均值能量比例只描述固定样本的几何，不能叫人格解释率，也不证明共同部分来自提示格式。',
            '中心化均值使用本次全部十段 dev；不作为新经历编码器或正式 test 变换。',
            '逐维正负配对仅是已有目标标签配对；同家族不等于只改结局的双胞胎。',
            '中心化保持向量配对差，但若用中心化向量注入，模型的非线性响应可能改变；本轮不测该行为。',
            '沿用每极十题配对 bootstrap 区间；不重新抽样，不作跨经历或跨家族推断。',
            '低极翻号只统一方向标记，不能与高极量表相加或平均。',
            '未生成、未重新评分、未做能力检查；所有质量结果完整保留，不能据此授予注入资格。',
        ], 'bundles': {},
    }
    for name in BUNDLES:
        values = torch.tensor([indexed[name][key]['delta'] for key in ordered], dtype=torch.float64, device='cpu')
        common = values.mean(0)
        centered = values - common
        projected = values @ gram.T
        axis_centered = centered @ gram.T
        matrices = {'dual_raw': values, 'dual_centered': centered,
                    'axis_raw': projected, 'axis_centered': axis_centered}
        if any(not torch.isfinite(matrix).all() for matrix in matrices.values()):
            raise ValueError('坐标变换产生非有限数值')
        norms = [_physical_norm(v, gram) for v in values]
        centered_norms = [_physical_norm(v, gram) for v in centered]
        common_norm = _physical_norm(common, gram)
        total_energy = math.fsum(n * n for n in norms)
        centered_energy = math.fsum(n * n for n in centered_norms)
        common_energy = len(rows) * common_norm ** 2
        if not all(math.isfinite(x) for x in (total_energy, centered_energy, common_energy)):
            raise ValueError('聚合能量非有限')
        if not math.isclose(total_energy, common_energy + centered_energy, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('均值能量分解不一致')
        episodes = []
        comparisons = {rep: {pole: {'point_sign_agree': 0, 'total': len(rows),
                                    'ci_decisive_sign_agree': 0, 'ci_decisive': 0, 'ci_uncertain': 0}
                             for pole in ('high', 'low')} for rep in REPRESENTATIONS}
        for index, row in enumerate(rows):
            dim, intended = next(iter(row['intended'].items()))
            target = DIMS.index(dim)
            reports = _behavior_readout(behavior_rows[row['id']])
            record = {key: deepcopy(row[key]) for key in ('id', 'text_sha256', 'group_id', 'intended')}
            record.update(
                coordinates={key: matrix[index].tolist() for key, matrix in matrices.items()},
                target_coordinates={key: float(matrix[index, target]) for key, matrix in matrices.items()},
                physical_norm_s=norms[index], physical_norm_absolute=norms[index] * scale,
                centered_physical_norm_s=centered_norms[index],
                centered_physical_norm_absolute=centered_norms[index] * scale,
                context_reports=reports, behavior_sign_comparisons={})
            for rep, matrix in matrices.items():
                sign = _sign(float(matrix[index, target]))
                detail = {}
                for report in reports:
                    pole, trait = report['pole'], report['trait_high_axis']
                    point_agree = sign == _sign(trait['effect'])
                    decisive = trait['ci_state'] != 'uncertain'
                    ci_agree = sign == (1 if trait['ci_state'] == 'positive' else -1) if decisive else None
                    detail[pole] = {'coordinate_sign': sign, 'behavior_point_sign': _sign(trait['effect']),
                                    'point_sign_agree': point_agree, 'ci_state': trait['ci_state'],
                                    'ci_decisive_sign_agree': ci_agree}
                    counter = comparisons[rep][pole]
                    counter['point_sign_agree'] += int(point_agree)
                    counter['ci_decisive'] += int(decisive)
                    counter['ci_uncertain'] += int(not decisive)
                    counter['ci_decisive_sign_agree'] += int(ci_agree is True)
                record['behavior_sign_comparisons'][rep] = detail
            episodes.append(record)
        pairs = []
        for dim in DIMS:
            minus = next(i for i, row in enumerate(rows) if row['intended'] == {dim: -1})
            plus = next(i for i, row in enumerate(rows) if row['intended'] == {dim: 1})
            difference = values[plus] - values[minus]
            centered_difference = centered[plus] - centered[minus]
            if not torch.allclose(difference, centered_difference, atol=1e-14, rtol=1e-12):
                raise ValueError('中心化错误地改变了正负配对差值')
            changes = {}
            for pole in ('high', 'low'):
                high_report = next(r for r in episodes[plus]['context_reports'] if r['pole'] == pole)
                low_report = next(r for r in episodes[minus]['context_reports'] if r['pole'] == pole)
                changes[pole] = {metric: {
                    'plus_context_minus_neutral': high_report['metrics'][metric]['context_minus_neutral'],
                    'minus_context_minus_neutral': low_report['metrics'][metric]['context_minus_neutral'],
                    'plus_minus_effect_difference': high_report['metrics'][metric]['context_minus_neutral']
                        - low_report['metrics'][metric]['context_minus_neutral'],
                    'difference_ci': None,
                    'note': '只给点差；不能用两份单独区间相减得到配对差值区间。',
                } for metric in ('trait', 'coherence', 'task')}
            projected_difference = gram @ difference
            pair_norm = _physical_norm(difference, gram)
            cosine = None
            if norms[plus] and norms[minus]:
                cosine = float(values[plus] @ gram @ values[minus]) / (norms[plus] * norms[minus])
                if not math.isfinite(cosine):
                    raise ValueError('配对物理余弦非有限')
                cosine = max(-1., min(1., cosine))
            pairs.append({
                'dim': dim, 'minus_id': rows[minus]['id'], 'plus_id': rows[plus]['id'],
                'minus_group': rows[minus]['group_id'], 'plus_group': rows[plus]['group_id'],
                'same_family': rows[minus]['group_id'] == rows[plus]['group_id'],
                'controlled_outcome_twins': False,
                'pair_scope': '已有目标正负标签配对，未验证情境和行动完全一致或仅结局不同',
                'plus_minus_dual': difference.tolist(),
                'plus_minus_dual_centered': centered_difference.tolist(),
                'plus_minus_axis': projected_difference.tolist(),
                'target_dual_difference': float(difference[DIMS.index(dim)]),
                'target_axis_difference': float(projected_difference[DIMS.index(dim)]),
                'physical_difference_norm_s': pair_norm,
                'physical_difference_norm_absolute': pair_norm * scale,
                'raw_pair_physical_cosine': cosine,
                'centering_preserves_pair_difference': True, 'behavior_effect_differences': changes,
            })
        result['bundles'][name] = {
            'method': name[0], 'axis_projection_scope': 'original_activation_difference_projection'
                if name[0] == 'A' else 'fitted_injection_vector_projection',
            'mean_dual': common.tolist(), 'mean_axis': (gram @ common).tolist(),
            'mean_physical_norm_s': common_norm, 'mean_physical_norm_absolute': common_norm * scale,
            'mean_centered_physical_norm_s': math.fsum(centered_norms) / len(rows),
            'mean_centered_physical_norm_absolute': math.fsum(centered_norms) / len(rows) * scale,
            'sum_physical_energy_s_squared': total_energy,
            'common_mean_energy_s_squared': common_energy,
            'sum_centered_physical_energy_s_squared': centered_energy,
            'common_mean_energy_fraction': common_energy / total_energy if total_energy else None,
            'sign_counts': {rep: _sign_counts(rows, matrix) for rep, matrix in matrices.items()},
            'behavior_sign_counts': comparisons, 'episodes': episodes, 'pairs': pairs,
        }
    # 最终序列化前遍历检查派生值，避免绝对单位乘法溢出被带进结果。
    def finite_tree(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('派生结果包含非有限数值')
        if isinstance(value, dict):
            for child in value.values():
                finite_tree(child)
        elif isinstance(value, list):
            for child in value:
                finite_tree(child)
    finite_tree(result)
    return result
