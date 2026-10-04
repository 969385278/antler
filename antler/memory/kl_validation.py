"""C11：共享老师token上的蒸馏与留出KL评估；不生成回答或声明行为通过。"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from .write import _seq_logprobs
from ..core.model import model_device


@dataclass(frozen=True)
class KLItem:
    """同一道探针的固定老师分布和五个中性记忆提示，全部保存在CPU。"""
    probe: str
    response_ids: torch.Tensor
    teacher_logp: torch.Tensor
    neutral_prompt_ids: tuple[torch.Tensor, ...]


def validate_items(items):
    if not items or len({i.probe.strip() for i in items}) != len(items):
        raise ValueError('探针集合不能为空或含重复题')
    for item in items:
        if not item.probe.strip() or len(item.neutral_prompt_ids) != 5:
            raise ValueError('每题须有五条中性记忆提示')
        for ids in (item.response_ids, *item.neutral_prompt_ids):
            if ids.ndim != 1 or not ids.numel() or ids.dtype != torch.long or ids.device.type != 'cpu' or (ids < 0).any():
                raise ValueError('token须为非空CPU long一维非负整数')
        logp = item.teacher_logp
        if (logp.ndim != 2 or logp.shape[0] != len(item.response_ids) or not logp.is_floating_point()
                or logp.device.type != 'cpu' or logp.requires_grad or not torch.isfinite(logp).all()
                or not torch.allclose(logp.logsumexp(-1), torch.zeros_like(logp[:, 0]), atol=1e-4)):
            raise ValueError('老师分布须为已归一化、脱离梯度的有限CPU log概率')
        if any(int(ids.max()) >= logp.shape[1] for ids in (item.response_ids, *item.neutral_prompt_ids)):
            raise ValueError('token超出老师词表')


def _check_model(model):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('模型必须处于eval且权重冻结')


def _kl(model, item, neutral_ids, layer, vector):
    device = model_device(model)
    student = _seq_logprobs(model, neutral_ids.to(device), item.response_ids.to(device),
                           steer=None if vector is None else {layer: vector})
    teacher = item.teacher_logp.to(device)
    if student.shape != teacher.shape:
        raise ValueError('老师与学生词表或token数量不一致')
    kl = (teacher.exp() * (teacher - student)).sum(-1).mean()
    if not torch.isfinite(kl):
        raise ValueError('KL出现非有限值')
    return kl


@torch.no_grad()
def evaluate_kl(model, items, *, layer, vector=None):
    """先按token平均，再按五条记忆、题目平均；不同长度题目等权。"""
    _check_model(model)
    validate_items(items)
    if vector is not None:
        if vector.shape != (model.config.hidden_size,) or not torch.isfinite(vector).all():
            raise ValueError('注入向量形状或数值无效')
        vector = vector.detach().to(model_device(model))
    per_probe = []
    for item in items:
        values = [float(_kl(model, item, ids, layer, vector)) for ids in item.neutral_prompt_ids]
        per_probe.append({'probe': item.probe, 'neutral_kls': values, 'kl': sum(values) / 5,
                          'n_response_tokens': len(item.response_ids)})
    return {'kl': sum(r['kl'] for r in per_probe) / len(items), 'per_probe': per_probe,
            'n_probes': len(items), 'neutral_count': 5, 'steer_phase': 'all'}


def fit_and_evaluate(model, fit_items, eval_items, *, basis, scale, layer,
                     steps=60, lr=.02, lam=1e-3, probes_per_step=4, seed=0, init=None,
                     diagnostics=False):
    """只在fit探针拟合，最后一次更新后完整评估留出集。

    basis为k×d；None直接优化d维，不构造巨大单位矩阵。正则为注入向量/s的平方范数，
    对不同坐标基采用同一个几何量。旧delta_distill入口不变。本入口仅实现all阶段。
    """
    _check_model(model)
    validate_items(fit_items)
    validate_items(eval_items)
    normalize = lambda s: ' '.join(s.casefold().split())
    fit_keys, eval_keys = [normalize(i.probe) for i in fit_items], [normalize(i.probe) for i in eval_items]
    if (len(set(fit_keys)) != len(fit_keys) or len(set(eval_keys)) != len(eval_keys)
            or set(fit_keys) & set(eval_keys)):
        raise ValueError('拟合与评估探针规范化后重复或重叠')
    if (type(steps) is not int or steps < 1 or type(probes_per_step) is not int or probes_per_step < 1
            or not all(math.isfinite(v) for v in (scale, lr, lam)) or scale <= 0 or lr <= 0 or lam < 0):
        raise ValueError('优化参数无效')
    device = model_device(model)
    hidden = model.config.hidden_size
    if basis is not None:
        basis = basis.detach().to(device=device, dtype=torch.float32)
        if (basis.ndim != 2 or basis.shape[1] != hidden or not 1 <= basis.shape[0] <= hidden
                or not torch.isfinite(basis).all() or int(torch.linalg.matrix_rank(basis)) != basis.shape[0]):
            raise ValueError('坐标基须为有限满行秩k×d矩阵')
    size = hidden if basis is None else basis.shape[0]
    params = torch.zeros(size, device=device) if init is None else init.detach().clone().to(device=device, dtype=torch.float32)
    if params.shape != (size,) or not torch.isfinite(params).all():
        raise ValueError('初始坐标形状或数值无效')
    params.requires_grad_(True)
    vector = lambda: scale * (params if basis is None else params @ basis)
    zero = evaluate_kl(model, eval_items, layer=layer)
    diagnostic = None
    if diagnostics:
        initial_vector = vector().detach().clone()
        diagnostic = {'initial_vector_norm': float(initial_vector.norm()),
                      'parameter_count': size, 'hidden_size': hidden,
                      'fit_baseline': evaluate_kl(model, fit_items, layer=layer),
                      'fit_initial': evaluate_kl(model, fit_items, layer=layer, vector=initial_vector),
                      'heldout_initial': evaluate_kl(model, eval_items, layer=layer, vector=initial_vector),
                      'step_metrics': []}
    optimizer = torch.optim.Adam([params], lr=lr)
    rng = random.Random(seed)
    history = []
    for step in range(steps):
        before_vector = vector().detach().clone() if diagnostics else None
        batch = rng.sample(fit_items, min(probes_per_step, len(fit_items)))
        optimizer.zero_grad()
        total = 0.
        for item in batch:
            for ids in item.neutral_prompt_ids:
                loss = _kl(model, item, ids, layer, vector()) / (5 * len(batch))
                loss.backward()
                total += float(loss.detach())
        penalty = lam * (vector() / scale).square().sum()
        penalty.backward()
        grad_norm = float(params.grad.detach().norm()) if diagnostics else None
        optimizer.step()
        if not torch.isfinite(params).all():
            raise ValueError('优化产生非有限坐标')
        history.append({'step': step, 'fit_minibatch_kl_before_update': total})
        if diagnostics:
            after_vector = vector().detach()
            diagnostic['step_metrics'].append({'step': step,
                'parameter_gradient_norm': grad_norm,
                'vector_update_norm': float((after_vector - before_vector).norm()),
                'vector_update_norm_over_scale': float((after_vector - before_vector).norm()) / scale,
                'vector_norm_after_update': float(after_vector.norm()),
                'regularizer_before_update': float(penalty.detach())})
    final_vector = vector().detach()
    final = evaluate_kl(model, eval_items, layer=layer, vector=final_vector)
    result = {'coordinates': params.detach().cpu(), 'vector': final_vector.cpu(), 'baseline': zero,
            'final': final, 'explained': (zero['kl'] - final['kl']) / zero['kl'] if zero['kl'] > 0 else None,
            'history': history, 'fit_probes': [i.probe for i in fit_items],
            'evaluation_probes': [i.probe for i in eval_items], 'steer_phase': 'all',
            'regularizer': 'squared_norm_of_vector_over_scale', 'behavior_validated': False}
    if diagnostics:
        diagnostic['fit_final'] = evaluate_kl(model, fit_items, layer=layer, vector=final_vector)
        diagnostic['optimizer'] = {'name': 'Adam', 'lr': lr, 'eps': optimizer.param_groups[0]['eps'],
                                   'betas': optimizer.param_groups[0]['betas']}
        result['diagnostics'] = diagnostic
    return result
