"""C11：完整开发网格的校验与固定选参，不接触正式test。"""
import math
from statistics import mean

from .e1_dev_design import learning_rates


def summarize_grid(results, manifest, *, hidden):
    """拒绝预检、缺臂或非有限结果；按每空间平均最终留出KL选参。"""
    episodes = manifest['episodes']
    expected = [(d, s) for d in 'OCEAN' for s in (-1, 1)]
    if (manifest.get('scope') != 'dev_optimization_only'
            or len(episodes) != 10
            or [r['intended'] for r in episodes] != [{d: s} for d, s in expected]
            or any(r['split'] != 'dev' for r in episodes)
            or len({r['id'] for r in episodes}) != 10):
        raise ValueError('需要冻结的十段平衡dev清单')
    rows = results['episodes']
    if results.get('preflight') is not False or results.get('formal_e1_pass') is not False:
        raise ValueError('只汇总完整开发诊断，不接受预检或正式通过标记')
    if len(rows) != 10 or {r['id'] for r in rows} != {r['id'] for r in episodes}:
        raise ValueError('十段dev结果缺失或重复')
    rates = learning_rates(hidden)
    names = {f'{space}_lr{i}' for space in rates for i in range(3)}
    intended = {r['id']: r['intended'] for r in episodes}

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('结果包含非有限或非法数值')
        return value

    for row in rows:
        if row['intended'] != intended[row['id']] or set(row['arms']) != names:
            raise ValueError('目标标签改变或九臂不完整')
        baseline = None
        probe_pair = None
        for name, arm in row['arms'].items():
            space, index = name.split('_lr')
            diag = arm['diagnostics']
            if (diag['optimizer']['name'] != 'Adam'
                    or diag['optimizer']['lr'] != rates[space][int(index)]
                    or arm['steer_phase'] != 'all'
                    or arm['behavior_validated'] is not False):
                raise ValueError('优化或注入配置不符登记')
            fit = tuple(arm['fit_probes'])
            evaluation = tuple(arm['evaluation_probes'])
            norm = lambda values: {' '.join(v.casefold().split()) for v in values}
            if (len(fit) != 20 or len(evaluation) != 10
                    or len(norm(fit)) != 20 or len(norm(evaluation)) != 10
                    or norm(fit) & norm(evaluation)):
                raise ValueError('拟合与留出探针数量或隔离不符')
            if probe_pair is not None and probe_pair != (fit, evaluation):
                raise ValueError('同一经历的各臂探针不一致')
            probe_pair = (fit, evaluation)
            if ([r['step'] for r in arm['history']] != list(range(60))
                    or [r['step'] for r in diag['step_metrics']] != list(range(60))):
                raise ValueError('未完成固定60步')
            for item in arm['history'] + diag['step_metrics']:
                for value in item.values():
                    number(value)
            for metric in [arm['baseline'], arm['final'], diag['fit_baseline'],
                           diag['fit_initial'], diag['fit_final'], diag['heldout_initial']]:
                number(metric['kl'])
            zero = arm['baseline']['kl']
            if baseline is not None and not math.isclose(zero, baseline, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError('同一经历的各臂零注入基线不一致')
            baseline = zero

    spaces = {}
    for space, values in rates.items():
        candidates = []
        for index, lr in enumerate(values):
            name = f'{space}_lr{index}'
            arms = [r['arms'][name] for r in rows]
            candidates.append({
                'arm': name, 'lr': lr,
                'mean_final_heldout_kl': mean(a['final']['kl'] for a in arms),
                'mean_initial_heldout_kl': mean(a['diagnostics']['heldout_initial']['kl'] for a in arms),
                'mean_fit_final_kl': mean(a['diagnostics']['fit_final']['kl'] for a in arms),
                'heldout_below_zero_count': sum(a['final']['kl'] < a['baseline']['kl'] for a in arms),
                'fit_below_initial_count': sum(a['diagnostics']['fit_final']['kl'] < a['diagnostics']['fit_initial']['kl'] for a in arms),
            })
        chosen = min(candidates, key=lambda c: (c['mean_final_heldout_kl'], c['lr']))
        spaces[space] = {'candidates': candidates, 'selected_arm': chosen['arm'], 'selected_lr': chosen['lr']}
    return {'scope': 'dev_optimization_only', 'n_episodes': 10, 'n_fits': 90,
            'selection_rule': 'mean_final_heldout_kl_then_lower_lr',
            'formal_e1_pass': False, 'behavior_validated': False, 'spaces': spaces}
