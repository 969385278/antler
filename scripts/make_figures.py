"""仅从results读取数值，重新生成七张学术展示图（不访问outputs或模型）。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
BLUE, ORANGE, GREEN, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
GRAY = '#697580'


def load(name):
    return json.loads((RESULTS / (name + '.json')).read_text(encoding='utf-8'))['data']


def setup():
    # 字体缺失时明确报错，避免静默生成乱码。
    font_manager.findfont('Microsoft YaHei', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 10,
                         'axes.titlesize': 12, 'axes.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#AAB2B9', 'axes.linewidth': 0.6,
                         'grid.color': '#DBE1E6', 'grid.linewidth': 0.5,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path',
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white',
                         'svg.hashsalt': 'antler-showcase'})
    FIGURES.mkdir(exist_ok=True)


def finish(fig, name, title, subtitle, footnote, bottom=0.20, left=0.10):
    # 固定宽8英寸×200dpi=1600px；为标题和边界说明预留独立区域。
    fig.suptitle(title, x=0.055, y=0.975, ha='left', fontsize=16, weight='bold')
    fig.text(0.055, 0.92, subtitle, ha='left', va='top', fontsize=9, color=GRAY)
    fig.text(0.055, 0.035, footnote, ha='left', va='bottom', fontsize=8, color=GRAY,
             linespacing=1.65)
    fig.subplots_adjust(left=left, right=0.96, top=0.82, bottom=bottom,
                        wspace=0.32, hspace=0.46)
    if name == 'e0_crosstalk':
        # 色条使用独立位置，避免后续边距调整把色条标签挤出画布。
        fig.axes[0].set_position([0.20, 0.20, 0.49, 0.49*8/6.5])
        fig.axes[1].set_position([0.74, 0.20, 0.025, 0.49*8/6.5])
    fig.savefig(FIGURES / (name + '.png'), dpi=200,
                metadata={'Software': 'Antler showcase / matplotlib'})
    svg_path = FIGURES / (name + '.svg')
    fig.savefig(svg_path, metadata={'Date': None})
    # matplotlib路径行默认带尾空格；统一UTF-8/LF便于仓库审阅。
    svg_path.write_text('\n'.join(line.rstrip() for line in
                                  svg_path.read_text(encoding='utf-8').splitlines()) + '\n',
                        encoding='utf-8', newline='\n')
    plt.close(fig)


def calibration():
    name = 'e0_calibration_curves'
    rows = load(name)
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.8), sharex=True, sharey=True)
    labels = {'O': '开放性 O', 'E': '外向性 E', 'A': '宜人性 A'}
    for i, axis in enumerate(['bipolar', 'pole']):
        for j, dim in enumerate(['O', 'E', 'A']):
            ax = axes[i, j]
            for pole, color, marker, label in [('high', BLUE, 'o', '高极量表'),
                                                ('low', ORANGE, 's', '低极量表')]:
                data = sorted((r for r in rows if r['axis'] == axis and r['dim'] == dim
                               and r['pole'] == pole), key=lambda r: r['coef'])
                ax.plot([r['coef'] for r in data], [r['score'] for r in data],
                        color=color, marker=marker, ms=4, lw=1.5, label=label)
            ax.set_title(labels[dim] + (' · 双极轴' if i == 0 else ' · 单极轴'), fontsize=10)
            ax.set_ylim(-2, 103)
            ax.set_xticks([-0.4, 0, 0.4])
            ax.axvline(0, color=GRAY, lw=0.6, ls=':')
            ax.grid(axis='y')
            if j == 0:
                ax.set_ylabel('人格分（0–100）')
            if i == 1:
                ax.set_xlabel('注入系数（s16）')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='center right')
    finish(fig, name, 'E0 | 人格读数随注入系数系统变化',
           '第16层 · 历史 calib · 两种轴定义、两套极性量表分别展示',
           '第16层 calib 题；高极与低极为独立量表，不能直接相加。\n'
           '来源汇总只含 O/E/A 三个维度，未提供区间。')


def crosstalk():
    name = 'e0_crosstalk'
    data = load(name)
    m = np.array(data['matrix'])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(m, cmap='PuOr_r', norm=TwoSlopeNorm(0, -30, 30))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{m[i,j]:+.2f}', ha='center', va='center', fontsize=11,
                    color='white' if abs(m[i,j]) > 20 else '#1F2933')
    ax.set_xticks(range(5), data['dims'])
    ax.set_yticks(range(5), data['dims'])
    ax.set_xlabel('注入维度（列）')
    ax.set_ylabel('测量维度（行）')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label='人格分变化（相对零注入）')
    finish(fig, name, 'E0 | 大五方向之间存在行为串扰',
           '第16层 · 历史单极方向 · calib · 每个单元格保留原始符号',
           'O/C/E/A/N 注入系数依次为 +0.20 / +0.20 / +0.125 / +0.20 / +0.185；\n'
           '行是测量维度，列是注入维度；数值为人格分变化（相对零注入）。\n'
           '量表端点可能掩盖部分串扰；来源汇总未提供区间。')


def tradeoff():
    name = 'e0_tradeoff'
    rows = load(name)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [1, 1.3]})
    y = np.arange(len(rows))
    gains = np.array([r['gain'] for r in rows])
    a.barh(y, gains, color=[BLUE, GREEN, ORANGE, ORANGE, ORANGE], alpha=0.85, height=0.5)
    a.errorbar(gains, y, xerr=[gains-[r['ci_low'] for r in rows],
                              np.array([r['ci_high'] for r in rows])-gains],
               fmt='none', color='#243746', capsize=3, lw=1)
    a.set_yticks(y, [r['condition'].replace(' Prompt', '\nPrompt') for r in rows], fontsize=9)
    a.set_xlim(0, 110)
    a.set_title('人格增益与 95% CI')
    a.set_xlabel('careless 分增益')
    for i, v in enumerate(gains):
        a.text(2, i, f'{v:.2f}', va='center', fontsize=8, color='white')
    for k, (label, color, marker) in enumerate(zip(
            ['高极连贯', '高极任务', '低极连贯', '低极任务'],
            [BLUE, ORANGE, GREEN, PURPLE], ['o', 's', '^', 'D'])):
        b.scatter([r['quality_losses'][k] for r in rows], y+(k-1.5)*0.13,
                  color=color, marker=marker, s=27, label=label)
    b.axvline(10, color=GRAY, ls='--', lw=1)
    b.axvline(0, color=GRAY, lw=0.6)
    b.set_xlim(-6, 46)
    b.set_yticks(y, [])
    b.set_title('四项质量损失（点估计）')
    b.set_xlabel('zero − 条件分；虚线为 10 分门槛')
    b.legend(frameon=False, ncol=2, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15))
    for ax in [a, b]:
        ax.set_ylim(4.6, -0.6)
        ax.grid(axis='x')
        ax.set_axisbelow(True)
    finish(fig, name, 'E0 | 单层注入的人格增益伴随质量代价',
           '第21层 · C−0.225 · decode-only · 同期共同 zero；不拼接旧轮分数',
           '第21层、仅生成阶段注入；误差线为人格增益的 95% CI。\n'
           '剥离版四项质量损失均超过预设的 10 分门槛。\n'
           'Prompt 对照与注入的人格强度未能匹配，仅作描述性比较。', bottom=0.31, left=0.17)
    # 长条件名为左侧单独预留空间。


def kl():
    name = 'e1_kl_explained'
    data = load(name)
    fig, ax = plt.subplots(figsize=(8, 5.7))
    for i, (space, color) in enumerate(zip(['personality', 'random', 'full'], [BLUE, GRAY, GREEN])):
        rows = [r for r in data['rows'] if r['space'] == space]
        vals = np.array([r['explained_pct'] for r in rows])
        mean = data['report_means_pct'][space]
        # 均值偏离报告时只画报告均值，不发布矛盾的逐经历分布。
        if abs(vals.mean()-mean) <= 0.0051:
            offsets = np.linspace(-0.14, 0.14, len(vals))
            ax.scatter(i+offsets, vals, color=color, s=40, alpha=0.72)
        ax.plot([i-0.23, i+0.23], [mean, mean], color=color, lw=3)
        ax.text(i, 1.025, f'均值 {mean:.2f}%', ha='center', weight='bold', color=color,
                transform=ax.get_xaxis_transform())
    ax.axhline(0, color=GRAY, ls='--', lw=0.8)
    ax.set_xticks(range(3), ['人格五维', '固定随机五维', '全维'])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-5, 72)
    ax.set_ylabel('KL 缩减比例（%）')
    ax.grid(axis='y')
    finish(fig, name, 'E1 | 大五子空间比随机子空间保留更多经历效应',
           '第16层 · 10 段 dev · 90 条件按登记选参；点为逐经历值，横线为报告均值',
           '10 段开发集经历，逐经历计算 1 − KL_final / KL_zero 后平均；单一随机基，仅作描述。\n'
           '该比例衡量输出分布差距的缩小，不是人格分或记忆保留率；\n'
           '全维为有限优化下的对照，不是理论上限。')


def gate():
    name = 'e1_twin_gate_forest'
    rows = load(name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6.1), sharey=True)
    scenarios = list(dict.fromkeys(r['scenario'] for r in rows))
    colors = {'已建立': GREEN, '部分证据': BLUE, '未建立': GRAY}
    for ax, pole, title in zip(axes, ['high', 'low'], ['高极量表', '低极量表（已统一方向）']):
        for i, scenario in enumerate(scenarios):
            r = next(r for r in rows if r['scenario'] == scenario and r['pole'] == pole)
            ax.errorbar(r['mean'], i, xerr=[[r['mean']-r['ci_low']], [r['ci_high']-r['mean']]],
                        fmt='o', color=colors[r['category']], capsize=3, ms=5, lw=1.3)
            ax.annotate(f"{r['mean']:.2f}", (r['mean'], i), xytext=(0, -16),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.axvline(0, color=GRAY, lw=0.8, ls='--')
        ax.set_xlim(-5, 22)
        ax.set_ylim(5.55, -0.55)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('行动 × 反馈交互 I（分）')
        ax.grid(axis='x')
    labels = []
    for s in scenarios:
        r = next(r for r in rows if r['scenario'] == s)
        labels.append(f"{s} ({r['dim']})\n{r['category']}")
    axes[0].set_yticks(range(6), labels, fontsize=8)
    finish(fig, name, 'E1 | 结局对人格的影响依赖具体情境',
           '1 个已建立 · 2 个部分证据 · 3 个未建立；分类沿用原报告',
           'I = (D+ − D−) / 2；误差线为十题配对 bootstrap 95% CI，不是跨情境区间。\n'
           '分类沿用原报告：1 个已建立、2 个部分证据、3 个未建立。', left=0.27)
    # 预留情境与分类两行标签。


def geometry():
    name = 'e1_state_geometry'
    data = load(name)
    rows = data['rows']
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(4)
    for offset, key, label, color in [(-0.18, 'full_norm', '全维范数', BLUE),
                                       (0.18, 'reconstructed_norm', '五维重建范数', ORANGE)]:
        vals = [r[key] for r in rows]
        bars = ax.bar(x+offset, vals, width=0.33, label=label, color=color, zorder=3)
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.6f}', (bar.get_x()+bar.get_width()/2, v),
                        xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xticks(x, ['正行动好−坏\nd+', '负行动好−坏\nd−', '行动敏感交互\ni = (d+ − d−)/2',
                      '共有反馈\nv = (d+ + d−)/2'], fontsize=9)
    ax.set_ylabel('物理范数 / 层尺度 s16')
    ax.set_ylim(0, 0.043)
    ax.ticklabel_format(axis='y', style='plain')
    ax.grid(axis='y', zorder=0)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    finish(fig, name, 'E1 | 经历差值主要落在共同的“好/坏反馈”方向上',
           '第16层 · 最后 prompt-token · 60 探针 · 固定情境 twin04_ideas · 未做行为注入',
           'd+ 与 d− 余弦 0.968；交互分量的五维重建只保留其平方范数的 0.55%。\n'
           '五维为非正交基重建的物理范数；0.55% 是投影能量比例，不是 KL 解释率或记忆保留率。')


def pilot():
    name = 'e1_l21_pilot'
    rows = load(name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    labels = [('high', 'full'), ('low', 'full'), ('high', '五维'), ('low', '五维'),
              ('high', '纯反馈'), ('low', '纯反馈')]
    for ax, key, title in zip(axes, ['mismatch_minus_match', 'random_minus_match'],
                              ['错配误差 − 匹配误差', '随机误差 − 匹配误差']):
        for i, (pole, fmt) in enumerate(labels):
            r = next(r for r in rows if r['pole'] == pole and r['format'] == fmt)[key]
            color = BLUE if pole == 'high' else ORANGE
            ax.errorbar(r['mean'], i, xerr=[[r['mean']-r['ci_low']], [r['ci_high']-r['mean']]],
                        fmt='o' if pole == 'high' else 's', color=color, capsize=3, lw=1.3)
            ax.annotate(f"{r['mean']:.2f}", (r['mean'], i), xytext=(0, -16),
                        textcoords='offset points', ha='center', fontsize=8)
        ax.axvline(0, color=GRAY, ls='--', lw=0.8)
        ax.set_xlim(-12, 23)
        ax.set_ylim(5.55, -0.55)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('误差差值（正值：匹配更近）')
        ax.grid(axis='x')
    axes[0].set_yticks(range(6), [f"{f} · {'高极' if p == 'high' else '低极'}" for p, f in labels], fontsize=9)
    finish(fig, name, 'E1 | 原幅度写入：匹配经历优于随机，与错配的差距较小',
           '第21层 · decode 原幅度 · 单情境、五题、seed0 · 原报告 95% 描述区间',
           '第21层、单情境、五题、seed0；区间为五题描述区间，不代表跨经历泛化。\n'
           '随机向量只与 full 同范数。纯反馈向量也有匹配优势，\n'
           '因此不能排除“只保存了共同反馈”的解释。', left=0.17)


def main():
    setup()
    for draw in [calibration, crosstalk, tradeoff, kl, gate, geometry, pilot]:
        draw()
    print('已生成七张图，各含1600px PNG及SVG。')


if __name__ == '__main__':
    main()
