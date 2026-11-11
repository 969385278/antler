"""只读本地归档和既有报告，导出不含回答、请求、向量的展示数据。"""
from pathlib import Path
import hashlib
import json
import re
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
AUDIT = []


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def table(path, prefix):
    # 只读取指定表格行，不导出报告正文或原始样本。
    text = (ROOT / path).read_text(encoding='utf-8')
    return [[c.strip() for c in line.strip().strip('|').split('|')]
            for line in text.splitlines() if line.startswith(prefix)]


def numbers(text):
    return [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', text.replace('−', '-'))]


def check(label, actual, expected, tolerance=0.0051):
    # 报告优先；数值不一致时留痕，不能悄悄替换结论。
    if abs(actual - expected) > tolerance:
        AUDIT.append({'item': label, 'raw': actual, 'report': expected})


def save(name, rows, source, report, run, note):
    files = [source] if isinstance(source, str) else source
    payload = {'source': files, 'report': report, 'run_id': run, 'note': note,
               'formal_e0_pass': False, 'formal_e1_pass': False, 'data': rows,
               'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in files}}
    (OUT / (name + '.json')).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')


def main():
    OUT.mkdir(exist_ok=True)
    source = 'outputs/main/analysis/2026-10-03T051134.770522_0000_4a1b7c43/curves.json'
    curves = []
    for group in read(source):
        axis = 'bipolar' if 'oea_bipolar' in group['source'] else 'pole'
        for c in group['curves']:
            for coef, score in zip(c['coefs'], c['traits']):
                curves.append({'axis': axis, 'dim': c['dim'], 'pole': c['pole'],
                               'layer': c['layer'], 'coef': coef, 'score': score})
    save('e0_calibration_curves', curves, source, 'docs/RESEARCH_SPEC.md',
         '2026-10-03T051134.770522_0000_4a1b7c43',
         '历史calib，只有O/E/A；双极与单极完整展示。源汇总没有区间，不补造误差线；不能授予当前候选资格。')

    source = 'outputs/analysis/crosstalk_7924314/matrix/matrix.json'
    j = read(source)
    save('e0_crosstalk', {'dims': j['dims'], 'matrix': j['matrix'], 'layer': 16,
                        'coefs': [0.2, 0.2, 0.125, 0.2, 0.185]}, source,
         'docs/RESEARCH_SPEC.md', 'crosstalk_7924314/matrix',
         '历史单极calib；行是测量维度，列是注入维度；数值是相对零注入的人格分变化。原汇总无区间。')

    report = 'docs/E0_ROUND_CLOSEOUT_2026-10-04.md'
    rows = []
    for label in ['原方向', '冻结剥离版', 'mild Prompt', 'moderate Prompt', 'strong Prompt']:
        cells = table(report, '| ' + label + ' |')[0]
        gain, lo, hi = numbers(cells[1])
        rows.append({'condition': label, 'gain': gain, 'ci_low': lo, 'ci_high': hi,
                     'quality_losses': [numbers(x)[0] for x in cells[2:]]})
    save('e0_tradeoff', rows, report, report, 'd09a117a119541a2985f54a77c8e4391',
         '报告同期共同zero；第21层decode-only、C−0.225。增益带95%区间；质量表无区间。Prompt强度未匹配，Strict/Practical未通过。')

    source = 'outputs/analysis/e1_dev_terminal_20261004/run/results.json'
    selection_path = 'outputs/analysis/e1_dev_terminal_20261004/selection.json'
    selection = read(selection_path)
    rows = []
    expected = {'personality': 11.40, 'random': 2.35, 'full': 36.34}
    for space, s in selection['spaces'].items():
        means = {c['arm']: statistics.mean(e['arms'][c['arm']]['final']['kl']
                 for e in read(source)['episodes']) for c in s['candidates']}
        chosen = min(s['candidates'], key=lambda c: (means[c['arm']], c['lr']))['arm']
        assert chosen == s['selected_arm'], '选参和封存记录不一致'
        values = []
        for e in read(source)['episodes']:
            arm = e['arms'][chosen]
            value = 100 * (1 - arm['final']['kl'] / arm['baseline']['kl'])
            check(e['id'] + '/' + space, value, 100 * arm['explained'], 1e-7)
            values.append(value)
            rows.append({'episode': e['id'], 'space': space, 'arm': chosen,
                         'explained_pct': value})
        check(space + '/mean_pct', statistics.mean(values), expected[space])
    save('e1_kl_explained', {'rows': rows, 'report_means_pct': expected},
         [source, selection_path], 'docs/E1_PROGRESS_AND_ISSUES.md',
         '31f70d1cb5874f7f9b3791f99eaa6d6a',
         '10段dev、每空间三学习率、固定单随机基。逐经历1−KL_final/KL_zero再平均；不是记忆保留率，全维不是理论上限。若均值与报告不符则不展示该空间分布。')

    source = 'outputs/analysis/e1_twin_context_gate_20261005/summary_final.json'
    report = 'docs/E1_TWIN_CONTEXT_GATE_RESULTS_2026-10-05.md'
    raw = {r['scenario_id']: r for r in read(source)['reports']}
    rows = []
    for cells in table(report, '| twin'):
        scenario, dim, high, low, category = cells
        for pole, cell in [('high', high), ('low', low)]:
            mean, lo, hi = numbers(cell)
            r = next(p for p in raw[scenario]['poles'] if p['pole'] == pole)
            stats = r['metrics']['trait']['interaction_high_axis']
            for name, a, e in zip(['mean', 'lo', 'hi'],
                                  [stats['mean'], *stats['question_bootstrap_ci']], [mean, lo, hi]):
                check(scenario + '/' + pole + '/' + name, a, e)
            rows.append({'scenario': scenario, 'dim': dim, 'pole': pole, 'mean': mean,
                         'ci_low': lo, 'ci_high': hi, 'category': category})
    save('e1_twin_gate_forest', rows, [source, report], report,
         '9c008133e6e94c7d91b0ff2d5a0af44f',
         'I=(D+−D−)/2；低极统一到高极方向；十题配对bootstrap 95%区间，非跨情境区间。1已建立、2部分证据、3未建立。')

    source = ('outputs/analysis/e1_twin_context_gate_20261005/outputs/main/'
              'e1_twin_state_extract/03e4e1c7f1d94fe0a31c87e623abf356/summary.json')
    report = 'docs/E1_TWIN_STATE_EXTRACTION_RESULTS_2026-10-05.md'
    raw = read(source)['groups']['all60']
    labels = ['正行动的好−坏 d+', '负行动的好−坏 d−',
              '行动敏感交互 i=(d+−d−)/2', '共同行动反馈 v=(d++d−)/2']
    rows = []
    for label, key in zip(labels, ['d_plus', 'd_minus', 'interaction', 'valence']):
        cells = table(report, '| ' + label + ' |')[0]
        full, rebuilt = float(cells[1]), float(cells[2])
        check(key + '/full', raw[key]['full_norm_s'], full, 0.00000051)
        check(key + '/rebuilt', raw[key]['reconstructed_norm_s'], rebuilt, 0.00000051)
        rows.append({'quantity': key, 'label': label, 'full_norm': full,
                     'reconstructed_norm': rebuilt})
    check('interaction_squared_norm_pct',
          100*(raw['interaction']['reconstructed_norm_s']/raw['interaction']['full_norm_s'])**2,
          0.5499, 0.000051)
    a, b = raw['d_plus']['full_state'], raw['d_minus']['full_state']
    cosine = sum(x*y for x, y in zip(a, b)) / (
        sum(x*x for x in a) * sum(y*y for y in b))**0.5
    check('action_cosine', cosine, 0.9680, 0.000051)
    save('e1_state_geometry', {'rows': rows, 'action_cosine': 0.9680,
                             'interaction_squared_norm_pct': 0.5499},
         [source, report], report, '03e4e1c7f1d94fe0a31c87e623abf356',
         '第16层最后prompt-token、60探针；范数以s16归一化。五维为非正交基重建的物理范数；无注入。平方范数比例不是KL解释率、解释方差或记忆保留率。')

    source = 'outputs/analysis/e1_twin_context_gate_20261005/l21_pilot_a351909f.tar.gz'
    report = 'docs/E1_L21_SMALL_PILOT_RESULTS_2026-10-05.md'
    with tarfile.open(ROOT / source) as archive:
        member = next(m for m in archive if m.name.endswith('/summary.json')
                      and '/source_snapshot/' not in m.name)
        raw = json.load(archive.extractfile(member))
    rows = []
    formats = {'full': 'full_state', '五维': 'reconstructed_state', '纯反馈': 'feedback_only'}
    for cells in table(report, '| 高极/') + table(report, '| 低极/'):
        pole_cn, fmt = cells[0].split('/')
        if fmt not in formats:
            continue
        pole = 'high' if pole_cn == '高极' else 'low'
        row = {'pole': pole, 'format': fmt}
        for metric, cell in zip(['mismatch_minus_match', 'random_minus_match'], cells[1:]):
            mean, lo, hi = numbers(cell)
            row[metric] = {'mean': mean, 'ci_low': lo, 'ci_high': hi}
        # 配对差值先在每题内平均四故事；区间沿用报告，不重新选择bootstrap设置。
        candidates = [r for r in raw['reports'] if r['pole'] == pole]
        match = next(r for r in candidates if r['format'] == formats[fmt])
        stats = [c['metrics']['trait']['mismatch_minus_match_error']['per_question']
                 for c in match['cells']]
        check(pole + '/' + fmt + '/mismatch_mean',
              statistics.mean(statistics.mean(x) for x in zip(*stats)),
              row['mismatch_minus_match']['mean'])
        random = next(r for r in candidates if r['format'] == 'random_norm')
        random_errors = statistics.mean(c['metrics']['trait']['matched_error']['mean']
                                        for c in random['cells'])
        match_errors = statistics.mean(c['metrics']['trait']['matched_error']['mean']
                                       for c in match['cells'])
        check(pole + '/' + fmt + '/random_mean', random_errors - match_errors,
              row['random_minus_match']['mean'])
        rows.append(row)
    save('e1_l21_pilot', rows, [source, report], report,
         'a351909f6f394034b5402398282a9f0c',
         '第21层decode原幅度；单情境、五题、seed0。区间仅覆盖五题；随机仅与full同范数。full低极任务损失14分超限，正式E1未通过。')

    (OUT / 'extraction_audit.json').write_text(json.dumps(
        {'discrepancies': AUDIT, 'policy': '报告优先；差异不更改实验结论'},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('已导出七组展示数据；与报告的非舍入差异：', len(AUDIT))


if __name__ == '__main__':
    main()
