"""C11：归档表示分析的CPU合成测试，无模型下载或评分。"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    return importlib.import_module('e1_subspace_pilot')


def test_known_geometry_and_change_of_basis(module):
    u = torch.eye(5, 8, dtype=torch.float64)
    x = torch.zeros(3, 8, dtype=torch.float64)
    x[0, 0], x[1, 7], x[2, 0], x[2, 7] = 2, 2, 2, 2
    values, mean = module.fractions(x, module.basis(u))
    assert values == pytest.approx([1, 0, .5])
    assert mean == pytest.approx(.5)
    u[1] += 2 * u[0]
    assert module.fractions(x, module.basis(u))[0] == pytest.approx(values)


def test_random_reproducibility_and_centering(module):
    x = torch.randn(12, 20, generator=torch.Generator().manual_seed(9))
    u = torch.eye(5, 20)
    first = module.analyze(x, u, 20)
    assert first == module.analyze(x, u, 20)
    assert first != module.analyze(x, u, 20, seed=1)
    assert first['formal_e1_pass'] is False
    for group in first['variants'].values():
        assert len(group['random_macro_means']) == 20
        assert 0 <= group['macro_mean'] <= 1
        assert 1 / 21 <= group['random_tail_fraction'] <= 1


def test_zero_nan_and_rank_failures(module):
    u = torch.eye(5, 8)
    with pytest.raises(ValueError, match='秩不足'):
        module.basis(torch.zeros(5, 8))
    with pytest.raises(ValueError, match='均为零'):
        module.analyze(torch.zeros(12, 8), u, 2)
    with pytest.raises(ValueError, match='非有限'):
        module.analyze(torch.full((12, 8), float('nan')), u, 2)
    x = torch.zeros(2, 8, dtype=torch.float64)
    x[1, 0] = 1
    assert module.fractions(x, module.basis(u)) == ([None, 1.0], 1.0)


def test_run_hash_guard_and_failure_archive(module, tmp_path, monkeypatch):
    directions, activations = tmp_path / 'd.pt', tmp_path / 'a.pt'
    torch.save({'dims': ['O', 'C', 'E', 'A', 'N'], 'U': torch.eye(5, 8).repeat(17, 1, 1)}, directions)
    torch.save({'layer': 16, 'episode_ids': [f'p{i:02}' for i in range(1, 13)],
                'mean_differences': torch.randn(12, 8, generator=torch.Generator().manual_seed(3))}, activations)
    monkeypatch.setattr(module, 'DIRECTION_HASH', module.sha256(directions))
    monkeypatch.setattr(module, 'ACTIVATION_HASH', module.sha256(activations))
    def setup(args, need_judge, need_model):
        assert need_judge is False and need_model is False
        return {'out_path': tmp_path}, None, None, None
    monkeypatch.setattr(module, 'setup', setup)
    analyze = module.analyze
    monkeypatch.setattr(module, 'analyze', lambda x, u: analyze(x, u, 2))
    args = SimpleNamespace(config='configs/debug.yaml', directions=str(directions), activations=str(activations))
    out = module.run(args)
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'completed'
    assert meta['results_sha256'] == module.sha256(out / 'results.json')
    activations.write_bytes(b'changed')
    with pytest.raises(ValueError, match='哈希'):
        module.run(args)
    metas = [json.loads(p.read_text(encoding='utf-8')) for p in tmp_path.glob('e1_subspace_pilot/*/run_meta.json')]
    assert sorted(m['status'] for m in metas) == ['completed', 'failed']
