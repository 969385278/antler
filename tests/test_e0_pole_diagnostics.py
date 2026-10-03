"""C19：离线检查批处理参数、来源保护和失败留档。"""
import argparse
import importlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from antler.traits.data import BIG_FIVE, DIMS
from antler.traits.vectors import build_directions


@pytest.mark.parametrize('fail', [False, True])
def test_isolated_batch_and_failure(monkeypatch, tmp_path, fail):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'experiments'))
    module = importlib.import_module('e0_pole_diagnostics')
    (tmp_path / 'vectors').mkdir()
    poles = {}
    for i, pole in enumerate(p for k in DIMS for p in BIG_FIVE[k]):
        poles[pole] = torch.eye(10)[i].repeat(17, 1)
        torch.save(poles[pole], tmp_path / 'vectors' / f'{pole}.pt')
    torch.save({'U': build_directions(poles), 's': torch.ones(17), 'dims': DIMS}, tmp_path / 'directions.pt')
    cfg = {'out_path': tmp_path, 'calib_questions': 10}
    monkeypatch.setattr(module, 'setup', lambda *a, **kw: (cfg, None, None, None))
    before = module.sha256(tmp_path / 'directions.pt')
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        assert command[command.index('--pole') + 1] == 'both'
        assert command[command.index('--n-samples') + 1] == '3'
        assert command[command.index('--max-new-tokens') + 1] == '320'
        assert command[command.index('--layers') + 1] == '16'
        if fail:
            raise subprocess.CalledProcessError(1, command)
        folder = Path(command[command.index('--config') + 1]).parent
        (folder / 'run_meta.json').write_text('{"status":"completed"}', encoding='utf-8')

    monkeypatch.setattr(module.subprocess, 'run', execute)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            module.run(argparse.Namespace(config='unused'))
    else:
        module.run(argparse.Namespace(config='unused'))
    out = next((tmp_path / 'pole_diagnostics').iterdir())
    meta = json.loads((out / 'run_meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == ('failed' if fail else 'completed')
    assert len(calls) == (1 if fail else 4)
    assert (out / 'results.zip').exists()
    assert module.sha256(tmp_path / 'directions.pt') == before
    assert not (tmp_path / 'run_meta.json').exists()
