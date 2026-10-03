"""C19：验证单极符号、尺度及真实微型模型上的注入。"""
import json
import sys

import pytest
import torch

from test_core import model
from test_e0_calibrate import calibration
from antler.core.hooks import Steer
from antler.traits.vectors import steer_vector_poles


def test_negative_selects_low_without_negating_and_normalizes():
    poles = {"dependable": torch.tensor([3., 0., 0.]), "careless": torch.tensor([0., 4., 0.]),
             "nervous": torch.tensor([0., 0., 5.])}
    z = torch.tensor([0., -.2, 0., 0., .25])
    assert torch.allclose(steer_vector_poles(poles, 2., z), torch.tensor([0., .4, .5]))
    z[1] = .2
    assert torch.allclose(steer_vector_poles(poles, 2., z), torch.tensor([.4, 0., .5]))
    assert torch.equal(steer_vector_poles(poles, 2., torch.zeros(5)), torch.zeros(3))


@pytest.mark.parametrize("vector", [torch.zeros(3), torch.tensor([float('nan'), 1., 0.]), torch.ones(4)])
def test_invalid_active_vector_rejected(vector):
    with pytest.raises(ValueError):
        steer_vector_poles({"dependable": torch.ones(3), "careless": vector}, 1.,
                           torch.tensor([0., -.2, 0., 0., 0.]))


def test_cli_poles_real_hook_and_provenance(calibration, model, monkeypatch, tmp_path):
    cfg = {"out_path": tmp_path, "layers": [1], "coefs": [-.2, 0., .2], "max_new_tokens": 8}
    size = model.config.hidden_size
    U = torch.eye(5, size).repeat(4, 1, 1)
    monkeypatch.setattr(calibration, "setup", lambda *a, **kw: (cfg, model, None, object()))
    monkeypatch.setattr(calibration, "load_directions", lambda p: {"U": U, "s": torch.full((4,), 2.)})
    (tmp_path / "vectors").mkdir()
    for name, index in [("dependable", 2), ("careless", 3)]:
        torch.save(torch.eye(size)[index].repeat(4, 1) * 7, tmp_path / "vectors" / f"{name}.pt")
    seen = []
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        baseline = model(input_ids=ids).logits

    def measure(*args, steer, **kw):
        seen.append(steer)
        with torch.no_grad():
            if steer:
                with Steer(model, steer):
                    assert not torch.allclose(model(input_ids=ids).logits, baseline)
            assert torch.equal(model(input_ids=ids).logits, baseline)
        return {"trait": 50., "coherence": 90., "samples": []}

    monkeypatch.setattr(calibration, "trait_score", measure)
    monkeypatch.setattr(sys, "argv", ["e0_calibrate.py", "--axis", "pole", "--dims", "C", "--save-samples"])
    calibration.main()
    assert torch.allclose(seen[0][1], torch.eye(size)[3] * .4)
    assert seen[1] is None
    assert torch.allclose(seen[2][1], torch.eye(size)[2] * .4)
    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["axis"] == "pole" and meta["status"] == "completed"
    assert sum(p.endswith('.pt') for p in meta['data_hashes']) == 2
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(not layer._forward_hooks for layer in model.model.layers)
