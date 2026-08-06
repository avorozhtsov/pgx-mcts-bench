import json
from pathlib import Path

import torch

from pgx_mcts_bench.solve_calibration import fit_solve_calibration


def test_fit_solve_calibration_changes_metadata_not_network(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save({"network": {"weight": torch.tensor([3.0])}}, source)
    validation = tmp_path / "validation.json"
    attempts = [
        {"p_solve": score, "solved": label}
        for score, label in ((0.9, False), (0.8, True), (0.7, False), (0.6, True))
    ]
    validation.write_text(
        json.dumps(
            {
                "scientist": "s-window-128",
                "protocol": {"simulations": 64},
                "trained": {"attempts": attempts},
            }
        )
    )
    output = tmp_path / "calibrated.pt"
    report_path = tmp_path / "report.json"

    report = fit_solve_calibration(source, validation, output, report_path)
    payload = torch.load(output, map_location="cpu", weights_only=False)

    torch.testing.assert_close(payload["network"]["weight"], torch.tensor([3.0]))
    assert payload["solve_calibration"]["scale"] > 0.0
    assert report["network_weights_changed"] is False
    assert report_path.exists()
