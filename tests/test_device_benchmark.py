from pgx_mcts_bench.device_benchmark import render


def test_device_report_requires_speed_and_equal_work_cost() -> None:
    payload = {
        "rows": [],
        "decisions": [
            {
                "candidate": "arm",
                "speedup": 4.0,
                "gpu_to_cpu_cost_ratio": 2.0,
                "use_gpu": False,
            }
        ],
    }

    report = render(payload)

    assert "4.00x speedup" in report
    assert "GPU/CPU cost 2.00x" in report
    assert "choose **CPU**" in report
