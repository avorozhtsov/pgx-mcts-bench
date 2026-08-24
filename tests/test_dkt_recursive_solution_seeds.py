import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/build_dkt_recursive_solution_seeds.py"
    spec = importlib.util.spec_from_file_location("dkt_recursive_solution_seeds", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_10_129_tail_is_exactly_certified():
    payload = load_builder().build()
    chain = payload["chains"][0]
    tail = chain["edges"][1]
    cert = tail["certificate"]
    assert chain["reported_upper_bound"] == 2
    assert tail["from"] == "10_129"
    assert cert["crossing_position_1_based"] == 1
    assert cert["tietze_reduction"]["reduces_to_infinite_cyclic"] is True
    assert cert["tietze_reduction"]["remaining_relators"] == []
    assert len(cert["tietze_reduction"]["remaining_generators"]) == 1
    assert len(cert["tietze_reduction"]["elimination_steps"]) == 9


def test_chain_does_not_overclaim_machine_replayability():
    chain = load_builder().build()["chains"][0]
    assert chain["edges"][0]["certificate_level"] == "paper-figure-only"
    assert chain["edges"][0]["machine_replayable"] is False
    assert chain["end_to_end_machine_replayable"] is False
