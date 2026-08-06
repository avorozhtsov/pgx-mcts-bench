from pgx_mcts_bench.native_learning_gate import admission_decision, analyze_native_learning


def panel(solved, capped):
    return {"solved": sorted(solved), "capped_objective": capped}


def test_admission_requires_retention_and_observable_improvement():
    before = panel({"canary"}, 100.0)
    accepted = admission_decision(
        before, panel({"canary", "target"}, 90.0), target="target"
    )
    lost = admission_decision(before, panel({"target"}, 80.0), target="target")
    unchanged = admission_decision(before, panel({"canary"}, 100.0), target="target")
    assert accepted["passed"]
    assert not lost["passed"] and lost["lost"] == ["canary"]
    assert not unchanged["passed"]


def test_gate_requires_two_distinct_replicated_rescues_and_exact_retention():
    rows = [
        {
            "gained": ["10_149", "12a_1168"],
            "lost": [],
            "initial": panel([], 100),
            "final": panel([], 90),
        },
        {
            "gained": ["10_149", "12a_1168"],
            "lost": [],
            "initial": panel([], 100),
            "final": panel([], 95),
        },
        {"gained": [], "lost": [], "initial": panel([], 100), "final": panel([], 100)},
    ]
    report = analyze_native_learning(rows)
    assert report["decision"]["passed"]
    assert report["replicated_rescues"] == ["10_149", "12a_1168"]


def test_gate_rejects_any_seed_with_a_lost_identity():
    rows = [
        {
            "gained": ["10_149", "12a_1168"],
            "lost": ["canary"],
            "initial": panel([], 100),
            "final": panel([], 90),
        },
        {
            "gained": ["10_149", "12a_1168"],
            "lost": [],
            "initial": panel([], 100),
            "final": panel([], 90),
        },
    ]
    assert not analyze_native_learning(rows)["decision"]["passed"]
