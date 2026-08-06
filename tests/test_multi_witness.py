from types import SimpleNamespace

from pgx_mcts_bench.multi_witness import _source_round, _stratified


def test_source_round_decodes_native_and_translated_episode_seeds() -> None:
    seed = 20_260_810
    assert (
        _source_round(
            seed + 500_000_000 + 17_000_000 + 20_000,
            seed=seed,
            scientist_index=2,
        )
        == 17
    )
    assert (
        _source_round(
            seed + 900_000_000 + 23 * 10_000 + 2,
            seed=seed,
            scientist_index=2,
        )
        == 23
    )


def test_stratified_panel_round_robins_structure_buckets() -> None:
    def item(identity: str, length: int, strands: int):
        return SimpleNamespace(
            id=identity,
            knot=SimpleNamespace(word=tuple(range(length)), strands=strands),
        )

    items = [
        item("a", 8, 3),
        item("b", 8, 3),
        item("c", 12, 3),
        item("d", 12, 4),
    ]
    assert [row.id for row in _stratified(items, 3)] == ["a", "c", "d"]
