from unittest.mock import Mock, patch

from pgx_mcts_bench.worker_runtime import worker_init


def test_worker_init_does_not_repeat_torch_interop_configuration() -> None:
    torch = Mock()
    torch.get_num_interop_threads.return_value = 1

    with (
        patch.dict("sys.modules", {"torch": torch}),
        patch("pgx_mcts_bench.worker_runtime.enable_jax_compilation_cache"),
    ):
        worker_init()
        worker_init()

    assert torch.set_num_threads.call_count == 2
    torch.set_num_interop_threads.assert_not_called()
