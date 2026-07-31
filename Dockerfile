# Ladder training, self-contained.
#
# Both repos go in side by side because `pgx-mcts-bench` depends on `rf-knots` by
# path (`[tool.uv.sources] rf-knots = { path = "../rf-knots" }`), so an image
# holding only one of them cannot construct the environment at all.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /work
COPY rf-knots /work/rf-knots
COPY pgx-mcts-bench /work/pgx-mcts-bench

ENV VIRTUAL_ENV=/work/venv
ENV PATH="/work/venv/bin:$PATH"
RUN uv venv --python 3.12 /work/venv

# CPU torch, installed *first* and from PyTorch's own index. On Linux the default
# wheel is the CUDA build, which drags in ~2.5 GB of nvidia-cusolver, cufft and
# nccl for a machine with no GPU. Installing it up front means the project's own
# resolve finds `torch>=2.3` already satisfied and never reaches for the CUDA one.
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install -e /work/rf-knots \
 && uv pip install -e /work/pgx-mcts-bench

WORKDIR /work/pgx-mcts-bench
# Artifacts are a bind mount, not a layer: checkpoints have to outlive the
# container, and a run that loses its ladder on restart cannot be resumed --
# which is the point of resuming by rung identity.
VOLUME ["/work/pgx-mcts-bench/artifacts"]

ENTRYPOINT ["pgx-mcts-bench"]
CMD ["--help"]
