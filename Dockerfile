# Ladder training, self-contained.
#
# Both repos go in side by side because `pgx-mcts-bench` depends on `rf-knots` by
# path (`[tool.uv.sources] rf-knots = { path = "../rf-knots" }`), so an image
# holding only one of them cannot construct the environment at all.
FROM python:3.12-slim

# All project and third-party dependencies used by this image publish Python
# wheels. Avoiding a compiler toolchain keeps cross-platform CUDA image builds
# small and, in particular, avoids emulating an entire Debian C/C++ install
# when an arm64 developer machine builds the amd64 Nebius image.

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /work
COPY rf-knots /work/rf-knots
COPY pgx-mcts-bench /work/pgx-mcts-bench

ENV VIRTUAL_ENV=/work/venv
ENV PATH="/work/venv/bin:$PATH"
RUN uv venv --python 3.12 /work/venv

# CPU remains the default. A GPU decision image is built from the same file with
# `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124`; installing
# torch first means the project resolve keeps the deliberately selected wheel.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv pip install torch --index-url ${TORCH_INDEX_URL}

RUN uv pip install -e /work/rf-knots \
 && uv pip install -e /work/pgx-mcts-bench

WORKDIR /work/pgx-mcts-bench
# Artifacts are a bind mount, not a layer: checkpoints have to outlive the
# container, and a run that loses its ladder on restart cannot be resumed --
# which is the point of resuming by rung identity.
VOLUME ["/work/pgx-mcts-bench/artifacts"]

ENTRYPOINT ["pgx-mcts-bench"]
CMD ["--help"]
