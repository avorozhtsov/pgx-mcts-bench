# Ladder training, self-contained.
#
# Both repos go in side by side because `pgx-mcts-bench` depends on `rf-knots` by
# path (`../rf-knots`), so a container holding only one of them cannot resolve the
# environment.
#
# The source is copied rather than cloned: `serial-fixes` has no upstream, so a
# clone from GitHub would build an image without any of the work it is meant to
# run. Copying also means nothing has to be published to deploy.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /work
COPY rf-knots /work/rf-knots
COPY pgx-mcts-bench /work/pgx-mcts-bench

WORKDIR /work/pgx-mcts-bench
# Resolve into the image so a container start is not also a dependency install.
RUN uv sync --extra dev --python 3.12

# Artifacts are a bind mount, not a layer: checkpoints have to outlive the
# container, and a run that loses its ladder on restart cannot be resumed.
VOLUME ["/work/pgx-mcts-bench/artifacts"]

ENTRYPOINT ["uv", "run", "pgx-mcts-bench"]
CMD ["--help"]
