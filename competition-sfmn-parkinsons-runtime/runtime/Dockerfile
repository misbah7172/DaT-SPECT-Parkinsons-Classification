# Base image with uv and all Python dependencies
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Create runtime user
ENV RUNTIME_USER=runtimeuser
ENV RUNTIME_UID=1000
ENV RUNTIME_GID=1000

RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_GID} ${RUNTIME_USER} \
    && useradd --create-home --gid ${RUNTIME_GID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER}

# Install system dependencies (no Python - managed by uv)
RUN apt-get update --fix-missing \
    && apt-get install -y \
        -o Dpkg::Options::="--force-confdef" \
        -o Dpkg::Options::="--force-confold" \
        libopenblas0 \
        libgomp1 \
        build-essential \
        cmake \
        g++ \
        gcc \
        git \
        libxml2 \
        tzdata \
        wget \
        zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up code execution working directory
RUN mkdir /code_execution && \
    chown -R ${RUNTIME_USER}:${RUNTIME_USER} /code_execution
WORKDIR /code_execution

# Switch to runtime user for Python package installation
USER ${RUNTIME_USER}

# Copy lockfile, pyproject.toml, and .python-version first
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} uv.lock pyproject.toml .python-version ./

# uv environment variables
ENV UV_FROZEN=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set up virtual environment path
ENV VIRTUAL_ENV=/code_execution/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python and dependencies (no dev/test deps)
ENV UV_CACHE_DIR=/home/runtimeuser/.cache/uv
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev


# Copy entrypoint
COPY --chmod=755 entrypoint.sh /entrypoint.sh
USER ${RUNTIME_USER}

# Runtime image
FROM base AS runtime
ENV UV_NO_SYNC=1
CMD ["bash", "/entrypoint.sh"]

# Test image
FROM base AS test-runtime

# Install test dependencies
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra test

USER root
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} tests ./tests
USER ${RUNTIME_USER}

ENV UV_NO_SYNC=1
CMD ["python", "-m", "pytest", "tests", "-v"]