FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

# Runtime libraries needed by the stormpy wheel that are not part of the slim image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/premise

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install the dependencies first (stormpy ships as a binary wheel), so this layer is cached.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Install premise itself.
COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/opt/premise/.venv/bin:$PATH"

CMD ["bash"]
