FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only what uv needs to resolve deps (cache layer)
COPY pyproject.toml README.md LICENSE ./

# Install package dependencies (no source yet — cached layer)
RUN uv pip install --system --no-cache ".[docs]" 2>/dev/null || true

# Copy source and install the package itself
COPY src/ ./src/

RUN uv pip install --system --no-cache .

# Non-root runtime user
RUN useradd --no-create-home --shell /bin/false app
USER app

# No home dir for this user; point matplotlib's cache somewhere writable
ENV MPLCONFIGDIR=/tmp/matplotlib

ENTRYPOINT ["metasignal"]
CMD ["--help"]
