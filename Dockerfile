ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Install build dependencies
ARG DEBIAN_MIRROR=""
RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s|http://deb.debian.org/debian|$DEBIAN_MIRROR/debian|g; s|http://deb.debian.org/debian-security|$DEBIAN_MIRROR/debian-security|g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install ta-lib C library
# Source: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --build="$(uname -m)-unknown-linux-gnu" --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies. CPU_ONLY is for portable local smoke tests; the
# competition/default path remains the frozen CUDA lock.
ARG CPU_ONLY=0
RUN if [ "$CPU_ONLY" = "1" ]; then \
        uv venv .venv && \
        uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.10.0" && \
        uv pip install --python .venv/bin/python --default-index https://pypi.org/simple \
          akshare==1.18.28 baostock==0.8.9 docker==7.1.0 joblib==1.5.2 \
          matplotlib==3.10.6 pandas==2.3.2 scikit-learn==1.7.2 seaborn==0.13.2 \
          ta-lib==0.6.8 tensorboard==2.20.0 tensorboardx==2.6.4 tqdm==4.67.1; \
    else \
        uv sync --frozen; \
    fi

# Copy the application code
COPY . .

# Set environment to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# Keep container running idle; execute train/predict manually via docker exec.
CMD ["sleep", "infinity"]
