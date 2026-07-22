FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# 基础工具
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    gcc g++ make wget tar \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 安装 TA-Lib C 库
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 工作目录
WORKDIR /app

# 依赖文件
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# 代码
COPY . .

# 虚拟环境路径
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# 赛事方评测时执行 data/run.sh（init + train + test）
CMD ["sleep", "infinity"]
