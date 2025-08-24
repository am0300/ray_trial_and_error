FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# 必須 OS パッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    vim \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# home directoryの変更
WORKDIR /app

# uvを使ってpythonパッケージをインストール
RUN uv init ./
RUN uv add ray
RUN uv add ray[rllib]
RUN uv add ray[tune]
RUN uv add ray[train]
RUN uv add ray[data]
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN uv add tensorboard
RUN uv add opencv-python
RUN uv add numba
RUN uv add mypy
RUN uv add ruff
RUN uv add open_spiel
RUN uv add pygame

CMD [ "bash" ]
