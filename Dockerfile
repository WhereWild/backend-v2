FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    less \
    htop \
    bash-completion \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN echo '\nif [ -f /usr/share/bash-completion/bash_completion ]; then\n  . /usr/share/bash-completion/bash_completion\nfi' >> /etc/bash.bashrc \
 && git config --global --add safe.directory /workspace

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

RUN echo '\n[ -f /etc/wherewild_aliases.sh ] && . /etc/wherewild_aliases.sh' >> /etc/bash.bashrc
COPY docker/aliases.sh /etc/wherewild_aliases.sh

CMD ["bash"]
