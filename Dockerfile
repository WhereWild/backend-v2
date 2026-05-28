FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    aria2 \
    git \
    curl \
    wget \
    vim \
    less \
    htop \
    bash-completion \
    build-essential \
    fuse \
    psmisc \
    rclone \
 && rm -rf /var/lib/apt/lists/*

RUN echo '\nif [ -f /usr/share/bash-completion/bash_completion ]; then\n  . /usr/share/bash-completion/bash_completion\nfi' >> /etc/bash.bashrc \
 && git config --global --add safe.directory /workspace

RUN mkdir -p /opt/venvs && chmod 777 /opt/venvs

ENV UV_PROJECT_ENVIRONMENT=/opt/venvs/venv
ENV UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project \
 && chmod -R a+w /opt/venvs

COPY . .

RUN echo '\n[ -f /etc/wherewild_aliases.sh ] && . /etc/wherewild_aliases.sh' >> /etc/bash.bashrc
COPY docker/aliases.sh /etc/wherewild_aliases.sh

COPY docker/entrypoint.sh /usr/local/bin/wherewild-entrypoint
RUN chmod +x /usr/local/bin/wherewild-entrypoint
ENTRYPOINT ["/usr/local/bin/wherewild-entrypoint"]
CMD ["bash"]
