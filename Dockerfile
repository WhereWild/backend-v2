# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

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
    python3-venv \
    rclone \
 && rm -rf /var/lib/apt/lists/*

RUN echo '\nif [ -f /usr/share/bash-completion/bash_completion ]; then\n  . /usr/share/bash-completion/bash_completion\nfi' >> /etc/bash.bashrc \
 && git config --global --add safe.directory /workspace

RUN mkdir -p /opt/venvs && chmod 777 /opt/venvs

ENV UV_PROJECT_ENVIRONMENT=/opt/venvs/venv
ENV UV_LINK_MODE=copy
# Store uv-managed Python under /opt so it's in the image layer and not in the
# root-home volume (/root/.local), which non-root container users can't traverse.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
RUN mkdir -p /opt/uv-python \
 && uv sync --frozen --no-install-project \
 && chmod -R a+rx /opt/uv-python \
 && chmod -R a+rwx /opt/venvs

COPY . .

RUN echo '\n[ -f /etc/wherewild_aliases.sh ] && . /etc/wherewild_aliases.sh' >> /etc/bash.bashrc
COPY docker/aliases.sh /etc/wherewild_aliases.sh

COPY docker/entrypoint.sh /usr/local/bin/wherewild-entrypoint
RUN chmod +x /usr/local/bin/wherewild-entrypoint
ENTRYPOINT ["/usr/local/bin/wherewild-entrypoint"]
CMD ["bash"]
