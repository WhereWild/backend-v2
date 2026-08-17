# SPDX-FileCopyrightText: 2025-2026 The WhereWild Contributors (see CONTRIBUTORS)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Pinned by digest (not :latest) so local/CI builds don't get cache-blasted
# and reproducibility doesn't silently drift whenever upstream repushes their
# floating tag. Bump deliberately: `docker buildx imagetools inspect
# ghcr.io/osgeo/gdal:ubuntu-full-latest` / `...astral-sh/uv:latest` for the
# current digest.
FROM ghcr.io/osgeo/gdal:ubuntu-full-latest@sha256:323828a57fd01e2f0a96ece1b2caf6b4ad41e2e47458386836697418fd67665c

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:2d890623d310b57771ce840f0da5eed5fc6d657da05ffaa45d82797b53fa3abc /uv /usr/local/bin/uv

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
    openjdk-21-jre-headless \
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
