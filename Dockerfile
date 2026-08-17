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
    gnupg \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# tileserver-gl (the self-hosted basemap tile renderer, see
# scripts/gis/build_basemap_tiles.py) runs in-process inside this same
# image/container rather than as a separate service — see
# docker/entrypoint.sh's _start_tileserver. It's a completely different
# runtime (Node.js + native MapLibre GL rendering) from everything else
# here, so it needs its own apt/npm install: the package list below is
# tileserver-gl's own official Dockerfile's dependency set (both its
# build-time and runtime libs, since we build+run in one image) —
# https://github.com/maptiler/tileserver-gl/blob/master/Dockerfile.
# Pin the Node major version (nodesource) and the tileserver-gl npm version
# together deliberately, same reasoning as the GDAL/uv image pins above —
# bump both together, not independently.
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    nodejs \
    pkg-config \
    xvfb \
    libglfw3-dev \
    libuv1-dev \
    libcairo2-dev \
    libpango1.0-dev \
    libpng-dev \
    libjpeg-dev \
    libgif-dev \
    librsvg2-dev \
    libcurl4-openssl-dev \
    libicu-dev \
    libopengl0 \
 && rm -rf /var/lib/apt/lists/*

# npm resolves a prebuilt `canvas` binary if one matches this platform/Node
# ABI; otherwise node-gyp compiles it from source using the -dev headers
# just installed above — either way this needs no extra flag.
RUN npm install -g tileserver-gl@5.6.0

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
