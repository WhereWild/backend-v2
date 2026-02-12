FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

WORKDIR /app

RUN sed -i 's|archive.ubuntu.com|us.archive.ubuntu.com|g' /etc/apt/sources.list

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
        bash-completion \
    fuse \
    psmisc \
    rclone \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

COPY docker/aliases.sh /etc/wherewild_aliases.sh
RUN echo '\n# WhereWild dev aliases\n[ -f /etc/wherewild_aliases.sh ] && . /etc/wherewild_aliases.sh' >> /etc/bash.bashrc \
    && echo '\n# Enable bash completion\nif [ -f /usr/share/bash-completion/bash_completion ]; then\n  . /usr/share/bash-completion/bash_completion\nfi' >> /etc/bash.bashrc

COPY docker/entrypoint.sh /usr/local/bin/wherewild-entrypoint
RUN chmod +x /usr/local/bin/wherewild-entrypoint
ENTRYPOINT ["/usr/local/bin/wherewild-entrypoint"]
CMD ["bash"]
