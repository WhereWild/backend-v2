FROM ghcr.io/osgeo/gdal:ubuntu-full-latest

WORKDIR /workspace

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY docker/aliases.sh /etc/wherewild_aliases.sh
RUN echo '\n# WhereWild dev aliases\n[ -f /etc/wherewild_aliases.sh ] && . /etc/wherewild_aliases.sh' >> /etc/bash.bashrc

CMD ["tail", "-f", "/dev/null"]
