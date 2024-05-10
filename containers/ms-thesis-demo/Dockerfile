# Local Usage:
# ```
# docker build -t ghcr.io/bouncmpe/cuda-python3 containers/cuda-python3/
# docker run -it --rm --gpus=all ghcr.io/bouncmpe/cuda-python3
# ```

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        git \
        wget \
        cmake \
        ninja-build \
        build-essential \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* 

RUN python3 -m pip install --upgrade pip && \
    python3 -m venv /opt/python3/venv/base

COPY requirements.txt /opt/python3/venv/base/
RUN /opt/python3/venv/base/bin/python3 -m pip install --no-cache-dir -r /opt/python3/venv/base/requirements.txt

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

COPY ms-thesis-demo .
# Set entrypoint to bash
ENTRYPOINT ["/entrypoint.sh"]
