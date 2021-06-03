#FROM ubuntu:20.04
#FROM nvidia/cuda:10.2-cudnn7-devel AS builder
FROM nvidia/cuda:11.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive 

# Add user
RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

# [Option] Install Node.js
#ARG INSTALL_NODE="true"
#ARG NODE_VERSION="lts/*"
#RUN if [ "${INSTALL_NODE}" = "true" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1
ENV CUDNN_VERSION 8.1.0.77

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.2 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.2 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

# Requirements
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pyqt5 \
    build-essential \
    python3-dev \
    python3-pip 

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

#RUN python3 -m pip install opencv-python

#COPY train.py /tmp/main.py
#COPY gen_data.py /tmp/main.py
COPY verification_ann.py /tmp/main.py
COPY /files /files
COPY /media /media
COPY /results /results
ENTRYPOINT ["/bin/bash", "-c", "python3 /tmp/main.py"]
