FROM ubuntu:24.04

RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/au.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt update && apt upgrade -y && apt install -y \
    gcc \
    musl-dev \
    libx11-dev \
    libxkbcommon-dev \
    chromium-browser \
    libnss3 \
    libfreetype6 \
    libharfbuzz0b \
    fonts-freefont-ttf \
    nodejs \
    npm \
    python3-pip \
    python3-venv \
    vim \
    curl \
    wget \
    git \
    iputils-ping

ENV VIRTUAL_ENV=/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN mkdir /src

COPY ./src/requirements.txt /src

RUN --mount=type=cache,id=pip,target=/root/.cache/pip cd /src && pip3 install -r requirements.txt

RUN playwright install --with-deps chromium

WORKDIR /src

CMD ["/bin/bash"]