FROM ubuntu:16.04

ENV PYTHON_VERSION=3.7.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    zlib1g-dev \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make altinstall && \
    cd ..

RUN ln -s /usr/local/bin/python3.7 /usr/bin/python3

COPY . .
ENV PYTHONPATH=/Regression
ENV TFDS_DATA_DIR=/tensorflow_datasets
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r ./requirements.txt

RUN python3.7 /Regression/datasets.py

WORKDIR /Regression/service
CMD python3.7 service.py

EXPOSE 8081

