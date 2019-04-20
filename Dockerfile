FROM willprice/opencv4:cuda-10.1-cudnn7
LABEL maintainer="will.price94+docker@gmail.com"

RUN apt-get update && \
    apt-get install -y python3 python3-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /src
WORKDIR /src

RUN python3 -m venv .venv
RUN . .venv/bin/activate
ADD dev-requirements.txt requirements.txt /src/
RUN python3 -m pip install -r dev-requirements.txt
ADD . /src/
RUN make build
RUN make install
ENTRYPOINT /usr/local/bin/flowty
