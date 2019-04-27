FROM willprice/opencv4:cuda-10.1-cudnn7
LABEL maintainer="will.price94+docker@gmail.com"

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /src
WORKDIR /src
RUN python3 -m pip install Cython numpy pytest
ADD . /src/
RUN make install
ENTRYPOINT ["/usr/local/bin/flowty"]
CMD ["--help"]
