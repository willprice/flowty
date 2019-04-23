FROM willprice/opencv4:cuda-10.1-cudnn7
LABEL maintainer="will.price94+docker@gmail.com"

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir /src
WORKDIR /src
ADD dev-requirements.txt requirements.txt /src/
RUN python3 -m pip install -r dev-requirements.txt
ADD . /src/
RUN make install
ENTRYPOINT ["/usr/local/bin/flowty"]
CMD ["--help"]
