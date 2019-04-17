FROM willprice/opencv4:cuda10.1-cudnn7
LABEL maintainer="will.price94+docker@gmail.com"

RUN mkdir /src
WORKDIR /src
ADD . /src

RUN make build
RUN make install
