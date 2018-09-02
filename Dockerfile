FROM ubuntu:18.10
LABEL maintainer="cesliens@gmail.com"

RUN apt-get update && \
    apt-get install python3-pip python3-dev -y && \
    pip3 install -U tensorflow && \
    pip3 install keras && \
    pip3 install sklearn
