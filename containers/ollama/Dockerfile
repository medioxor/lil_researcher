FROM ubuntu:24.04

RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/au.archive.ubuntu.com\/ubuntu\//g' /etc/apt/sources.list.d/ubuntu.sources

RUN apt update && apt upgrade -y && apt install -y curl

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN mkdir /models

COPY ./containers/ollama/entry.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]