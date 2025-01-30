# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app
#RUN pip3 install spacy telethon urlextract pandas

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt 

COPY . /app

FROM builder as dev-envs
