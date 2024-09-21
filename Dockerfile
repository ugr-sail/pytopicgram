# syntax=docker/dockerfile:1.4
FROM python:3.12-slim AS builder

EXPOSE 5000
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY . /app


CMD ["bash"]
