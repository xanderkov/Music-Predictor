FROM ubuntu:24.04 as build

RUN apt-get update

RUN apt-get update && apt-get install -y python3.12 python3.12-venv python3-pip
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


RUN pip install --upgrade pip
RUN pip install poetry


FROM build as prod

WORKDIR /app
COPY ./pyproject.toml /app/pyproject.toml
COPY . /app/
RUN poetry install

FROM prod as test
WORKDIR /app
RUN poetry install --with=test

FROM test as dev
WORKDIR /app

RUN poetry install --all-extras