FROM python:3.8.2-buster

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY ./app /app