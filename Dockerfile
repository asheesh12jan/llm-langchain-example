FROM python:3.11.8-slim-bullseye

WORKDIR /

COPY requirements.txt /requirements.txt

RUN python3 -m pip install -r /requirements.txt --default-timeout=1000


# Copy app scripts
COPY app/ /opt/app

WORKDIR /opt/app


CMD [ "python3", "/opt/app/app.py" ]
