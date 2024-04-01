FROM python:3.9.9-bullseye

WORKDIR /src

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY scripts ./scripts/
COPY training_data ./training_data/
COPY ./config.yaml ./


ENTRYPOINT ["python3", "-u", "./scripts/main.py"]
