FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip build-essential && apt-get clean
WORKDIR /workspace
COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python3.9 -m pip install --no-deps .
ENTRYPOINT ["colonoscopy-analysis"]
