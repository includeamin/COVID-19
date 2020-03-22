FROM ubuntu:18.04
EXPOSE 8000
RUN   export LC_ALL=C.UTF-8
RUN   export LANG=C.UTF-8
RUN apt update
RUN apt install -y python3
RUN apt-get install -y python3-pip
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libsm6 libxrender1 libfontconfig1
WORKDIR app
COPY requirements.txt .requirements.txt
