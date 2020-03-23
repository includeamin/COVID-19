#FROM ubuntu:18.04
#FROM covid_base:v1
#EXPOSE 8000
#RUN   export LC_ALL=C.UTF-8
#RUN   export LANG=C.UTF-8
#RUN apt update
#RUN apt install -y python3
#RUN apt-get install -y python3-pip
#RUN apt update && apt install -y libsm6 libxext6
#RUN apt-get install -y libsm6 libxrender1 libfontconfig1
FROM registry.api.chichiapp.ir:4443/chichi/covid_base:4443
WORKDIR app
RUN   export LC_ALL=C.UTF-8
RUN   export LANG=C.UTF-8
COPY . .
#RUN pip3 install -r requirements.txt
RUN pip3 install h5py
RUN pip3 install pybadges
CMD ["uvicorn" , "Server:app" , "--host" ,"0.0.0.0" ,"--port" ,"8000", "--log-level" , "debug"]

