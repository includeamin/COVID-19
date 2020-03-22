# UNDER CHANGE DO NOT USE
#FROM python:latest
#<<<<<<< HEAD
#FROM covid_base:v2
FROM m.docker-registry.ir/tensorflow/tensorflow:latest-py3
EXPOSE 8000
#WORKDIR app
#COPY requirements.txt requirements.txt
COPY . .

RUN pip3 install -r requirements.txt

#CMD ["uvicorn","Server:app"]
#FROM registry.api.chichiapp.ir:4443/covid:v1
#RUN ls
#WORKDIR app
RUN pip3 install gunicorn
CMD ["python3" , "Server.py"]
#CMD  ["uvicorn" ,"Server:app"]
#CMD ["gunicorn" , "Server:app","-w","2", "-k", "uvicorn.workers.UvicornWorker","--bind","0.0.0.0:8000","--log-level","debug","--threads=2"]
#RUN mv Server.py app.py
#CMD ["uvicorn" , "app:app" , "--host" ,"0.0.0.0" ,"--port" ,"8000", "--log-level" , "trace"]
#=======
#
#EXPOSE 8000
#WORKDIR app
#COPY . .
#
#RUN pip install -r requirements.txt
#
#CMD ["uvicorn","Server:app"]
#
FROM registry.api.chichiapp.ir:4443/covid:v1

CMD['python3','Server.py']
>>>>>>> 44fd15efd82ec164f0b5bd6e19ad5926288d7a5a
