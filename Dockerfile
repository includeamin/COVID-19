#FROM python:latest
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
