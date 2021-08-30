FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8502

COPY . /app

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]