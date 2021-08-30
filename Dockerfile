FROM python:3.8

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8502

COPY . /app

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]