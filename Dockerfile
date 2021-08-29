FROM python:3.8

WORKDIR /app

RUN pip install -U pip

COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app

ENTRYPOINT ["streamlit","run"]

CMD ["main.py"]