# FROM python:3.8

# WORKDIR /app

# COPY requirements.txt ./requirements.txt

# RUN pip install -r requirements.txt

# EXPOSE 8501

# COPY . /app

# ENTRYPOINT ["streamlit","run"]
# CMD ["main.py"]


FROM python:3.8

WORKDIR /Face-Mask_Inpainting

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /Face-Mask_Inpainting

ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]