FROM tensorflow/tensorflow:2.10.1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

ENV PORT=8080
CMD ["gunicorn", "-b", "0.0.0.0:8080", "api:app"]

