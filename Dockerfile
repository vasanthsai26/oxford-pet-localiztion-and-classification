FROM python:3.10

RUN mkdir objloc
WORKDIR /objloc

COPY requirements.txt /objloc/
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn"  , "-b", "0.0.0.0:8888", "app:app"] 
