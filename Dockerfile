# Dockerfile, Image, Container
FROM python:3.11

ENV WANDB_API_KEY=default_value

ADD main.py .
ADD GLUE_Transformer.py .
ADD requirements.txt . 

RUN pip install -r requirements.txt

CMD ["python", "./main.py"]