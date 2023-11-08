FROM python:3.11-slim-bookworm

RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["src/predict.py", "models/xgb_model.bin", "./"]

EXPOSE 9595

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
