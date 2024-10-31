ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /app
CMD ["python", "setup.py", "--help-commands"]
